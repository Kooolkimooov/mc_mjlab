"""Runner that makes the external base-controller configuration reproducible."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from mjlab.rl import MjlabOnPolicyRunner

from mc_mjlab.utils.mc_rtc_config import get_controller_name


def _file_record(role: str, path: Path) -> dict[str, str | bool]:
  """Capture one provenance file as content plus a stable digest."""
  path = path.expanduser().resolve()
  if not path.is_file():
    return {"role": role, "source": str(path), "present": False}
  content = path.read_text(errors="replace")
  return {
    "role": role,
    "source": str(path),
    "present": True,
    "sha256": hashlib.sha256(content.encode()).hexdigest(),
    "content": content,
  }


def _controller_config_paths(controller_name: str) -> list[Path]:
  """Find installed mc_rtc YAML files that configure the selected controller."""
  paths: set[Path] = set()
  for value in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
    if not value:
      continue
    library_dir = Path(value).expanduser()
    for suffix in ("yaml", "yml"):
      paths.update(library_dir.glob(f"*/etc/{controller_name}.{suffix}"))
  return sorted(path.resolve() for path in paths if path.is_file())


def collect_controller_provenance(env) -> dict:
  """Collect all non-checkpoint inputs that define the base controller."""
  action_cfg = env.unwrapped.cfg.actions["mc_rtc_residual"]
  project_cfg = Path(action_cfg.mc_rtc_config_path)
  controller_name = get_controller_name(project_cfg)
  records = [
    _file_record("project_mc_rtc", project_cfg),
    _file_record("user_mc_rtc", Path.home() / ".config/mc_rtc/mc_rtc.yaml"),
  ]
  if action_cfg.pd_gains_path is not None:
    records.append(_file_record("pd_gains", Path(action_cfg.pd_gains_path)))
  records.extend(
    _file_record(f"controller_config_{index}", path)
    for index, path in enumerate(_controller_config_paths(controller_name))
  )
  return {"controller_name": controller_name, "files": records}


def _materialize_provenance(provenance: dict, log_dir: Path) -> None:
  """Write readable copies alongside the run's ordinary parameter dump."""
  output_dir = log_dir / "base_controller_config"
  output_dir.mkdir(parents=True, exist_ok=True)
  manifest = {"controller_name": provenance["controller_name"], "files": []}
  for index, record in enumerate(provenance["files"]):
    public_record = {key: value for key, value in record.items() if key != "content"}
    if record["present"]:
      source = Path(record["source"])
      snapshot = f"{index:02d}_{record['role']}{source.suffix}"
      shutil.copy2(source, output_dir / snapshot)
      public_record["snapshot"] = snapshot
    manifest["files"].append(public_record)
  (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _provenance_signature(provenance: dict) -> tuple:
  """Reduce provenance to the values that affect reproducibility."""
  files = tuple(
    (record["role"], record["present"], record.get("sha256"))
    for record in provenance["files"]
  )
  return provenance["controller_name"], files


class ResidualBalanceOnPolicyRunner(MjlabOnPolicyRunner):
  """Persist and validate the base-controller files used by a residual run."""

  PROVENANCE_KEY = "base_controller_provenance"

  def __init__(self, env, train_cfg: dict, log_dir=None, device: str = "cpu") -> None:
    super().__init__(env, train_cfg, log_dir, device)
    self._controller_provenance = collect_controller_provenance(env)
    if log_dir is not None and int(os.environ.get("RANK", "0")) == 0:
      _materialize_provenance(self._controller_provenance, Path(log_dir))

  def save(self, path: str, infos=None) -> None:
    """Embed controller inputs in every checkpoint as well as the run directory."""
    infos = {**(infos or {}), self.PROVENANCE_KEY: self._controller_provenance}
    super().save(path, infos)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Reject a checkpoint when its recorded base controller is different."""
    infos = super().load(path, load_cfg, strict, map_location)
    saved = (infos or {}).get(self.PROVENANCE_KEY)
    if saved is None:
      print(f"[mc_mjlab] checkpoint {path} predates base-controller provenance")
      return infos
    if _provenance_signature(saved) != _provenance_signature(
      self._controller_provenance
    ):
      raise RuntimeError(
        "Checkpoint base-controller configuration differs from the active "
        "configuration. Restore the YAML/PD files embedded under infos/"
        f"{self.PROVENANCE_KEY} before loading {path}."
      )
    return infos
