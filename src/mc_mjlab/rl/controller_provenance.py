"""Capture the external mc_rtc files that define a run's base controller."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import cast

from mjlab.rl import RslRlVecEnvWrapper

from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionCfg
from mc_mjlab.bridge.config import get_controller_name
from mc_mjlab.utils.controller_install import controller_config_paths


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


def collect_controller_provenance(env: RslRlVecEnvWrapper) -> dict:
  """Collect all non-checkpoint inputs that define the base controller."""
  action_cfg = cast(
    McRtcResidualActionCfg, env.unwrapped.cfg.actions["mc_rtc_residual"]
  )
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
    for index, path in enumerate(controller_config_paths(controller_name))
  )
  return {"controller_name": controller_name, "files": records}


def materialize_provenance(provenance: dict, log_dir: Path) -> None:
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


def provenance_signature(provenance: dict) -> tuple:
  """Reduce provenance to the values that affect reproducibility."""
  files = tuple(
    (record["role"], record["present"], record.get("sha256"))
    for record in provenance["files"]
  )
  return provenance["controller_name"], files
