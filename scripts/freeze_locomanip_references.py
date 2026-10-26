"""Inventory study checkpoints and controller inputs without starting workers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import mjlab  # noqa: F401
import torch
from mjlab.tasks.registry import load_env_cfg

from mc_mjlab.rl.controller_provenance import (
  collect_controller_provenance,
  provenance_matches,
)
from mc_mjlab.tasks.locomanip_study import CONTROL_TASK_ID

REFERENCES = {
  "original": ("2026-10-21_10-52-44_push-friction", "model_1180.pt"),
  "no_magnitude": ("2026-10-22_10-00-47_no-magnitude-penalty", "model_1660.pt"),
  "arms_2x": ("2026-10-23_17-39-18_arm-scale-2x", "model_1280.pt"),
}


def file_identity(path: Path) -> dict:
  """Hash exact artifact bytes without storing checkpoint contents in Git."""
  if not path.is_file():
    return {"path": str(path.resolve()), "present": False}
  with path.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
  return {"path": str(path.resolve()), "present": True, "sha256": digest}


def revision(path: Path) -> dict:
  """Record a checkout revision and tracked worktree changes separately."""

  def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

  return {
    "path": str(path.resolve()),
    "head": git("rev-parse", "HEAD"),
    "status": git("status", "--short", "--untracked-files=no"),
  }


def main() -> None:
  """Write reproducible identities and distinguish provenance from load validation."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  cfg = load_env_cfg(CONTROL_TASK_ID)
  provenance = collect_controller_provenance(
    SimpleNamespace(unwrapped=SimpleNamespace(cfg=cfg))
  )
  root = Path(__file__).resolve().parents[1]
  runs = (
    root / "logs/rsl_rl/Mc-Mjlab-Locomanip-Locomanipcontroller-Hrp5P-Position-Residual"
  )
  records = {}
  for name, (run, checkpoint) in REFERENCES.items():
    path = runs / run / checkpoint
    record = file_identity(path)
    if path.is_file():
      saved = torch.load(path, map_location="cpu", weights_only=False)
      infos = saved.get("infos") or {}
      previous = infos.get("base_controller_provenance")
      manifest = infos.get("effective_training_manifest", {})
      record.update(
        provenance_matches=provenance_matches(previous, provenance)
        if previous
        else None,
        load_validation="pending",
        manifest_hashes={
          key: value for key, value in manifest.items() if key.endswith("sha256")
        },
        inputs=[
          file_identity(runs / run / "params" / file)
          for file in ("env.yaml", "agent.yaml")
        ],
      )
    records[name] = record
  result = {
    "repository": revision(root),
    "mjlab": revision(Path(mjlab.__file__).resolve().parents[2]),
    "packages": {
      name: importlib.metadata.version(name)
      for name in ("mjlab", "torch", "mujoco", "mujoco-warp", "rsl-rl-lib")
    },
    "zero": {"task": CONTROL_TASK_ID, "action": "exact zero", "checkpoint": None},
    "controller_provenance": provenance,
    "references": records,
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(records, indent=2))


if __name__ == "__main__":
  main()
