"""Check the registered motion reward profile against controlled rollout records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mjlab  # noqa: F401
from evaluation.motion_preferences import PROFILE_TERMS, check_motion_preferences
from mjlab.tasks.registry import load_env_cfg

from mc_mjlab.tasks.locomanip_motion import TASK_IDS


def main() -> None:
  """Write explicit preference margins for the live registered reward weights."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("runs", type=Path, nargs="+")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--minimum-margin", type=float, default=0.05)
  args = parser.parse_args()
  cfg = load_env_cfg(TASK_IDS["HRP5P"])
  weights = {name: cfg.rewards[name].weight for name in PROFILE_TERMS}
  report = check_motion_preferences(args.runs, weights, args.minimum_margin)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
  print(
    json.dumps(
      {
        key: value
        for key, value in report.items()
        if key not in ("outcomes", "comparisons")
      },
      indent=2,
    )
  )
  for row in report["comparisons"]:
    if not row["passed"]:
      print(json.dumps(row))
  if not report["all_preferences_satisfied"]:
    raise SystemExit(1)


if __name__ == "__main__":
  main()
