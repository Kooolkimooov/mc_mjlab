"""Screen scheduled checkpoints, then validate one safe-mass candidate off seed."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

SCREEN_STEPS = (200, 400, 600, 800, 999)


def select_candidate(reports: dict[int, dict]) -> int | None:
  """Choose the strongest pooled reduction among complete, safe-mass-protected screens."""
  eligible = [
    (step, report["pooled"]["relative_reduction"])
    for step, report in reports.items()
    if report["complete_matched_data"]
    and report["per_mass_protection"]
    and report["pooled"] is not None
    and report["pooled"]["relative_reduction"] is not None
  ]
  return max(eligible, key=lambda item: item[1])[0] if eligible else None


def main() -> None:
  """Write every screen and the selected checkpoint's separate 32-pair validation."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-dir", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--workers", type=int, default=4)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  reports = {}
  for step in SCREEN_STEPS:
    checkpoint = args.run_dir / f"model_{step}.pt"
    if not checkpoint.is_file():
      raise FileNotFoundError(checkpoint)
    output = args.output / f"screen-{step}"
    subprocess.run(
      [
        "uv",
        "run",
        "python",
        "scripts/evaluate_locomanip_safety.py",
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--workers",
        str(args.workers),
        "--device",
        args.device,
      ],
      check=True,
    )
    reports[step] = json.loads((output / "report.json").read_text())
  selected = select_candidate(reports)
  selection = {
    "screen_steps": SCREEN_STEPS,
    "selected_step": selected,
    "reason": "highest pooled relative reduction among screens with complete pairs and no safe-mass regression",
  }
  (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
  if selected is None:
    print("No checkpoint passed the safe-mass screen; validation was not started.")
    return
  subprocess.run(
    [
      "uv",
      "run",
      "python",
      "scripts/evaluate_locomanip_safety.py",
      "--checkpoint",
      str(args.run_dir / f"model_{selected}.pt"),
      "--output",
      str(args.output / "validation"),
      "--validation",
      "--workers",
      str(args.workers),
      "--device",
      args.device,
    ],
    check=True,
  )


if __name__ == "__main__":
  main()
