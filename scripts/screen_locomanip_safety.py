"""Screen scheduled checkpoints, then validate one safe-mass candidate off seed."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

SCREEN_MILESTONES = (200, 400, 600, 800)
EVALUATOR = Path(__file__).resolve().with_name("evaluate_locomanip_safety.py")
REPO_ROOT = EVALUATOR.parent.parent


def checkpoint_steps(run_dir: Path) -> tuple[int, ...]:
  """Check the final checkpoint and collect available scheduled milestones."""
  agent = yaml.safe_load((run_dir / "params" / "agent.yaml").read_text())
  available = {
    int(path.stem.removeprefix("model_"))
    for path in run_dir.glob("model_*.pt")
    if path.stem.removeprefix("model_").isdigit()
  }
  if not available:
    raise FileNotFoundError(f"no checkpoints under {run_dir}")
  final = max(available)
  if final < agent["max_iterations"] - 1:
    raise ValueError(f"final checkpoint is missing from {run_dir}")
  milestones = [step for step in SCREEN_MILESTONES if step in available]
  return tuple(dict.fromkeys([*milestones, final]))


def select_candidate(reports: dict[int, dict]) -> int | None:
  """Choose the strongest improving screen with no per-mass regression."""
  eligible = [
    (step, report["pooled"]["relative_reduction"])
    for step, report in reports.items()
    if report["complete_matched_data"]
    and report["per_mass_protection"]
    and report["no_mass_regression"]
    and report["pooled"] is not None
    and report["pooled"]["relative_reduction"] is not None
    and report["pooled"]["relative_reduction"] > 0
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
  args.run_dir = args.run_dir.resolve()
  args.output = args.output.resolve()
  steps = checkpoint_steps(args.run_dir)
  args.output.mkdir(parents=True, exist_ok=True)
  reports = {}
  for step in steps:
    checkpoint = args.run_dir / f"model_{step}.pt"
    if not checkpoint.is_file():
      raise FileNotFoundError(checkpoint)
    output = args.output / f"screen-{step}"
    subprocess.run(
      [
        "uv",
        "run",
        "python",
        str(EVALUATOR),
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
      cwd=REPO_ROOT,
    )
    reports[step] = json.loads((output / "report.json").read_text())
  selected = select_candidate(reports)
  selection = {
    "screen_steps": steps,
    "selected_step": selected,
    "reason": (
      "highest positive pooled relative reduction among complete screens "
      "without a per-mass regression"
    ),
  }
  (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
  if selected is None:
    print("No checkpoint improved safely; validation was not started.")
    return
  subprocess.run(
    [
      "uv",
      "run",
      "python",
      str(EVALUATOR),
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
    cwd=REPO_ROOT,
  )


if __name__ == "__main__":
  main()
