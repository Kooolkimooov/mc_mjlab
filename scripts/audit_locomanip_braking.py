"""Compare braking costs on reward-bearing live motion-probe traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.motion_reward import score_braking


def main() -> None:
  """Write discounted reward comparisons without altering measured trajectories."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("runs", type=Path, nargs="+")
  parser.add_argument(
    "--costs", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 1.0, 1.1]
  )
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  reports = []
  for run in args.runs:
    manifest = json.loads((run / "report.json").read_text())
    frames = [
      json.loads(line) for line in (run / "trace.jsonl").read_text().splitlines()
    ]
    scored = score_braking(
      frames, step_dt=manifest["step_dt"], gamma=manifest["gamma"], costs=args.costs
    )
    scored.update(
      run=str(run.resolve()),
      mass_kg=manifest["mass_kg"],
      seed=manifest["seed"],
      encoder_bias=manifest["encoder_bias"],
      provenance=manifest["provenance"],
    )
    reports.append(scored)
    print(f"{manifest['mass_kg']:g} kg, decision at {scored['decision_time_s']:.2f} s")
    for arm, values in scored["arms"].items():
      returns = ", ".join(
        f"c={c:g}: {values['returns'][str(c)]['discounted']:.4f}" for c in args.costs
      )
      print(
        f"  {arm}: complete={values['complete']}, failed={values['failed']}; {returns}"
      )
      if "cost_at_equal_baseline_return" in values:
        print(
          f"    baseline break-even cost: {values['cost_at_equal_baseline_return']}"
        )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
  main()
