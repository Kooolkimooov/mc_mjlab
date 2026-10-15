"""Compare braking costs on reward-bearing live motion-probe traces."""

from __future__ import annotations

import argparse
import json
import math
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
  parser.add_argument("--event-costs", type=float, nargs="+", default=[0.0])
  parser.add_argument("--holding-costs", type=float, nargs="+", default=[0.0])
  parser.add_argument(
    "--reward-weight", type=_parse_weight, action="append", default=[]
  )
  args = parser.parse_args()
  reports = []
  for run in args.runs:
    manifest = json.loads((run / "report.json").read_text())
    frames = [
      json.loads(line) for line in (run / "trace.jsonl").read_text().splitlines()
    ]
    scales = {}
    for name, value in args.reward_weight:
      captured = manifest["reward_weights"].get(name)
      if captured is None:
        parser.error(f"reward term {name} was not captured")
      if captured == 0:
        parser.error(f"cannot rescale zero-weight term {name}")
      scales[name] = value / captured
    scored = score_braking(
      frames,
      step_dt=manifest["step_dt"],
      gamma=manifest["gamma"],
      costs=args.costs,
      event_costs=args.event_costs,
      holding_costs=args.holding_costs,
      reward_scales=scales,
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
      if args.event_costs != [0.0]:
        print(
          f"    discounted braking events: {values['discounted_braking_events']:.4f}"
        )
        print(f"    returns with event costs: {values['event_returns']}")
      if args.holding_costs != [0.0]:
        print(
          f"    discounted held seconds: {values['discounted_holding_seconds']:.4f}"
        )
        print(f"    returns with holding costs: {values['holding_returns']}")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")


def _parse_weight(value: str) -> tuple[str, float]:
  """Parse an absolute recorded reward weight without silently ignoring typos."""
  try:
    name, text = value.split("=", 1)
    weight = float(text)
  except ValueError as exc:
    raise argparse.ArgumentTypeError("use NAME=WEIGHT") from exc
  if not name or not math.isfinite(weight):
    raise argparse.ArgumentTypeError("a name and finite weight are required")
  return name, weight


if __name__ == "__main__":
  main()
