"""Evaluate fixed paired mass cohorts with pooled and per-mass safety gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mjlab  # noqa: F401
from evaluation.locomanip_inputs import condition_id
from evaluation.locomanip_promotion import summarize_safety
from evaluation.locomanip_rollout import run_conditions

from mc_mjlab.tasks.locomanip_safety import TASK_ID
from mc_mjlab.tasks.locomanip_safety.evaluation import MASS_GRID_KG, TRANSITION_GRID_KG


def main() -> None:
  """Screen with eight pairs or validate with 32 pairs at each explicit seed."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default=TASK_ID)
  parser.add_argument("--config", type=Path)
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--validation", action="store_true")
  parser.add_argument("--seeds", nargs="+", type=int)
  parser.add_argument("--repetitions", type=int)
  parser.add_argument("--masses", nargs="+", type=float, default=list(MASS_GRID_KG))
  parser.add_argument(
    "--transition-masses", nargs="+", type=float, default=list(TRANSITION_GRID_KG)
  )
  parser.add_argument(
    "--encoder-bias", action=argparse.BooleanOptionalAction, default=True
  )
  parser.add_argument("--workers", type=int, default=4)
  parser.add_argument("--trace", action="store_true")
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  seeds = args.seeds or ([43, 44] if args.validation else [42])
  repetitions = args.repetitions or (32 if args.validation else 8)
  if repetitions < 2 or not set(args.transition_masses) <= set(args.masses):
    parser.error("use >=2 repetitions and include every transition mass in --masses")
  if args.task == TASK_ID and (
    tuple(args.masses) != MASS_GRID_KG
    or tuple(args.transition_masses) != TRANSITION_GRID_KG
  ):
    parser.error("the safety task's mass and transition grids are frozen")
  args.output.mkdir(parents=True, exist_ok=True)
  manifest = {
    "task": args.task,
    "masses_kg": args.masses,
    "transition_masses_kg": args.transition_masses,
    "seeds": seeds,
    "repetitions": repetitions,
    "encoder_bias": args.encoder_bias,
    "checkpoint": str(args.checkpoint) if args.checkpoint else None,
  }
  (args.output / "cohort.json").write_text(json.dumps(manifest, indent=2) + "\n")
  all_rows = []
  for seed in seeds:
    for mass in args.masses:
      rows = [
        {
          "seed": seed,
          "mass_kg": mass,
          "repeat": repeat,
          "condition_id": condition_id(seed, mass, repeat),
          "arm": arm,
          "channel": "zero",
          "level": 0.0,
        }
        for repeat in range(repetitions)
        for arm in ("baseline", "policy")
      ]
      label = f"seed-{seed}-mass-{mass:g}"
      result = run_conditions(
        args.task,
        rows,
        config=args.config,
        checkpoint=args.checkpoint,
        encoder_bias=args.encoder_bias,
        workers=args.workers,
        device=args.device,
        trace_path=args.output / f"{label}.jsonl" if args.trace else None,
      )
      all_rows.extend(result["rows"])
      (args.output / f"{label}.json").write_text(json.dumps(result, indent=2) + "\n")
      print(
        f"[safety] {label}: {sum(row['combined_failure'] for row in result['rows'])} combined failures",
        flush=True,
      )
  report = summarize_safety(
    all_rows, args.masses, args.transition_masses, seeds, repetitions
  )
  if args.checkpoint is None:
    report["promote"] = False
    report["purpose"] = "Repeated zero controls; no policy was evaluated."
  (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
