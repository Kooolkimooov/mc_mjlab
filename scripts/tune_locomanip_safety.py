"""Measure zero-force tuning envelopes and retain policy-step failure traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mjlab  # noqa: F401
import yaml
from evaluation.locomanip_inputs import condition_id
from evaluation.locomanip_rollout import run_conditions
from evaluation.locomanip_tuning import (
  INITIAL_GRID,
  LIGHT_GRID,
  TUNINGS,
  effective_configuration,
  rank_tunings,
  write_variant,
)

from mc_mjlab.tasks.locomanip import FEEDBACK_TASK_IDS
from mc_mjlab.tasks.locomanip.profiles import HRP5P


def main() -> None:
  """Sweep three repeats per mass, extending only completely passing envelopes."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--tunings", nargs="+", choices=list(TUNINGS), default=list(TUNINGS)
  )
  parser.add_argument("--masses", nargs="+", type=float)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--repetitions", type=int, default=3)
  parser.add_argument(
    "--encoder-bias", action=argparse.BooleanOptionalAction, default=False
  )
  parser.add_argument("--workers", type=int, default=9)
  parser.add_argument("--native-logs", action="store_true")
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  if args.repetitions < 1:
    parser.error("--repetitions must be positive")
  args.output.mkdir(parents=True, exist_ok=True)
  effective = effective_configuration(HRP5P.mc_rtc_yaml)
  (args.output / "effective-controller.yaml").write_text(
    yaml.safe_dump(effective, sort_keys=False)
  )
  reports = {}
  for tuning in args.tunings:
    destination = args.output / tuning
    native = destination / "native" if args.native_logs else None
    config = write_variant(
      HRP5P.mc_rtc_yaml, effective, tuning, destination / "mc_rtc.yaml", native
    )
    masses = args.masses or [*LIGHT_GRID, *INITIAL_GRID]
    all_rows = []
    while masses:
      rows = [
        {
          "mass_kg": mass,
          "repeat": repeat,
          "seed": args.seed,
          "condition_id": condition_id(args.seed, mass, repeat),
          "channel": "zero",
          "level": 0.0,
          "arm": "baseline",
        }
        for mass in masses
        for repeat in range(args.repetitions)
      ]
      result = run_conditions(
        FEEDBACK_TASK_IDS["HRP5P"],
        rows,
        config=config,
        encoder_bias=args.encoder_bias,
        workers=len(rows) if native else args.workers,
        device=args.device,
        trace_path=destination / f"trace-{masses[0]:g}-{masses[-1]:g}.jsonl",
      )
      all_rows.extend(result["rows"])
      result["rows"] = all_rows
      (destination / "results.json").write_text(json.dumps(result, indent=2) + "\n")
      print(
        f"[tuning] {tuning}: "
        + ", ".join(
          f"{mass:g} kg {sum(row['complete'] for row in all_rows if row['mass_kg'] == mass)}/{args.repetitions} complete"
          for mass in masses
        ),
        flush=True,
      )
      if (
        args.masses
        or not all(row["complete"] for row in all_rows)
        or max(masses) >= 1000
      ):
        break
      masses = [max(masses) + 25]
    reports[tuning] = all_rows
  ranking = rank_tunings(reports)
  (args.output / "selection.json").write_text(json.dumps(ranking, indent=2) + "\n")
  print(json.dumps(ranking, indent=2))


if __name__ == "__main__":
  main()
