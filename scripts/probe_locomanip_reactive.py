"""Measure six-channel reactive force authority at a frozen failure transition."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import mjlab  # noqa: F401
from evaluation.locomanip_inputs import condition_id
from evaluation.locomanip_response import measure_responses
from evaluation.locomanip_rollout import run_conditions
from evaluation.locomanip_schedules import SCHEDULES, reactive_conditions

from mc_mjlab.tasks.locomanip import FEEDBACK_TASK_IDS


def main() -> None:
  """Run complete first episodes for every schedule, sign, magnitude and repeat."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--config", type=Path, required=True)
  parser.add_argument("--masses", nargs=2, type=float, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--repetitions", type=int, default=3)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--workers", type=int, default=12)
  parser.add_argument(
    "--encoder-bias", action=argparse.BooleanOptionalAction, default=False
  )
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  if args.repetitions < 1 or args.masses[1] <= args.masses[0]:
    parser.error(
      "positive repetitions and ascending passing/failing masses are required"
    )
  args.output.mkdir(parents=True, exist_ok=True)
  all_rows = []
  for mass in args.masses:
    for schedule in SCHEDULES:
      rows = [
        {
          **row,
          "mass_kg": mass,
          "seed": args.seed,
          "condition_id": condition_id(args.seed, mass, row["repeat"]),
          "arm": "probe",
        }
        for row in reactive_conditions(args.repetitions)
        if row["schedule"] == schedule
      ]
      path = args.output / f"{mass:g}-{schedule}"
      result = run_conditions(
        FEEDBACK_TASK_IDS["HRP5P"],
        rows,
        config=args.config,
        encoder_bias=args.encoder_bias,
        workers=args.workers,
        device=args.device,
        trace_path=path.with_suffix(".jsonl"),
      )
      measure_responses(result["rows"], path.with_suffix(".jsonl"))
      path.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
      all_rows.extend(result["rows"])
      print(
        f"[authority] {mass:g} kg {schedule}: "
        f"{sum(row['measurable_physical_response'] for row in result['rows'])} physical responses, "
        f"{sum(row['observed_safety_benefit'] for row in result['rows'])} observed rescues",
        flush=True,
      )
  (args.output / "rows.json").write_text(json.dumps(all_rows, indent=2) + "\n")
  failing = [row for row in all_rows if row["mass_kg"] == args.masses[1]]
  delivered = {
    (row["channel"], 1 if row["level"] > 0 else -1)
    for row in failing
    if row["schedule"] == "constant"
    and abs(row["level"]) == 1.0
    and row["input_delivery_verified"]
  }
  authority = {
    "profile_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
    "encoder_bias": args.encoder_bias,
    "passing_mass_kg": args.masses[0],
    "failing_mass_kg": args.masses[1],
    "all_channels_delivered": len(delivered) == 12,
    "measurable_near_failure": any(
      row["measurable_physical_response"] and row["measurable_target_response"]
      for row in failing
    ),
    "observed_safety_benefit": any(row["observed_safety_benefit"] for row in failing),
    "measured_rows": sum(row["probe_status"] == "measured" for row in failing),
    "untriggered_rows": sum(row["probe_status"] == "not_measured" for row in failing),
  }
  (args.output / "authority_gate.json").write_text(
    json.dumps(authority, indent=2) + "\n"
  )
  print(json.dumps(authority, indent=2))


if __name__ == "__main__":
  main()
