"""Evaluate independent candidate/reference policies on identical study conditions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import mjlab  # noqa: F401
from evaluation.study_cohort import StudyCohort
from evaluation.study_io import write_json
from evaluation.study_rollout import run_study_conditions
from evaluation.study_scenarios import scenarios
from evaluation.study_statistics import summarize
from freeze_locomanip_references import file_identity, revision

from mc_mjlab.tasks.locomanip_study import CONTROL_TASK_ID


def main() -> None:
  """Persist exact inputs and every infrastructure retry before summarizing results."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--candidate-task", default=CONTROL_TASK_ID)
  parser.add_argument("--reference-task", default=CONTROL_TASK_ID)
  parser.add_argument("--candidate-checkpoint", type=Path)
  parser.add_argument("--reference-checkpoint", type=Path)
  parser.add_argument("--cohort", type=Path, help="JSON fields of StudyCohort")
  parser.add_argument("--held-out", action="store_true")
  parser.add_argument("--nominal", action="store_true")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--batch-size", type=int, default=32)
  parser.add_argument("--workers", type=int, default=4)
  parser.add_argument("--max-retries", type=int, choices=(0, 1, 2), default=2)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--trace", action="store_true")
  parser.add_argument("--reverse", action="store_true")
  parser.add_argument("--pd-telemetry", action="store_true")
  parser.add_argument("--scenario", choices=sorted(scenarios()))
  parser.add_argument(
    "--console-output", choices=("none", "single", "all"), default="none"
  )
  parser.add_argument(
    "--pd-rate-trace",
    action="append",
    default=[],
    metavar="CONDITION_ID",
    help="full-rate PD trace for this condition (repeatable)",
  )
  args = parser.parse_args()
  if args.batch_size < 1 or args.workers < 1:
    parser.error("positive batch size and worker count required")
  if args.cohort and (args.held_out or args.nominal):
    parser.error("a custom cohort cannot be combined with preset modifiers")
  if args.pd_rate_trace and not args.pd_telemetry:
    parser.error("--pd-rate-trace requires --pd-telemetry")
  scenario = scenarios()[args.scenario] if args.scenario else None
  if args.cohort:
    fields = json.loads(args.cohort.read_text())
    for key in ("masses", "seeds", "lifecycles"):
      if key in fields:
        fields[key] = tuple(fields[key])
    cohort = StudyCohort(**fields)
  else:
    cohort = StudyCohort(
      seeds=(2001, 2002, 2003) if args.held_out else (1001, 1002),
      draws=32 if args.held_out else 8,
      nominal=args.nominal,
    )
  if scenario is not None and scenario.encoder_bias == cohort.nominal:
    parser.error(
      "encoder-bias scenarios need encoder draws; all others a nominal cohort"
    )
  args.output.mkdir(parents=True, exist_ok=False)
  manifest = {
    "cohort": asdict(cohort),
    "conditions": cohort.conditions(),
    "source": revision(Path(__file__).resolve().parents[1]),
    "batch_size": args.batch_size,
    "max_retries": args.max_retries,
    "action_mode": "deterministic",
    "actor_white_noise": bool(scenario and scenario.white_noise),
    "scenario": asdict(scenario) if scenario else None,
    "pd_telemetry": args.pd_telemetry,
    "pd_rate_trace": sorted(args.pd_rate_trace),
    "candidate": {
      "task": args.candidate_task,
      "checkpoint": file_identity(args.candidate_checkpoint)
      if args.candidate_checkpoint
      else None,
    },
    "reference": {
      "task": args.reference_task,
      "checkpoint": file_identity(args.reference_checkpoint)
      if args.reference_checkpoint
      else None,
    },
    "purpose": "held_out"
    if args.held_out
    else "custom"
    if args.cohort
    else "validation",
  }
  write_json(args.output / "cohort.json", manifest)
  all_rows = []
  history = []
  for arm in ("reference", "candidate"):
    rows = [{**row, "arm": arm} for row in cohort.conditions()]
    if args.reverse:
      rows.reverse()
    for batch, start in enumerate(range(0, len(rows), args.batch_size)):
      pending = rows[start : start + args.batch_size]
      latest = {}
      for attempt in range(args.max_retries + 1):
        label = f"{arm}-batch-{batch}-attempt-{attempt}"
        result = run_study_conditions(
          getattr(args, f"{arm}_task"),
          pending,
          checkpoint=getattr(args, f"{arm}_checkpoint"),
          workers=args.workers,
          device=args.device,
          trace_path=args.output / f"{label}.jsonl" if args.trace else None,
          pd_telemetry=args.pd_telemetry,
          pd_rate_ids=frozenset(args.pd_rate_trace),
          pd_rate_prefix=args.output / label,
          scenario=scenario,
          console_output=args.console_output,
        )
        write_json(args.output / f"{label}.json", result)
        for row in result["rows"]:
          latest[row["condition_id"]] = row
        workers = {r["condition_id"] for r in result["rows"] if r["worker_failure"]}
        starts = {
          r["condition_id"] for r in result["rows"] if r["controller_start_failure"]
        }
        retry = workers | starts
        history.append(
          {
            "label": label,
            "worker_failures": sorted(workers),
            "controller_start_failures": sorted(starts),
          }
        )
        pending = [row for row in pending if row["condition_id"] in retry]
        print(
          f"[study] {label}: {len(latest)} records, {len(pending)} worker failures",
          flush=True,
        )
        if not pending:
          break
      all_rows.extend(latest.values())
      write_json(args.output / "episodes.json", all_rows)
      write_json(args.output / "retry_history.json", history)
  report = summarize(all_rows, cohort)
  report["reference_checkpoint"] = manifest["reference"]["checkpoint"]
  report["candidate_checkpoint"] = manifest["candidate"]["checkpoint"]
  write_json(args.output / "report.json", report)
  print(json.dumps(report, indent=2))


if __name__ == "__main__":
  main()
