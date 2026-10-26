"""Phase-resolved locomanip authority: zero, references and constant group requests."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import mjlab  # noqa: F401
from evaluation.phase_windows import analyze_episode, load_traces
from evaluation.study_cohort import StudyCohort
from evaluation.study_io import write_json
from evaluation.study_rollout import ReferenceUnavailable, run_study_conditions
from freeze_locomanip_references import file_identity, revision

from mc_mjlab.tasks.locomanip import RESIDUAL_TASK_IDS
from mc_mjlab.tasks.locomanip_study import CONTROL_TASK_ID

PRODUCTION_TASK_ID = RESIDUAL_TASK_IDS["HRP5P"]
#: Which scale profile each frozen reference was trained under.
REFERENCE_TASKS = {
  "original": CONTROL_TASK_ID,
  "no_magnitude": CONTROL_TASK_ID,
  "arms_2x": PRODUCTION_TASK_ID,
}
PROFILE_TASKS = {"control": CONTROL_TASK_ID, "production": PRODUCTION_TASK_ID}
CONSTANT_GROUPS = {
  "arms": ("arms",),
  "waist": ("waist",),
  "combined": ("arms", "waist"),
}


@dataclass(frozen=True)
class ProbeArm:
  """One independently simulated policy source."""

  name: str
  task: str
  checkpoint: str | None = None
  groups: tuple[str, ...] = ()
  level: float = 0.0


def main() -> None:
  """Run every arm on identical conditions, then segment each episode by phase."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--masses", default="600,650,700,750,800,900")
  parser.add_argument("--lifecycles", default="fresh,recycled")
  parser.add_argument("--seeds", default="1001")
  parser.add_argument("--draws", type=int, default=1)
  parser.add_argument("--encoder-draws", action="store_true")
  parser.add_argument("--levels", default="-1,-0.5,0.5,1")
  parser.add_argument("--groups", default="arms,waist,combined")
  parser.add_argument("--profiles", default="control")
  parser.add_argument("--references", type=Path)
  parser.add_argument("--arms", help="comma-separated arm names to run")
  parser.add_argument("--workers", type=int, default=4)
  parser.add_argument("--max-retries", type=int, choices=(0, 1, 2), default=2)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()

  cohort = StudyCohort(
    masses=tuple(float(m) for m in args.masses.split(",")),
    seeds=tuple(int(s) for s in args.seeds.split(",")),
    draws=args.draws,
    lifecycles=tuple(args.lifecycles.split(",")),
    nominal=not args.encoder_draws,
  )
  arms = probe_arms(args)
  if args.arms:
    keep = set(args.arms.split(","))
    arms = [arm for arm in arms if arm.name in keep]
  args.output.mkdir(parents=True, exist_ok=False)
  write_json(
    args.output / "probe.json",
    {
      "cohort": asdict(cohort),
      "source": revision(Path(__file__).resolve().parents[1]),
      "arms": [
        asdict(arm)
        | {
          "checkpoint": file_identity(Path(arm.checkpoint)) if arm.checkpoint else None
        }
        for arm in arms
      ],
    },
  )

  records, traces, unavailable = [], [], []
  for arm in arms:
    try:
      records.extend(run_arm(arm, cohort, args, traces))
    except ReferenceUnavailable as error:
      unavailable.append({"arm": arm.name, "reason": str(error)})
      write_json(args.output / "unavailable.json", unavailable)
      print(f"[phases] {arm.name} unavailable: {error}")
    write_json(args.output / "episodes.json", records)

  episodes = load_traces(traces)
  names = json.loads((args.output / "names.json").read_text())
  dt = names.pop("step_dt")
  phases = [
    analyze_episode(episodes[(r["arm"], r["condition_id"])], r, names, dt)
    for r in records
    if r["observed"] and (r["arm"], r["condition_id"]) in episodes
  ]
  write_json(args.output / "phases.json", phases)
  print(f"[phases] {len(phases)} segmented episodes of {len(records)} records")


def probe_arms(args: argparse.Namespace) -> list[ProbeArm]:
  """Zero, available frozen references, then each constant request and profile."""
  arms = [ProbeArm("zero", CONTROL_TASK_ID)]
  if args.references:
    inventory = json.loads(args.references.read_text())["references"]
    for name, task in REFERENCE_TASKS.items():
      if inventory.get(name, {}).get("present"):
        arms.append(ProbeArm(name, task, checkpoint=inventory[name]["path"]))
  for profile in args.profiles.split(","):
    for group in args.groups.split(","):
      for level in (float(v) for v in args.levels.split(",")):
        arms.append(
          ProbeArm(
            f"{profile}/{group}/{level:g}",
            PROFILE_TASKS[profile],
            groups=CONSTANT_GROUPS[group],
            level=level,
          )
        )
  return arms


def run_arm(
  arm: ProbeArm, cohort: StudyCohort, args: argparse.Namespace, traces: list[Path]
) -> list[dict]:
  """Simulate one arm, retrying only worker failures, and keep its traces."""
  label = arm.name.replace("/", "_")
  pending = [{**row, "arm": arm.name} for row in cohort.conditions()]
  latest: dict[str, dict] = {}
  for attempt in range(args.max_retries + 1):
    trace = args.output / f"{label}-attempt-{attempt}.jsonl"
    result = run_study_conditions(
      arm.task,
      pending,
      checkpoint=Path(arm.checkpoint) if arm.checkpoint else None,
      workers=args.workers,
      device=args.device,
      trace_path=trace,
      pd_telemetry=True,
      constant=(arm.groups, arm.level) if arm.groups else None,
      names_path=args.output / "names.json",
    )
    traces.append(trace)
    write_json(args.output / f"{label}-attempt-{attempt}.json", result)
    for row in result["rows"]:
      latest[row["condition_id"]] = row
    retry = {row["condition_id"] for row in result["rows"] if row["worker_failure"]}
    pending = [row for row in pending if row["condition_id"] in retry]
    print(f"[phases] {arm.name} attempt {attempt}: {len(retry)} worker failures")
    if not pending:
      break
  return list(latest.values())


if __name__ == "__main__":
  main()
