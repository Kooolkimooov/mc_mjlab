"""Reanalyze archived phase probes without replacing their original measurements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mjlab  # noqa: F401
from evaluation.objective_audit import (
  category,
  episode_returns,
  ranking_defects,
  verdict,
)
from evaluation.phase_summary import event_timing, outcome_matrix, zero_contrasts
from evaluation.phase_windows import analyze_episode, find_events, load_traces
from evaluation.study_io import write_json
from freeze_locomanip_references import file_identity, revision
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.string import resolve_matching_names

from mc_mjlab.tasks.locomanip.locomanip_ppo_cfg import locomanip_ppo_cfg
from mc_mjlab.tasks.locomanip.mdp.accessors import ACTION_NAME


def main() -> None:
  """Keep the first non-worker outcome and attach its own attempt and joint order."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("probe", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--names", type=Path, required=True, help="trusted full joint-order metadata"
  )
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=False)
  base_names = json.loads(args.names.read_text())
  latest = {}
  paths = sorted(
    args.probe.glob("*-attempt-*.json"), key=lambda p: int(p.stem.rsplit("-", 1)[1])
  )
  identities = []
  for path in paths:
    batch = json.loads(path.read_text())
    if "rows" not in batch:
      continue
    identities.append(file_identity(path))
    names = batch.get("trace_names") or recover_names(batch, base_names)
    trace = path.with_suffix(".jsonl")
    if trace.exists():
      (args.output / trace.name).symlink_to(trace.resolve())
      identities.append(file_identity(trace))
    for raw in batch["rows"]:
      key = (raw["arm"], raw["condition_id"])
      if key in latest and not latest[key]["worker_failure"]:
        continue
      row = dict(raw)
      flags = row["terminations"]
      row["combined_failure"] = any(
        flags.get(n, False) for n in ("fell_over", "collapsed", "controller_failed")
      )
      row["complete"] = bool(row["complete"] and not row["combined_failure"])
      row["trace_file"] = trace.name
      row["trace_names"] = names
      row["source_attempt"] = path.name
      latest[key] = row
  records = list(latest.values())
  write_json(args.output / "episodes.json", records)
  gamma = locomanip_ppo_cfg().algorithm.gamma
  phases, objectives = [], []
  for path in sorted(args.output.glob("*.jsonl")):
    traces = load_traces([path], records)
    for key, steps in traces.items():
      row = latest[key]
      if not row["observed"] or row["worker_failure"]:
        continue
      names = row["trace_names"]
      phases.append(analyze_episode(steps, row, names, names["step_dt"]))
      objectives.append(
        {
          "arm": row["arm"],
          "condition_id": row["condition_id"],
          "category": category(row),
          "objectives": episode_returns(
            steps,
            row,
            names["rewards"],
            names["step_dt"],
            gamma,
            find_events(steps)["hold_entry"],
          ),
        }
      )
  defects = ranking_defects(objectives)
  write_json(args.output / "phases.json", phases)
  write_json(
    args.output / "summary.json",
    {
      "outcomes": outcome_matrix(phases),
      "events": event_timing(phases),
      "zero_contrasts": zero_contrasts(phases),
    },
  )
  write_json(
    args.output / "objective.json",
    {
      "gamma": gamma,
      "episodes": objectives,
      "defects": defects,
      "verdict": verdict(defects),
    },
  )
  write_json(
    args.output / "reanalysis.json",
    {
      "source": revision(Path.cwd()),
      "input": str(args.probe.resolve()),
      "artifacts": identities,
      "names": file_identity(args.names),
      "rule": "first non-worker outcome; all QP failures count; legacy names reconstructed from recorded config or unchanged task registration",
    },
  )
  print(
    json.dumps(
      {
        "records": len(records),
        "phases": len(phases),
        "failures": sum(r["combined_failure"] for r in records),
        "objective": verdict(defects),
      }
    )
  )


def recover_names(batch: dict, base: dict) -> dict:
  """Resolve archived joint patterns against the recorded robot joint ordering."""
  manifest = batch.get("effective_manifest")
  if manifest:
    config = manifest["record"]["managers"]["actions"]["terms"][ACTION_NAME]["config"]
    patterns = config["residual_actuator_names"]
  else:
    cfg = load_env_cfg(batch["task"])
    patterns = cfg.actions[ACTION_NAME].residual_actuator_names
  _, residual = resolve_matching_names(patterns, base["joints"])
  return {**base, "residual": residual}


if __name__ == "__main__":
  main()
