"""Rank matched phase-probe episodes under the current and the alive-only objective."""

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
from evaluation.phase_windows import find_events, load_traces
from evaluation.study_io import write_json

from mc_mjlab.tasks.locomanip.locomanip_ppo_cfg import locomanip_ppo_cfg


def main() -> None:
  """Write objective.json into the probe directory and print the decision."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("probe", type=Path, help="probe_locomanip_phases output")
  args = parser.parse_args()

  records = json.loads((args.probe / "episodes.json").read_text())
  traces = load_traces(sorted(args.probe.glob("*-attempt-*.jsonl")), records)
  gamma = locomanip_ppo_cfg().algorithm.gamma

  episodes = []
  for record in records:
    names = record["trace_names"]
    steps = traces.get((record["arm"], record["condition_id"]))
    if not record["observed"] or not steps:
      continue
    episodes.append(
      {
        "arm": record["arm"],
        "condition_id": record["condition_id"],
        "category": category(record),
        "end_s": record["time_s"],
        "displacement_m": record["robot_displacement_m"],
        "objectives": episode_returns(
          steps,
          record,
          names["rewards"],
          names["step_dt"],
          gamma,
          find_events(steps)["hold_entry"],
        ),
      }
    )
  defects = ranking_defects(episodes)
  report = {
    "gamma": gamma,
    "episodes": episodes,
    "defects": defects,
    "verdict": verdict(defects),
  }
  write_json(args.probe / "objective.json", report)
  print(json.dumps({"defects": defects, "verdict": report["verdict"]}, indent=2))


if __name__ == "__main__":
  main()
