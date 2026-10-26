"""Summarize a phase-probe directory: outcomes, event timing and zero contrasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.phase_summary import event_timing, outcome_matrix, zero_contrasts
from evaluation.study_io import write_json


def main() -> None:
  """Write summary.json beside phases.json and print the outcome matrix."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("probe", type=Path, help="probe_locomanip_phases output")
  args = parser.parse_args()

  phases = json.loads((args.probe / "phases.json").read_text())
  unavailable_path = args.probe / "unavailable.json"
  summary = {
    "outcomes": outcome_matrix(phases),
    "events": event_timing(phases),
    "zero_contrasts": zero_contrasts(phases),
    "unavailable": json.loads(unavailable_path.read_text())
    if unavailable_path.exists()
    else [],
  }
  write_json(args.probe / "summary.json", summary)

  masses = sorted({m for row in summary["outcomes"].values() for m in row}, key=float)
  print("arm".ljust(26) + "".join(m.rjust(6) for m in masses) + "  (fresh|recycled)")
  for arm, row in summary["outcomes"].items():
    print(arm.ljust(26) + "".join(row.get(m, "--").rjust(6) for m in masses))
  for item in summary["unavailable"]:
    print(f"unavailable {item['arm']}: {item['reason'][:160]}")


if __name__ == "__main__":
  main()
