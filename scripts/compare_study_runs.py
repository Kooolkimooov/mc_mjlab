"""Compare zero-policy study runs: outcome agreement, spreads and rollout timing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.study_repeatability import compare_sources, load_sources, run_timing


def main() -> None:
  """Print one JSON report over every run and arm given."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("runs", type=Path, nargs="+", help="study output directories")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()

  report = compare_sources(load_sources(args.runs))
  report["timing"] = [run_timing(run) for run in args.runs]
  text = json.dumps(report, indent=2)
  if args.output:
    args.output.write_text(text + "\n")
  print(text)


if __name__ == "__main__":
  main()
