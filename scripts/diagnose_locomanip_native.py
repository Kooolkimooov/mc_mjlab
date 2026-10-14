"""Summarize one isolated locomanip native controller episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.locomanip_native import summarize_native


def main() -> None:
  """Match a single binary controller log to a single episode record."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-dir", type=Path, required=True)
  args = parser.parse_args()
  logs = [
    path
    for path in (args.run_dir / "current/native").glob("*.bin")
    if not path.is_symlink() and path.stat().st_size > 1000000
  ]
  if len(logs) != 1:
    parser.error(f"expected one nonempty native log, found {len(logs)}")
  rows = json.loads((args.run_dir / "current/results.json").read_text())["rows"]
  if len(rows) != 1:
    parser.error("expected one episode row")
  native = summarize_native(logs[0])
  native["condition_id"] = rows[0]["condition_id"]
  native["mass_kg"] = rows[0]["mass_kg"]
  native["terminal_time_s"] = rows[0]["time_s"]
  native["aligned"] = abs(native["end_s"] - native["terminal_time_s"]) < 0.1
  (args.run_dir / "native-summary.json").write_text(json.dumps(native, indent=2) + "\n")
  print(json.dumps(native, indent=2))


if __name__ == "__main__":
  main()
