"""Kill one controller worker mid-episode and verify the study retries only its rows."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

WORKER = "mc_rtc_interface_worker"


def main() -> None:
  """Run the evaluator, SIGKILL its first worker after a delay, then audit outputs."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument(
    "--cohort", type=Path, default=Path("docs/locomanip-study-smoke.json")
  )
  parser.add_argument("--kill-after-s", type=float, default=60.0)
  parser.add_argument("--workers", type=int, default=3)
  args = parser.parse_args()

  command = [
    sys.executable,
    "scripts/evaluate_locomanip_study.py",
    "--cohort",
    str(args.cohort),
    "--batch-size",
    "6",
    "--workers",
    str(args.workers),
    "--output",
    str(args.output),
  ]
  evaluator = subprocess.Popen(command)
  victim = _kill_first_worker(evaluator.pid, args.kill_after_s)
  code = evaluator.wait()
  report = _audit(args.output, victim, code)
  (args.output / "worker_replacement.json").write_text(json.dumps(report, indent=2))
  print(json.dumps(report, indent=2))
  if not report["passed"]:
    raise SystemExit(1)


def _kill_first_worker(parent: int, delay_s: float) -> dict:
  """Wait until workers exist and the delay passed, then kill the lowest row range."""
  deadline = time.monotonic() + delay_s
  while True:
    workers = _workers(parent)
    if workers and time.monotonic() >= deadline:
      pid, endpoint = min(workers, key=lambda item: item[1])
      os.kill(pid, signal.SIGKILL)
      return {"pid": pid, "endpoint": endpoint, "killed_at_s": delay_s}
    time.sleep(1.0)


def _workers(parent: int) -> list[tuple[int, str]]:
  found = []
  for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
      continue
    try:
      stat = (entry / "stat").read_text().split()
      args = (entry / "cmdline").read_bytes().split(b"\0")
    except OSError:
      continue
    if int(stat[3]) == parent and args[0].endswith(WORKER.encode()):
      endpoint = args[args.index(b"--endpoint") + 1].decode()
      found.append((int(entry.name), endpoint.rsplit("/", 1)[-1]))
  return found


def _audit(output: Path, victim: dict, code: int) -> dict:
  history = json.loads((output / "retry_history.json").read_text())
  episodes = json.loads((output / "episodes.json").read_text())
  cohort = json.loads((output / "cohort.json").read_text())
  failed = sorted({c for entry in history for c in entry["worker_failures"]})
  expected = len(cohort["conditions"]) * 2
  return {
    "victim": victim,
    "evaluator_exit": code,
    "first_attempt_failures": history[0]["worker_failures"],
    "retried_conditions": failed,
    "attempts": [entry["label"] for entry in history],
    "records": len(episodes),
    "expected_records": expected,
    "remaining_worker_failures": sum(e["worker_failure"] for e in episodes),
    "passed": code == 0
    and bool(history[0]["worker_failures"])
    and len(episodes) == expected
    and not any(e["worker_failure"] for e in episodes),
  }


if __name__ == "__main__":
  main()
