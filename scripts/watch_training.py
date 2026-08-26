"""Monitor one verified training process and request checkpointed intervention."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mc_mjlab.tasks.residual_balance.training_watchdog import (
  LEVELS,
  HealthObservation,
  WatchdogThresholds,
  atomic_json,
  decide_health,
  intervention_for,
  process_matches,
  qualification_baseline,
  qualification_regressions,
  qualification_snapshot,
  read_json,
)


def _gpu_memory(index: int) -> tuple[dict[str, float | int] | None, str | None]:
  """Read physical GPU memory through nvidia-smi without opening CUDA."""
  command = [
    "nvidia-smi",
    "--query-gpu=index,memory.total,memory.used,memory.free",
    "--format=csv,noheader,nounits",
  ]
  try:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
      fields = [field.strip() for field in line.split(",")]
      if int(fields[0]) == index:
        return {
          "index": index,
          "total_mb": float(fields[1]),
          "used_mb": float(fields[2]),
          "free_mb": float(fields[3]),
        }, None
    return None, f"nvidia-smi reported no GPU index {index}"
  except (
    FileNotFoundError,
    IndexError,
    subprocess.SubprocessError,
    ValueError,
  ) as error:
    return None, str(error)


def _latest_checkpoint(run_dir: Path) -> Path | None:
  """Return the newest ordinary or watchdog-preserved checkpoint."""
  candidates = list(run_dir.glob("model_*.pt"))
  candidates.extend((run_dir / "watchdog").glob("model_*.pt"))
  if not candidates:
    return None
  return sorted(candidates, key=lambda path: path.stat().st_mtime_ns)[-1]


def _append_event(path: Path, event: dict[str, Any]) -> None:
  """Append one timestamped machine-readable watchdog event."""
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a") as stream:
    stream.write(json.dumps(event, allow_nan=True) + "\n")


def _response_for(root: Path, request: dict[str, Any] | None) -> dict | None:
  """Return the response matching the currently pending request."""
  if request is None:
    return None
  response = read_json(root / "response.json")
  if response is None or response.get("request_id") != request["request_id"]:
    return None
  return response


def _terminal_verdict(status: dict[str, Any] | None) -> tuple[str, str, int]:
  """Classify a dead trainer without conflating completion and watchdog stop."""
  state = status.get("state") if status else None
  if state == "completed":
    return "completed", "configured training budget completed", 0
  if state == "stopped_by_watchdog":
    return "stopped_by_watchdog", "trainer honored a checkpointed stop", 0
  if state == "failed":
    cause = status.get("cause", "unknown") if status is not None else "unknown"
    return "trainer_failed", f"trainer reported {cause}", 2
  return "trainer_exited_unexpectedly", "process exited without terminal status", 2


def _validate_thresholds(values: WatchdogThresholds) -> None:
  """Reject threshold orderings that invert the intended escalation levels."""
  increasing = (
    (values.heartbeat_warn_s, values.heartbeat_preserve_s, values.heartbeat_stop_s),
    (
      values.checkpoint_warn_s,
      values.checkpoint_preserve_s,
      values.checkpoint_stop_s,
    ),
    (values.worker_warn, values.worker_preserve, values.worker_stop),
    (
      values.qualification_warn,
      values.qualification_preserve,
      values.qualification_stop,
    ),
  )
  if any(not (warn <= preserve <= stop) for warn, preserve, stop in increasing):
    raise ValueError("warn/preserve/stop thresholds must be nondecreasing")
  if not values.gpu_warn_mb >= values.gpu_preserve_mb >= values.gpu_stop_mb:
    raise ValueError("GPU warn/preserve/stop thresholds must be nonincreasing")


def _thresholds(args: argparse.Namespace) -> WatchdogThresholds:
  """Build a typed health policy from command-line overrides."""
  values = WatchdogThresholds(
    heartbeat_warn_s=args.heartbeat_warn_s,
    heartbeat_preserve_s=args.heartbeat_preserve_s,
    heartbeat_stop_s=args.heartbeat_stop_s,
    checkpoint_warn_s=args.checkpoint_warn_s,
    checkpoint_preserve_s=args.checkpoint_preserve_s,
    checkpoint_stop_s=args.checkpoint_stop_s,
    gpu_warn_mb=args.gpu_warn_mb,
    gpu_preserve_mb=args.gpu_preserve_mb,
    gpu_stop_mb=args.gpu_stop_mb,
    worker_warn=args.worker_warn,
    worker_preserve=args.worker_preserve,
    worker_stop=args.worker_stop,
    qualification_warn=args.qualification_warn,
    qualification_preserve=args.qualification_preserve,
    qualification_stop=args.qualification_stop,
  )
  _validate_thresholds(values)
  return values


def _parser() -> argparse.ArgumentParser:
  """Build the explicit-attachment watchdog command line."""
  defaults = WatchdogThresholds()
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--pid", type=int, required=True)
  parser.add_argument("--run-dir", type=Path, required=True)
  parser.add_argument("--gpu-index", type=int, default=0)
  parser.add_argument("--poll-s", type=float, default=30.0)
  parser.add_argument(
    "--max-level", choices=("warn", "preserve", "stop"), default="stop"
  )
  parser.add_argument("--qualification-report", type=Path)
  parser.add_argument("--qualification-baseline-samples", type=int, default=2)
  parser.add_argument("--qualification-hazard-margin", type=float, default=0.10)
  parser.add_argument("--qualification-recovery-margin", type=float, default=0.05)
  parser.add_argument(
    "--heartbeat-warn-s", type=float, default=defaults.heartbeat_warn_s
  )
  parser.add_argument(
    "--heartbeat-preserve-s", type=float, default=defaults.heartbeat_preserve_s
  )
  parser.add_argument(
    "--heartbeat-stop-s", type=float, default=defaults.heartbeat_stop_s
  )
  parser.add_argument(
    "--checkpoint-warn-s", type=float, default=defaults.checkpoint_warn_s
  )
  parser.add_argument(
    "--checkpoint-preserve-s", type=float, default=defaults.checkpoint_preserve_s
  )
  parser.add_argument(
    "--checkpoint-stop-s", type=float, default=defaults.checkpoint_stop_s
  )
  parser.add_argument("--gpu-warn-mb", type=float, default=defaults.gpu_warn_mb)
  parser.add_argument("--gpu-preserve-mb", type=float, default=defaults.gpu_preserve_mb)
  parser.add_argument("--gpu-stop-mb", type=float, default=defaults.gpu_stop_mb)
  parser.add_argument("--worker-warn", type=int, default=defaults.worker_warn)
  parser.add_argument("--worker-preserve", type=int, default=defaults.worker_preserve)
  parser.add_argument("--worker-stop", type=int, default=defaults.worker_stop)
  parser.add_argument(
    "--qualification-warn", type=int, default=defaults.qualification_warn
  )
  parser.add_argument(
    "--qualification-preserve", type=int, default=defaults.qualification_preserve
  )
  parser.add_argument(
    "--qualification-stop", type=int, default=defaults.qualification_stop
  )
  parser.add_argument("--once", action="store_true")
  return parser


def main() -> int:
  """Attach, monitor, escalate cooperatively, and emit a final verdict."""
  args = _parser().parse_args()
  thresholds = _thresholds(args)
  if args.poll_s <= 0.0:
    raise ValueError("--poll-s must be positive")
  if args.qualification_baseline_samples <= 0:
    raise ValueError("--qualification-baseline-samples must be positive")
  run_dir = args.run_dir.expanduser().resolve()
  root = run_dir / "watchdog"
  runtime = read_json(root / "runtime.json")
  if runtime is None:
    raise FileNotFoundError(f"watchdog runtime marker is absent under {run_dir}")
  if int(runtime.get("pid", -1)) != args.pid:
    raise RuntimeError(
      f"run directory belongs to PID {runtime.get('pid')}, not requested PID {args.pid}"
    )
  start_ticks = int(runtime["process_start_ticks"])
  if not process_matches(args.pid, start_ticks):
    raise RuntimeError(f"PID {args.pid} is not the live process recorded by {run_dir}")
  if not bool(runtime.get("cooperative_stop_supported", False)):
    raise RuntimeError("this run does not support cooperative watchdog intervention")

  started = time.time()
  events_path = root / "events.jsonl"
  verdict_path = root / "verdict.json"
  qualification_token = None
  if args.qualification_report is not None and args.qualification_report.exists():
    qualification_token = args.qualification_report.stat().st_mtime_ns
  qualification_samples: list[dict[str, Any]] = []
  baseline = None
  current_qualification = None
  qualification_error = None
  qualification_reasons: tuple[str, ...] = ()
  qualification_bad_streak = 0
  gpu_low_streak = 0
  worker_baseline = None
  preserve_acknowledged = False
  stop_acknowledged = False
  pending_request = None
  last_event_signature = None
  exit_code = 0
  state = "monitoring"
  print(f"[watchdog] attached PID {args.pid} to {run_dir}", flush=True)

  try:
    while True:
      now = time.time()
      status = read_json(root / "status.json")
      alive = process_matches(args.pid, start_ticks)
      if not alive:
        state, reason, exit_code = _terminal_verdict(status)
        _append_event(
          events_path,
          {"timestamp_unix": now, "state": state, "message": reason},
        )
        print(f"[watchdog] {state}: {reason}", flush=True)
        break

      heartbeat = read_json(root / "heartbeat.json")
      heartbeat_time = (
        float(heartbeat["updated_unix"])
        if heartbeat
        else float(runtime["started_unix"])
      )
      heartbeat_age = max(0.0, now - heartbeat_time)
      checkpoint = _latest_checkpoint(run_dir)
      checkpoint_time = (
        checkpoint.stat().st_mtime if checkpoint else float(runtime["started_unix"])
      )
      checkpoint_age = max(0.0, now - checkpoint_time)
      total_worker_failures = 0
      if heartbeat is not None:
        totals = heartbeat.get("episode_terminations_total", {})
        total_worker_failures = int(totals.get("controller_worker_failed", 0))
        if worker_baseline is None:
          worker_baseline = total_worker_failures
      worker_failures = (
        max(0, total_worker_failures - worker_baseline)
        if worker_baseline is not None
        else 0
      )

      gpu, gpu_error = _gpu_memory(args.gpu_index)
      gpu_free = float(gpu["free_mb"]) if gpu is not None else None
      if gpu_free is not None and gpu_free <= thresholds.gpu_warn_mb:
        gpu_low_streak += 1
      else:
        gpu_low_streak = 0

      if args.qualification_report is not None and args.qualification_report.exists():
        token = args.qualification_report.stat().st_mtime_ns
        if token != qualification_token:
          try:
            snapshot = qualification_snapshot(args.qualification_report)
          except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
            qualification_error = str(error)
          else:
            qualification_token = token
            qualification_error = None
            current_qualification = snapshot
            if baseline is None:
              qualification_samples.append(current_qualification)
              if len(qualification_samples) >= args.qualification_baseline_samples:
                baseline = qualification_baseline(qualification_samples)
              qualification_reasons = ()
              qualification_bad_streak = 0
            else:
              qualification_reasons = qualification_regressions(
                baseline,
                current_qualification,
                args.qualification_hazard_margin,
                args.qualification_recovery_margin,
              )
              qualification_bad_streak = (
                qualification_bad_streak + 1 if qualification_reasons else 0
              )

      response = _response_for(root, pending_request)
      if response is not None:
        assert pending_request is not None
        if response.get("state") == "acknowledged":
          if pending_request["action"] == "preserve":
            preserve_acknowledged = True
          else:
            stop_acknowledged = True
          _append_event(
            events_path,
            {
              "timestamp_unix": now,
              "state": "request_acknowledged",
              "request": pending_request,
              "response": response,
            },
          )
        else:
          _append_event(
            events_path,
            {
              "timestamp_unix": now,
              "state": "request_rejected",
              "request": pending_request,
              "response": response,
            },
          )
        pending_request = None

      observation = HealthObservation(
        heartbeat_age_s=heartbeat_age,
        checkpoint_age_s=checkpoint_age,
        gpu_free_mb=gpu_free,
        gpu_low_streak=gpu_low_streak,
        worker_failures=worker_failures,
        qualification_bad_streak=qualification_bad_streak,
      )
      decision = decide_health(observation, thresholds)
      reasons = list(decision.reasons) + list(qualification_reasons)
      action = (
        None
        if stop_acknowledged
        else intervention_for(
          decision.level,
          args.max_level,
          preserve_acknowledged,
          pending_request is not None,
        )
      )
      if LEVELS[decision.level] <= LEVELS["warn"]:
        preserve_acknowledged = False
      signature = (decision.level, tuple(reasons), action)
      if (
        signature != last_event_signature
        and decision.level != "ok"
        and not stop_acknowledged
        and pending_request is None
      ):
        _append_event(
          events_path,
          {
            "timestamp_unix": now,
            "state": "health_degraded",
            "level": decision.level,
            "reasons": reasons,
            "action": action,
          },
        )
        print(
          f"[watchdog] {decision.level}: {'; '.join(reasons)}"
          + (f" -> {action}" if action else ""),
          flush=True,
        )
      last_event_signature = signature

      if action in ("preserve", "stop"):
        pending_request = {
          "schema_version": 1,
          "request_id": uuid.uuid4().hex,
          "action": action,
          "reason": reasons,
          "created_unix": now,
          "watchdog_pid": os.getpid(),
        }
        atomic_json(root / "request.json", pending_request)

      verdict = {
        "schema_version": 1,
        "state": state,
        "level": decision.level,
        "reasons": reasons,
        "next_action": action,
        "trainer_pid": args.pid,
        "watchdog_pid": os.getpid(),
        "run_dir": str(run_dir),
        "updated_unix": now,
        "runtime": runtime,
        "runner_status": status,
        "heartbeat": heartbeat,
        "heartbeat_age_s": heartbeat_age,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_age_s": checkpoint_age,
        "gpu": gpu,
        "gpu_error": gpu_error,
        "worker_failures_since_attach": worker_failures,
        "qualification": {
          "baseline": baseline,
          "baseline_samples": qualification_samples,
          "current": current_qualification,
          "bad_streak": qualification_bad_streak,
          "reasons": qualification_reasons,
          "error": qualification_error,
        },
        "pending_request": pending_request,
        "preserve_acknowledged": preserve_acknowledged,
        "stop_acknowledged": stop_acknowledged,
        "thresholds": asdict(thresholds),
        "max_level": args.max_level,
        "monitor_started_unix": started,
      }
      atomic_json(verdict_path, verdict)
      if args.once:
        break
      time.sleep(args.poll_s)
  except KeyboardInterrupt:
    state = "monitor_interrupted"
    exit_code = 130
    print("[watchdog] monitor interrupted; trainer was not signaled", flush=True)
  finally:
    final = read_json(verdict_path) or {}
    final.update({"state": state, "finished_unix": time.time()})
    atomic_json(verdict_path, final)
  return exit_code


if __name__ == "__main__":
  raise SystemExit(main())
