"""Provide cooperative trainer control and pure watchdog health decisions."""

from __future__ import annotations

import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

LEVELS = {"ok": 0, "warn": 1, "preserve": 2, "stop": 3}
TERMINATION_PREFIX = "Episode_Termination/"


class WatchdogStop(RuntimeError):
  """End training successfully after a requested checkpoint is durable."""


@dataclass(frozen=True)
class WatchdogThresholds:
  """Thresholds that map one monitor observation to an intervention level."""

  heartbeat_warn_s: float = 300.0
  heartbeat_preserve_s: float = 900.0
  heartbeat_stop_s: float = 1800.0
  checkpoint_warn_s: float = 3600.0
  checkpoint_preserve_s: float = 7200.0
  checkpoint_stop_s: float = 14400.0
  gpu_warn_mb: float = 4096.0
  gpu_preserve_mb: float = 2048.0
  gpu_stop_mb: float = 768.0
  worker_warn: int = 1
  worker_preserve: int = 3
  worker_stop: int = 8
  qualification_warn: int = 1
  qualification_preserve: int = 2
  qualification_stop: int = 3


@dataclass(frozen=True)
class HealthObservation:
  """Current external facts used by the watchdog decision policy."""

  heartbeat_age_s: float
  checkpoint_age_s: float
  gpu_free_mb: float | None
  gpu_low_streak: int
  worker_failures: int
  qualification_bad_streak: int


@dataclass(frozen=True)
class HealthDecision:
  """Highest requested intervention and all contributing reasons."""

  level: str
  reasons: tuple[str, ...]


def process_start_ticks(pid: int) -> int | None:
  """Read Linux's non-reused process start token for one PID."""
  try:
    suffix = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(suffix[19])
  except (FileNotFoundError, IndexError, OSError, ValueError):
    return None


def process_matches(pid: int, start_ticks: int) -> bool:
  """Return whether PID still names the exact process originally attached."""
  return process_start_ticks(pid) == start_ticks


def atomic_json(path: Path, value: dict[str, Any]) -> None:
  """Replace one JSON document without exposing a partially written file."""
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
  temporary.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")
  temporary.replace(path)


def read_json(path: Path) -> dict[str, Any] | None:
  """Read a JSON object, returning ``None`` while it is absent."""
  try:
    value = json.loads(path.read_text())
  except FileNotFoundError:
    return None
  if not isinstance(value, dict):
    raise TypeError(f"{path} must contain a JSON object")
  return value


def _threshold_level(
  value: float,
  warn: float,
  preserve: float,
  stop: float,
  *,
  low_is_bad: bool = False,
) -> str:
  """Map one monotone value through warn, preserve, and stop thresholds."""
  if low_is_bad:
    if value <= stop:
      return "stop"
    if value <= preserve:
      return "preserve"
    if value <= warn:
      return "warn"
  else:
    if value >= stop:
      return "stop"
    if value >= preserve:
      return "preserve"
    if value >= warn:
      return "warn"
  return "ok"


def decide_health(
  observation: HealthObservation, thresholds: WatchdogThresholds
) -> HealthDecision:
  """Combine liveness, freshness, memory, worker, and qualification guards."""
  issues: list[tuple[str, str]] = []
  heartbeat = _threshold_level(
    observation.heartbeat_age_s,
    thresholds.heartbeat_warn_s,
    thresholds.heartbeat_preserve_s,
    thresholds.heartbeat_stop_s,
  )
  if heartbeat != "ok":
    issues.append(
      (heartbeat, f"heartbeat stale for {observation.heartbeat_age_s:.0f}s")
    )
  checkpoint = _threshold_level(
    observation.checkpoint_age_s,
    thresholds.checkpoint_warn_s,
    thresholds.checkpoint_preserve_s,
    thresholds.checkpoint_stop_s,
  )
  if checkpoint != "ok":
    issues.append(
      (checkpoint, f"latest checkpoint is {observation.checkpoint_age_s:.0f}s old")
    )
  if observation.gpu_free_mb is not None and observation.gpu_low_streak > 0:
    gpu = _threshold_level(
      observation.gpu_free_mb,
      thresholds.gpu_warn_mb,
      thresholds.gpu_preserve_mb,
      thresholds.gpu_stop_mb,
      low_is_bad=True,
    )
    if gpu == "stop" and observation.gpu_low_streak < 2:
      gpu = "preserve"
    if gpu != "ok":
      issues.append((gpu, f"GPU free memory is {observation.gpu_free_mb:.0f} MiB"))
  worker = _threshold_level(
    float(observation.worker_failures),
    float(thresholds.worker_warn),
    float(thresholds.worker_preserve),
    float(thresholds.worker_stop),
  )
  if worker != "ok":
    issues.append((worker, f"{observation.worker_failures} controller worker failures"))
  qualification = _threshold_level(
    float(observation.qualification_bad_streak),
    float(thresholds.qualification_warn),
    float(thresholds.qualification_preserve),
    float(thresholds.qualification_stop),
  )
  if qualification != "ok":
    issues.append(
      (
        qualification,
        f"{observation.qualification_bad_streak} consecutive qualification regressions",
      )
    )
  level = max((item[0] for item in issues), key=lambda name: LEVELS[name], default="ok")
  return HealthDecision(level, tuple(reason for _, reason in issues))


def intervention_for(
  desired: str,
  maximum: str,
  preserve_acknowledged: bool,
  request_pending: bool,
) -> str | None:
  """Choose the next action while requiring preservation before stopping."""
  if request_pending:
    return None
  bounded = min(LEVELS[desired], LEVELS[maximum])
  if bounded < LEVELS["warn"]:
    return None
  if bounded == LEVELS["warn"]:
    return "warn"
  if bounded >= LEVELS["stop"] and preserve_acknowledged:
    return "stop"
  if bounded == LEVELS["preserve"] and preserve_acknowledged:
    return None
  return "preserve"


def qualification_snapshot(path: Path) -> dict[str, Any]:
  """Extract the latest checkpoint's promotion metrics from a qualifier report."""
  report = read_json(path)
  if report is None:
    raise FileNotFoundError(path)
  checkpoints = report.get("checkpoints")
  if not isinstance(checkpoints, dict) or not checkpoints:
    raise ValueError(f"{path} has no checkpoint summaries")

  def checkpoint_key(item: tuple[str, Any]) -> tuple[int, str]:
    match = re.search(r"model_(\d+)", Path(item[0]).name)
    return (int(match.group(1)) if match else -1, item[0])

  items: list[tuple[str, Any]] = [
    (str(checkpoint), values) for checkpoint, values in checkpoints.items()
  ]
  checkpoint, values = sorted(items, key=checkpoint_key)[-1]
  if not isinstance(values, dict):
    raise ValueError(f"{path} has an invalid summary for {checkpoint}")
  promotion = values.get("promotion", {})
  rank = promotion.get("rank", [])
  recovery_gain = float(rank[0]) if rank else float("nan")
  return {
    "path": str(path.resolve()),
    "checkpoint": checkpoint,
    "iteration": checkpoint_key((checkpoint, values))[0],
    "eligible": bool(promotion.get("eligible", False)),
    "hazard_ratio": float(promotion.get("hazard_ratio", float("nan"))),
    "recovery_gain": recovery_gain,
  }


def qualification_baseline(samples: list[dict[str, Any]]) -> dict[str, Any]:
  """Build a robust post-attach baseline from fresh qualification snapshots."""
  if not samples:
    raise ValueError("qualification baseline needs at least one sample")

  def median(name: str) -> float:
    values = sorted(
      float(sample[name]) for sample in samples if math.isfinite(float(sample[name]))
    )
    if not values:
      return float("nan")
    middle = len(values) // 2
    if len(values) % 2:
      return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0

  return {
    "samples": len(samples),
    "eligible": sum(bool(sample["eligible"]) for sample in samples) > len(samples) / 2,
    "hazard_ratio": median("hazard_ratio"),
    "recovery_gain": median("recovery_gain"),
  }


def qualification_regressions(
  baseline: dict[str, Any],
  current: dict[str, Any],
  hazard_margin: float = 0.10,
  recovery_margin: float = 0.05,
) -> tuple[str, ...]:
  """Report promotion-metric regressions relative to a post-attach baseline."""
  reasons = []
  if bool(baseline["eligible"]) and not bool(current["eligible"]):
    reasons.append("a previously eligible qualification is now ineligible")
  base_hazard = float(baseline["hazard_ratio"])
  hazard = float(current["hazard_ratio"])
  if math.isfinite(base_hazard) and math.isfinite(hazard):
    if hazard > base_hazard + hazard_margin:
      reasons.append(f"hazard ratio regressed from {base_hazard:.3f} to {hazard:.3f}")
  base_recovery = float(baseline["recovery_gain"])
  recovery = float(current["recovery_gain"])
  if math.isfinite(base_recovery) and math.isfinite(recovery):
    if recovery < base_recovery - recovery_margin:
      reasons.append(
        f"recovery gain regressed from {base_recovery:.3f} to {recovery:.3f}"
      )
  return tuple(reasons)


class RunnerWatchdogBridge:
  """Publish health and honor checkpointed control requests at iteration boundaries."""

  def __init__(self, runner: Any, log_dir: str | Path | None) -> None:
    self.runner = runner
    self.enabled = log_dir is not None and int(os.environ.get("RANK", "0")) == 0
    self.root = Path(log_dir) / "watchdog" if log_dir is not None else None
    self.started_unix = time.time()
    self._last_request_id: str | None = None
    self._last_checkpoint: str | None = None
    self._termination_totals: dict[str, int] = {}
    self._heartbeat: dict[str, Any] = {}
    if self.enabled:
      assert log_dir is not None
      assert self.root is not None
      self.root.mkdir(parents=True, exist_ok=True)
      runtime = {
        "schema_version": 1,
        "pid": os.getpid(),
        "process_start_ticks": process_start_ticks(os.getpid()),
        "run_dir": str(Path(log_dir).resolve()),
        "started_unix": self.started_unix,
        "distributed": bool(getattr(runner, "is_distributed", False)),
        "cooperative_stop_supported": not bool(
          getattr(runner, "is_distributed", False)
        ),
      }
      atomic_json(self.root / "runtime.json", runtime)
      self._write_status("starting")

  def iteration(
    self,
    iteration: int,
    diagnostics: dict[str, float],
    episode_extras: list[dict[str, Any]],
    collect_time: float,
    learn_time: float,
  ) -> None:
    """Publish a completed iteration and service any cooperative request."""
    if not self.enabled:
      return
    delta = self._termination_delta(episode_extras)
    for name, count in delta.items():
      self._termination_totals[name] = self._termination_totals.get(name, 0) + count
    self._heartbeat = {
      "schema_version": 1,
      "state": "running",
      "pid": os.getpid(),
      "updated_unix": time.time(),
      "iteration": int(iteration),
      "common_step_counter": int(self.runner.env.unwrapped.common_step_counter),
      "collect_time_s": float(collect_time),
      "learn_time_s": float(learn_time),
      "last_checkpoint": self._last_checkpoint,
      "episode_terminations_delta": delta,
      "episode_terminations_total": dict(self._termination_totals),
      "ppo": diagnostics,
      "cuda": self._cuda_memory(),
    }
    assert self.root is not None
    atomic_json(self.root / "heartbeat.json", self._heartbeat)
    self._write_status("running", iteration=int(iteration))
    self._service_request(iteration)

  def checkpoint_saved(self, path: str | Path) -> None:
    """Attach the newest durable checkpoint to the next heartbeat."""
    if not self.enabled:
      return
    self._last_checkpoint = str(Path(path).resolve())
    if self._heartbeat:
      self._heartbeat["last_checkpoint"] = self._last_checkpoint
      self._heartbeat["updated_unix"] = time.time()
      assert self.root is not None
      atomic_json(self.root / "heartbeat.json", self._heartbeat)

  def completed(self) -> None:
    """Publish normal exhaustion of the configured training budget."""
    self._write_status(
      "completed",
      iteration=int(self.runner.current_learning_iteration),
      checkpoint=self._last_checkpoint,
    )

  def failed(self, error: BaseException) -> None:
    """Publish an exception category without swallowing the trainer failure."""
    message = str(error)
    lowered = message.lower()
    cause = "oom" if "out of memory" in lowered else type(error).__name__
    self._write_status(
      "failed",
      iteration=int(self.runner.current_learning_iteration),
      cause=cause,
      message=message[-2000:],
      checkpoint=self._last_checkpoint,
    )

  def _termination_delta(self, extras: list[dict[str, Any]]) -> dict[str, int]:
    """Sum exact episode termination counts emitted during one rollout."""
    output: dict[str, int] = {}
    for values in extras:
      for name, value in values.items():
        if not name.startswith(TERMINATION_PREFIX):
          continue
        if isinstance(value, torch.Tensor):
          count = int(value.detach().sum().item())
        else:
          count = int(value)
        short_name = name.removeprefix(TERMINATION_PREFIX)
        output[short_name] = output.get(short_name, 0) + count
    return output

  def _cuda_memory(self) -> dict[str, float] | None:
    """Read allocator and device memory without starting a second CUDA context."""
    if not torch.cuda.is_available() or not str(self.runner.device).startswith("cuda"):
      return None
    free, total = torch.cuda.mem_get_info(self.runner.device)
    divisor = 1024.0**2
    return {
      "free_mb": free / divisor,
      "total_mb": total / divisor,
      "allocated_mb": torch.cuda.memory_allocated(self.runner.device) / divisor,
      "reserved_mb": torch.cuda.memory_reserved(self.runner.device) / divisor,
    }

  def _service_request(self, iteration: int) -> None:
    """Save before acknowledging preserve or raising a cooperative stop."""
    assert self.root is not None
    request_path = self.root / "request.json"
    try:
      request = read_json(request_path)
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
      atomic_json(
        self.root / "response.json",
        {"state": "rejected", "message": str(error), "updated_unix": time.time()},
      )
      return
    if request is None:
      return
    request_id = str(request.get("request_id", ""))
    if not request_id or request_id == self._last_request_id:
      return
    self._last_request_id = request_id
    action = request.get("action")
    if action not in ("preserve", "stop"):
      atomic_json(
        self.root / "response.json",
        {
          "request_id": request_id,
          "state": "rejected",
          "message": f"unsupported action {action!r}",
          "updated_unix": time.time(),
        },
      )
      return
    if bool(getattr(self.runner, "is_distributed", False)):
      atomic_json(
        self.root / "response.json",
        {
          "request_id": request_id,
          "state": "rejected",
          "message": "cooperative watchdog control is single-GPU only",
          "updated_unix": time.time(),
        },
      )
      return
    checkpoint = self.root / f"model_{iteration}.pt"
    self.runner.save(
      str(checkpoint),
      infos={"watchdog_request": request, "watchdog_iteration": int(iteration)},
    )
    with checkpoint.open("rb") as stream:
      os.fsync(stream.fileno())
    response = {
      "request_id": request_id,
      "action": action,
      "state": "acknowledged",
      "iteration": int(iteration),
      "checkpoint": str(checkpoint.resolve()),
      "updated_unix": time.time(),
    }
    atomic_json(self.root / "response.json", response)
    if action == "stop":
      self._write_status(
        "stopped_by_watchdog",
        iteration=int(iteration),
        checkpoint=str(checkpoint.resolve()),
        request_id=request_id,
      )
      raise WatchdogStop(f"watchdog stop acknowledged at iteration {iteration}")

  def _write_status(self, state: str, **values: Any) -> None:
    """Publish runner lifecycle state for unambiguous external classification."""
    if not self.enabled:
      return
    assert self.root is not None
    atomic_json(
      self.root / "status.json",
      {
        "schema_version": 1,
        "state": state,
        "pid": os.getpid(),
        "updated_unix": time.time(),
        **values,
      },
    )

  def as_config(self) -> dict[str, Any]:
    """Return the bridge's persistent runtime facts for checkpoint metadata."""
    return {
      "enabled": self.enabled,
      "started_unix": self.started_unix,
      "protocol_version": 1,
    }
