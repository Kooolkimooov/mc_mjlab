"""Checkpointed held-out qualification state for the achievement curriculum."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from mc_mjlab.tasks.residual_balance.curriculum_stages import (
  ACHIEVEMENT_SCHEMA_VERSION,
  ACHIEVEMENT_STAGES,
  MINIMUM_QUALIFICATION_SEEDS,
  REQUIRED_PASS_REPORTS,
  REQUIRED_QUALIFICATION_SCENARIOS,
  REQUIRED_REGRESSIONS,
  achievement_contract,
  achievement_contract_sha256,
)
from mc_mjlab.tasks.residual_balance.training_watchdog import atomic_json


@dataclass(frozen=True)
class QualificationEvidence:
  """Validated held-out result consumed at one trainer iteration boundary."""

  token: str
  report_sha256: str
  stage: int
  checkpoint: str
  checkpoint_iteration: int
  seeds: tuple[int, ...]
  eligible: bool
  reasons: tuple[str, ...]


@dataclass(frozen=True)
class AchievementState:
  """Serializable curriculum state independent of the global step counter."""

  schema_version: int = ACHIEVEMENT_SCHEMA_VERSION
  current_stage: int = 0
  highest_passed_stage: int = -1
  pass_streak: int = 0
  regression_streak: int = 0
  mastered: bool = False
  processed_reports: tuple[str, ...] = ()
  qualified_checkpoints: tuple[str | None, ...] = field(
    default_factory=lambda: (None,) * len(ACHIEVEMENT_STAGES)
  )
  last_report_checkpoint: str | None = None
  last_report_iteration: int = -1
  last_report_sha256: str | None = None
  last_good_checkpoint: str | None = None

  @classmethod
  def from_dict(cls, value: dict[str, Any]) -> AchievementState:
    """Restore and validate checkpointed achievement state."""
    fields = dict(value)
    fields["processed_reports"] = tuple(fields.get("processed_reports", ()))
    fields["qualified_checkpoints"] = tuple(
      fields.get("qualified_checkpoints", (None,) * len(ACHIEVEMENT_STAGES))
    )
    state = cls(**fields)
    if state.schema_version != ACHIEVEMENT_SCHEMA_VERSION:
      raise ValueError(f"unsupported achievement state schema {state.schema_version}")
    if not 0 <= state.current_stage < len(ACHIEVEMENT_STAGES):
      raise ValueError(f"invalid checkpointed stage {state.current_stage}")
    if not -1 <= state.highest_passed_stage < len(ACHIEVEMENT_STAGES):
      raise ValueError(
        f"invalid checkpointed passed stage {state.highest_passed_stage}"
      )
    if len(state.qualified_checkpoints) != len(ACHIEVEMENT_STAGES):
      raise ValueError("checkpointed qualified-stage path count is invalid")
    return state

  def to_dict(self) -> dict[str, Any]:
    """Convert state to checkpoint-safe built-in values."""
    return asdict(self)


@dataclass(frozen=True)
class AchievementDecision:
  """Result of applying one new qualification report."""

  state: AchievementState
  event: str
  qualified_stage: int | None = None


def apply_qualification(
  state: AchievementState, evidence: QualificationEvidence
) -> AchievementDecision:
  """Apply repeated-pass and sustained-regression hysteresis."""
  if evidence.token in state.processed_reports:
    return AchievementDecision(state, "duplicate")
  if evidence.stage != state.current_stage:
    raise ValueError(
      f"report stage {evidence.stage} does not match active stage {state.current_stage}"
    )
  common = {
    "processed_reports": (*state.processed_reports, evidence.token),
    "last_report_checkpoint": evidence.checkpoint,
    "last_report_iteration": evidence.checkpoint_iteration,
    "last_report_sha256": evidence.report_sha256,
  }

  if evidence.eligible:
    passed = state.pass_streak + 1
    if passed < REQUIRED_PASS_REPORTS:
      return AchievementDecision(
        replace(state, pass_streak=passed, regression_streak=0, **common),
        "pass_pending",
      )

    qualified = state.current_stage
    last_stage = len(ACHIEVEMENT_STAGES) - 1
    next_stage = min(qualified + 1, last_stage)
    qualified_checkpoints = list(state.qualified_checkpoints)
    qualified_checkpoints[qualified] = evidence.checkpoint

    return AchievementDecision(
      replace(
        state,
        current_stage=next_stage,
        highest_passed_stage=qualified,
        pass_streak=0,
        regression_streak=0,
        mastered=qualified == last_stage,
        qualified_checkpoints=tuple(qualified_checkpoints),
        last_good_checkpoint=evidence.checkpoint,
        **common,
      ),
      "mastered" if qualified == last_stage else "advanced",
      qualified,
    )

  regressions = state.regression_streak + 1
  if regressions < REQUIRED_REGRESSIONS:
    return AchievementDecision(
      replace(state, pass_streak=0, regression_streak=regressions, **common),
      "regression_pending",
    )

  if state.current_stage == 0:
    target = 0
    highest = -1
  else:
    target = state.current_stage - 1
    highest = min(state.highest_passed_stage, target)

  last_good = state.qualified_checkpoints[target] if highest >= target else None

  return AchievementDecision(
    replace(
      state,
      current_stage=target,
      highest_passed_stage=highest,
      pass_streak=0,
      regression_streak=0,
      mastered=False,
      last_good_checkpoint=last_good,
      **common,
    ),
    "rolled_back" if target != state.current_stage else "regression_reset",
  )


def read_qualification_evidence(
  path: Path,
  expected_stage: int,
  run_dir: Path,
  current_iteration: int,
) -> QualificationEvidence:
  """Validate one qualifier report as independent current-stage evidence."""
  payload = path.read_bytes()
  report_sha256 = hashlib.sha256(payload).hexdigest()
  report = json.loads(payload)

  if not isinstance(report, dict):
    raise TypeError(f"{path} must contain a JSON object")
  config = report.get("config")
  if not isinstance(config, dict):
    raise ValueError(f"{path} has no qualifier config")
  record = config.get("achievement")
  if not isinstance(record, dict):
    raise ValueError(f"{path} was not run with --achievement-stage")

  stage = int(record.get("stage", -1))
  if stage != expected_stage:
    raise ValueError(f"report stage {stage} does not match stage {expected_stage}")
  expected_digest = achievement_contract_sha256(stage)
  if record.get("contract") != achievement_contract(stage):
    raise ValueError(f"stage {stage} qualification contract is malformed")
  if record.get("contract_sha256") != expected_digest:
    raise ValueError(f"stage {stage} qualification contract differs from training")

  seeds = tuple(sorted({int(seed) for seed in config.get("seeds", [])}))
  if len(seeds) < MINIMUM_QUALIFICATION_SEEDS:
    raise ValueError(
      f"achievement qualification needs {MINIMUM_QUALIFICATION_SEEDS} seeds"
    )
  scenarios = config.get("scenarios", [])
  if set(scenarios) != set(REQUIRED_QUALIFICATION_SCENARIOS):
    raise ValueError("achievement qualification must run every required scenario")

  checkpoints = report.get("checkpoints")
  if not isinstance(checkpoints, dict) or len(checkpoints) != 1:
    raise ValueError("achievement qualification must contain exactly one checkpoint")

  checkpoint, values = next(iter(checkpoints.items()))
  if not isinstance(checkpoint, str):
    raise ValueError("qualified checkpoint key must be a path string")
  checkpoint_path = Path(checkpoint).expanduser().resolve()
  if not checkpoint_path.is_file():
    raise FileNotFoundError(checkpoint_path)
  try:
    checkpoint_path.relative_to(run_dir.resolve())
  except ValueError as error:
    raise ValueError(
      "qualified checkpoint is outside the active run directory"
    ) from error

  match = re.search(r"model_(\d+)", checkpoint_path.name)
  if match is None:
    raise ValueError("qualified checkpoint name has no model iteration")
  iteration = int(match.group(1))
  if iteration > current_iteration:
    raise ValueError(
      f"qualified iteration {iteration} is newer than trainer {current_iteration}"
    )

  if not isinstance(values, dict):
    raise ValueError("qualified checkpoint summary must be an object")
  promotion = values.get("promotion")
  if not isinstance(promotion, dict):
    raise ValueError("qualified checkpoint has no promotion verdict")
  reasons = promotion.get("reasons", [])
  if not isinstance(reasons, list):
    raise ValueError("qualification reasons must be a list")

  identity = json.dumps(
    {"checkpoint": str(checkpoint_path), "seeds": seeds, "stage": stage},
    sort_keys=True,
    separators=(",", ":"),
  )

  return QualificationEvidence(
    token=hashlib.sha256(identity.encode()).hexdigest(),
    report_sha256=report_sha256,
    stage=stage,
    checkpoint=str(checkpoint_path),
    checkpoint_iteration=iteration,
    seeds=seeds,
    eligible=bool(promotion.get("eligible", False)),
    reasons=tuple(str(reason) for reason in reasons),
  )


class AchievementCurriculumBridge:
  """Consume qualifier reports and synchronize checkpointed event difficulty."""

  def __init__(self, runner: Any, log_dir: str | Path | None) -> None:
    self.runner = runner
    self.env = runner.env.unwrapped
    self.term = self._resolve_term()
    self.active = self.term is not None
    if self.active and bool(getattr(runner, "is_distributed", False)):
      raise RuntimeError("achievement curriculum requires single-process training")
    initial_stage = int(getattr(self.term, "current_stage", 0))
    self.state = AchievementState(current_stage=initial_stage)
    self.run_dir = Path(log_dir).resolve() if log_dir is not None else None
    self.root = self.run_dir / "curriculum" if self.run_dir is not None else None
    self.enabled = (
      self.active and self.root is not None and int(os.environ.get("RANK", "0")) == 0
    )
    self._last_observed_token = self._existing_report_token()
    self._last_report: dict[str, Any] | None = None
    if self.enabled:
      assert self.root is not None
      self.root.mkdir(parents=True, exist_ok=True)
      self._publish_state()

  def iteration(self, iteration: int) -> dict[str, float]:
    """Consume at most one new report and return logger scalars."""
    if self.enabled:
      self._consume_report(iteration)
    return {
      "stage": float(self.state.current_stage),
      "highest_passed_stage": float(self.state.highest_passed_stage),
      "pass_streak": float(self.state.pass_streak),
      "regression_streak": float(self.state.regression_streak),
      "mastered": float(self.state.mastered),
    }

  def snapshot(self) -> dict[str, Any] | None:
    """Return checkpoint metadata when the active task uses this curriculum."""
    return self.state.to_dict() if self.active else None

  def restore(self, value: dict[str, Any] | None) -> None:
    """Restore stage hysteresis without deriving it from elapsed steps."""
    if not self.active:
      return
    if value is None:
      print("[mc_mjlab] checkpoint predates achievement curriculum state")
      return
    self.state = AchievementState.from_dict(value)
    if self._last_observed_token != self.state.last_report_sha256:
      self._last_observed_token = None
    self._set_stage(self.state.current_stage)
    if self.enabled:
      self._publish_state()

  def _resolve_term(self) -> Any | None:
    """Find the marked achievement disturbance without coupling to its class."""
    try:
      term = self.env.event_manager.get_term_cfg("push_robot").func
    except (AttributeError, ValueError):
      return None
    if not bool(getattr(term, "is_achievement_curriculum", False)):
      return None
    if not callable(getattr(term, "set_stage", None)):
      raise TypeError("achievement disturbance has no set_stage method")
    return term

  def _existing_report_token(self) -> str | None:
    """Ignore a report inherited before this trainer attached."""
    if self.root is None:
      return None
    try:
      return hashlib.sha256((self.root / "qualification.json").read_bytes()).hexdigest()
    except FileNotFoundError:
      return None

  def _consume_report(self, iteration: int) -> None:
    """Apply a newly replaced qualification report if it satisfies the contract."""
    assert self.root is not None and self.run_dir is not None
    report_path = self.root / "qualification.json"
    try:
      payload = report_path.read_bytes()
    except FileNotFoundError:
      return
    token = hashlib.sha256(payload).hexdigest()
    if token == self._last_observed_token:
      return
    self._last_observed_token = token
    try:
      evidence = read_qualification_evidence(
        report_path, self.state.current_stage, self.run_dir, iteration
      )
      decision = apply_qualification(self.state, evidence)
      next_state = decision.state
      if decision.qualified_stage is not None:
        preserved = self._preserve_checkpoint(
          Path(evidence.checkpoint), decision.qualified_stage
        )
        qualified = list(next_state.qualified_checkpoints)
        qualified[decision.qualified_stage] = str(preserved)
        next_state = replace(
          next_state,
          qualified_checkpoints=tuple(qualified),
          last_good_checkpoint=str(preserved),
        )
      self.state = next_state
      self._set_stage(next_state.current_stage)
      self._last_report = {
        "state": "accepted",
        "event": decision.event,
        "eligible": evidence.eligible,
        "checkpoint": evidence.checkpoint,
        "checkpoint_iteration": evidence.checkpoint_iteration,
        "seeds": list(evidence.seeds),
        "reasons": list(evidence.reasons),
        "processed_unix": time.time(),
      }
      self._append_event(self._last_report)
      print(
        f"[mc_mjlab] achievement curriculum {decision.event}: "
        f"stage {self.state.current_stage}"
      )
    except (json.JSONDecodeError, OSError, TypeError, ValueError) as error:
      self._last_report = {
        "state": "rejected",
        "message": str(error),
        "processed_unix": time.time(),
      }
      self._append_event(self._last_report)
      print(f"[mc_mjlab] rejected achievement report: {error}")
    self._publish_state()

  def _preserve_checkpoint(self, source: Path, stage: int) -> Path:
    """Keep the exact checkpoint that passed a stage gate."""
    assert self.root is not None
    if not source.is_file():
      raise FileNotFoundError(source)
    target = self.root / f"qualified_stage_{stage}_{source.name}"
    if target.is_file():
      if (
        hashlib.sha256(target.read_bytes()).digest()
        != hashlib.sha256(source.read_bytes()).digest()
      ):
        raise ValueError(f"qualified checkpoint collision at {target}")
      return target.resolve()
    shutil.copy2(source, target)
    with target.open("rb") as stream:
      os.fsync(stream.fileno())
    return target.resolve()

  def _set_stage(self, stage: int) -> None:
    """Update future reset cohorts and their manager-visible state."""
    if self.term is None:
      return
    self.term.set_stage(stage)
    self.env.curriculum_manager.compute()

  def _append_event(self, event: dict[str, Any]) -> None:
    """Append one durable transition or report rejection."""
    assert self.root is not None
    with (self.root / "events.jsonl").open("a") as stream:
      stream.write(json.dumps(event, allow_nan=True) + "\n")
      stream.flush()
      os.fsync(stream.fileno())

  def _publish_state(self) -> None:
    """Publish state and the exact next qualification request."""
    assert self.root is not None
    stage = self.state.current_stage
    atomic_json(
      self.root / "state.json",
      {
        **self.state.to_dict(),
        "stage_contract": achievement_contract(stage),
        "stage_contract_sha256": achievement_contract_sha256(stage),
        "last_report": self._last_report,
      },
    )
    atomic_json(
      self.root / "qualification_request.json",
      {
        "schema_version": ACHIEVEMENT_SCHEMA_VERSION,
        "stage": stage,
        "stage_contract": achievement_contract(stage),
        "stage_contract_sha256": achievement_contract_sha256(stage),
        "report_path": str((self.root / "qualification.json").resolve()),
      },
    )
