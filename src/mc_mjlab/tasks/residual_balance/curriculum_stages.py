"""Shared contracts for achievement-gated residual-balance difficulty."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

ACHIEVEMENT_SCHEMA_VERSION = 1
REQUIRED_QUALIFICATION_SCENARIOS = (
  "nominal",
  "current_kick",
  "finite_impulse",
  "robust",
)
REQUIRED_PASS_REPORTS = 2
REQUIRED_REGRESSIONS = 3
MINIMUM_QUALIFICATION_SEEDS = 2


@dataclass(frozen=True)
class AchievementStage:
  """Define one push level and its training rehearsal mixture."""

  name: str
  training_velocity_range: tuple[float, float]
  qualification_velocity: float
  robust_velocity_range: tuple[float, float]
  rehearsal_weights: tuple[float, ...]


ACHIEVEMENT_STAGES = (
  AchievementStage(
    "foundation",
    (0.10, 0.25),
    0.25,
    (0.30, 0.35),
    (0.25, 0.75, 0.00, 0.00),
  ),
  AchievementStage(
    "recovery",
    (0.10, 0.40),
    0.40,
    (0.45, 0.50),
    (0.15, 0.25, 0.60, 0.00),
  ),
  AchievementStage(
    "hard",
    (0.10, 0.50),
    0.50,
    (0.55, 0.60),
    (0.15, 0.15, 0.20, 0.50),
  ),
)


def achievement_stage(index: int) -> AchievementStage:
  """Return one validated achievement stage."""
  if not 0 <= index < len(ACHIEVEMENT_STAGES):
    raise ValueError(f"achievement stage must be 0-{len(ACHIEVEMENT_STAGES) - 1}")
  return ACHIEVEMENT_STAGES[index]


def achievement_contract(index: int) -> dict:
  """Return the canonical training and held-out contract for one stage."""
  stage = achievement_stage(index)
  return {
    "schema_version": ACHIEVEMENT_SCHEMA_VERSION,
    "stage": index,
    "stage_definition": json.loads(json.dumps(asdict(stage))),
    "units": "equivalent_delta_velocity_m_per_s",
    "required_scenarios": list(REQUIRED_QUALIFICATION_SCENARIOS),
    "minimum_seeds": MINIMUM_QUALIFICATION_SEEDS,
    "required_pass_reports": REQUIRED_PASS_REPORTS,
    "required_regressions": REQUIRED_REGRESSIONS,
  }


def achievement_contract_sha256(index: int) -> str:
  """Hash one stage contract for trainer/evaluator agreement."""
  payload = json.dumps(
    achievement_contract(index), sort_keys=True, separators=(",", ":")
  )
  return hashlib.sha256(payload.encode()).hexdigest()


def _validate_stages() -> None:
  """Reject malformed rehearsal mixtures at import time."""
  width = len(ACHIEVEMENT_STAGES) + 1
  for index, stage in enumerate(ACHIEVEMENT_STAGES):
    if len(stage.rehearsal_weights) != width:
      raise ValueError(f"stage {index} rehearsal mixture must have {width} entries")
    if abs(sum(stage.rehearsal_weights) - 1.0) > 1e-9:
      raise ValueError(f"stage {index} rehearsal mixture must sum to one")
    if stage.rehearsal_weights[0] <= 0.0:
      raise ValueError(f"stage {index} must retain standing practice")
    if any(stage.rehearsal_weights[index + 2 :]):
      raise ValueError(f"stage {index} samples an unqualified future stage")


_validate_stages()
