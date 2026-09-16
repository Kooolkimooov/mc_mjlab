"""What scoring a residual-balance checkpoint has to measure."""

from __future__ import annotations

from mc_mjlab.tasks.evaluation import TaskEvaluation

#: Push speed buckets, in m/s: the qualifier's own bands. docs/difficulty.md
IMPULSE_STRATA = (0.25, 0.40, 0.60)

RESIDUAL_BALANCE_EVALUATION = TaskEvaluation(
  metrics=(
    "zmp_error",
    "zmp_grounded",
    "dcm_error",
    "recovery_dcm_error",
    "recovery_active",
    "impulse_speed",
  ),
  stratify="impulse_speed",
  strata=IMPULSE_STRATA,
)
