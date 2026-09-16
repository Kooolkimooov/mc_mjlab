"""What scoring a locomanip checkpoint has to measure."""

from __future__ import annotations

from mc_mjlab.tasks.evaluation import TaskEvaluation

#: Cart mass buckets, in kilograms. The sweep's own breakpoints: the base
#: controller holds to ~100 kg and its adaptation fails above ~300 kg.
#: docs/locomanip.md#cart_mass_range_kg
CART_MASS_STRATA = (10.0, 30.0, 100.0, 300.0)

LOCOMANIP_EVALUATION = TaskEvaluation(
  metrics=(
    "object_position_error",
    "object_yaw_error",
    "zmp_error",
    "zmp_grounded",
    "task_complete",
    "hands_released",
    "cart_mass",
  ),
  success="task_complete",
  stratify="cart_mass",
  strata=CART_MASS_STRATA,
)
