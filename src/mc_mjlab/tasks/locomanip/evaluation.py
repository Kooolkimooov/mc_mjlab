"""What scoring a locomanip checkpoint has to measure."""

from __future__ import annotations

from mc_mjlab.tasks.evaluation import TaskEvaluation

#: Cart mass buckets, in kilograms. The sweep's own breakpoints: the base
#: controller completes to ~640 kg and fails from ~700 kg.
#: docs/locomanip.md#cart_push_friction
CART_MASS_STRATA = (600.0, 700.0, 800.0)

LOCOMANIP_EVALUATION = TaskEvaluation(
  metrics=(
    "contact_dcm_stability",
    "forward_cart_lag",
    "object_position_error",
    "object_yaw_error",
    "zmp_error",
    "zmp_grounded",
    "task_complete",
    "safe_completion",
    "robot_displacement",
    "task_success",
    "hands_released",
    "cart_mass",
  ),
  success="safe_completion",
  stratify="cart_mass",
  strata=CART_MASS_STRATA,
)
