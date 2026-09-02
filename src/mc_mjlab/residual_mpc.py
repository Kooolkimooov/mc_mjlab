"""Pure tensor contracts shared by the ResidualMPC task and its checks."""

import torch

#: Bumped when the blending equation changes meaning; version 1 referenced the
#: controller target where eq (23) requires the default posture, so its actions
#: are not comparable. docs/residual-mpc.md#paper_torque_blend
ACTION_SEMANTICS_VERSION = 2


def paper_joint_action_scale(
  effort_limit: torch.Tensor, kp: torch.Tensor, blend_factor: float
) -> torch.Tensor:
  """Convert normalized actions to paper-scaled joint radians."""
  return (
    torch.minimum(torch.full_like(kp, 0.01), 0.20 * effort_limit.abs() / kp)
    / blend_factor
  )


def advance_action_history(
  current: torch.Tensor, previous: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return previous and second-previous physical action snapshots."""
  return current.clone(), previous.clone()


def paper_torque_blend(
  controller_q: torch.Tensor,
  default_q: torch.Tensor,
  q_biased: torch.Tensor,
  qd: torch.Tensor,
  controller_qd: torch.Tensor,
  controller_torque: torch.Tensor,
  joint_action: torch.Tensor,
  kp: torch.Tensor,
  kd: torch.Tensor,
  blend_factor: torch.Tensor,
  residual_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Return nominal, residual, and blended-residual torques."""
  # Two different references, and swapping them silently changes the architecture:
  # the fallback tracks the controller, while eq (23)'s q_hat is the *default*
  # posture. docs/residual-mpc.md#paper_torque_blend
  fallback = kp * (controller_q - q_biased) + kd * (controller_qd - qd)
  nominal = torch.where(controller_torque != 0.0, controller_torque, fallback)
  residual = kp * (joint_action + default_q - q_biased) - kd * qd
  residual = residual * residual_mask
  blended = blend_factor.unsqueeze(-1) * residual
  return nominal, residual, blended


def contact_phases(
  step_time: torch.Tensor,
  step_duration: torch.Tensor,
  support_side: torch.Tensor,
) -> torch.Tensor:
  """Build right-toe, right-heel, left-toe, left-heel phases."""
  progress = 0.5 * (step_time / step_duration.clamp_min(1.0e-6)).clamp(0.0, 1.0)
  right = torch.remainder(progress + 0.5 * support_side, 1.0)
  left = torch.remainder(right + 0.5, 1.0)
  return torch.stack((right, right, left, left), dim=-1)
