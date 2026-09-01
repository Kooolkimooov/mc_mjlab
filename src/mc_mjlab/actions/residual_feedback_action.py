"""Residual feedback learning: steer the controller's input, not only its output."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mc_mjlab.actions.residual_mpc_joint_torque_action import (
  ResidualMpcJointTorqueAction,
  ResidualMpcJointTorqueActionCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ResidualFeedbackJointTorqueActionCfg(ResidualMpcJointTorqueActionCfg):
  """Configuration for Ranjbar-style residual feedback beside the torque residual."""

  feedback_scale: float = 0.02
  """Radians of encoder offset at a saturated feedback action."""

  torque_channel: bool = True
  """Keep the torque residual; False gives the paper's feedback-only variant."""

  def build(self, env: ManagerBasedRlEnv) -> "ResidualFeedbackJointTorqueAction":
    return ResidualFeedbackJointTorqueAction(self, env)


class ResidualFeedbackJointTorqueAction(ResidualMpcJointTorqueAction):
  """Add a residual to the encoders mc_rtc reads, so it replans with the offset."""

  cfg: ResidualFeedbackJointTorqueActionCfg

  def __init__(self, cfg: ResidualFeedbackJointTorqueActionCfg, env: ManagerBasedRlEnv):
    if cfg.feedback_scale <= 0.0:
      raise ValueError("feedback_scale must be positive")
    super().__init__(cfg, env)
    ids = self._residual_ids
    self._feedback_dim = self._num_targets if ids is None else int(ids.numel())
    # Only the env-facing width grows: `_raw_actions` keeps the base class's
    # size because `process_actions` hands it the block ahead of the feedback.
    self._action_dim += self._feedback_dim
    self._feedback_requested = torch.zeros(
      self.num_envs, self._feedback_dim, device=self.device
    )
    self._feedback_offset = torch.zeros(
      self.num_envs, self._num_targets, device=self.device
    )
    print(
      f"[mc_rtc] ResidualFeedback: {self._feedback_dim} encoder channel(s) at "
      f"{cfg.feedback_scale:.4f} rad"
      f"{'' if cfg.torque_channel else '; torque residual disabled'}."
    )

  def process_actions(self, actions: torch.Tensor) -> None:
    actions = actions.clamp(-1.0, 1.0)
    head, feedback = (
      actions[:, : -self._feedback_dim],
      actions[:, -self._feedback_dim :],
    )
    self._feedback_requested.copy_(feedback * self.cfg.feedback_scale)
    # Gated with the torque residual so a suppressed residual cannot keep lying
    # to the controller about where its joints are.
    gated = self._feedback_requested * self._last_gate.unsqueeze(-1)
    if self._residual_ids is None:
      self._feedback_offset.copy_(gated)
    else:
      self._feedback_offset.zero_()
      self._feedback_offset[:, self._residual_ids] = gated
    self._io.set_feedback_offset(self._feedback_offset)
    if not self.cfg.torque_channel:
      head = head.clone()
      head[:, : self._residual_action_dim] = 0.0
    super().process_actions(head)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self._feedback_requested[env_ids] = 0.0
    self._feedback_offset[env_ids] = 0.0

  @property
  def feedback_offset(self) -> torch.Tensor:
    """Per-target encoder offset currently handed to the controller, in rad."""
    return self._feedback_offset

  @property
  def requested_feedback(self) -> torch.Tensor:
    """Per-residual-joint feedback residual before gating, in rad."""
    return self._feedback_requested
