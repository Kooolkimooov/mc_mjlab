"""Residual feedback learning: steer the controller's input, not only its output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mc_mjlab.actions.residual_mpc_joint_torque_action import (
  ResidualMpcJointTorqueAction,
  ResidualMpcJointTorqueActionCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

#: Root pose carries 3 translation and 3 rotation channels; joint position is
#: sized by the residual joints. docs/residual-feedback.md#feedback_modalities
ROOT_POSE_DIM = 6
SUPPORTED_MODALITIES = ("joint_position", "joint_velocity", "root_pose", "wrench")


@dataclass(kw_only=True)
class ResidualFeedbackJointTorqueActionCfg(ResidualMpcJointTorqueActionCfg):
  """Configuration for Ranjbar-style residual feedback beside the torque residual."""

  feedback_modalities: tuple[str, ...] = field(
    default_factory=lambda: ("joint_position",)
  )
  """Feedback spaces the policy may offset, in action order."""

  feedback_scale: float = 0.02
  """Radians of encoder offset at a saturated joint_position action."""

  root_translation_scale: float = 0.0025
  """Metres of root offset at a saturated root_pose translation action."""

  root_rotation_scale: float = 0.02
  """Radians of root tilt at a saturated root_pose rotation action."""

  joint_velocity_scale: float = 0.05
  """Rad/s of encoder-velocity offset at a saturated joint_velocity action."""

  wrench_force_scale: float = 50.0
  """Newtons of force offset at a saturated wrench action."""

  wrench_torque_scale: float = 20.0
  """Newton-metres of moment offset at a saturated wrench action."""

  torque_channel: bool = True
  """Keep the torque residual; False gives the paper's feedback-only variant."""

  def __post_init__(self) -> None:
    """Reject an unusable feedback spec here, not minutes into a training run."""
    super().__post_init__()
    if self.feedback_scale <= 0.0:
      raise ValueError("feedback_scale must be positive")
    if not self.feedback_modalities:
      raise ValueError("at least one feedback modality is required")

    unknown = set(self.feedback_modalities) - set(SUPPORTED_MODALITIES)
    if unknown:
      raise ValueError(f"unsupported feedback modalities: {sorted(unknown)}")
    if len(set(self.feedback_modalities)) != len(self.feedback_modalities):
      raise ValueError("feedback modalities must be unique")

  def build(self, env: ManagerBasedRlEnv) -> "ResidualFeedbackJointTorqueAction":
    return ResidualFeedbackJointTorqueAction(self, env)


class ResidualFeedbackJointTorqueAction(ResidualMpcJointTorqueAction):
  """Add a residual to the state mc_rtc reads, so it replans with the offset."""

  cfg: ResidualFeedbackJointTorqueActionCfg

  def __init__(
    self, cfg: ResidualFeedbackJointTorqueActionCfg, env: ManagerBasedRlEnv
  ) -> None:
    super().__init__(cfg, env)
    ids = self._residual_ids
    joint_dim = self._num_targets if ids is None else int(ids.numel())
    wrench_dim = 6 * len(self._bridge.layout.input.force_sensors)
    self._modality_dims = {
      "joint_position": joint_dim,
      "joint_velocity": joint_dim,
      "root_pose": ROOT_POSE_DIM,
      "wrench": wrench_dim,
    }
    if "wrench" in cfg.feedback_modalities and wrench_dim == 0:
      raise ValueError("wrench feedback needs force sensors, the model has none")

    self._modalities = tuple(cfg.feedback_modalities)
    self._feedback_dim = sum(self._modality_dims[m] for m in self._modalities)
    # Only the env-facing width grows: `_raw_actions` keeps the base class's
    # size because `process_actions` hands it the block ahead of the feedback.
    self._action_dim += self._feedback_dim

    self._feedback_offset = torch.zeros(
      self.num_envs, self._num_targets, device=self.device
    )
    self._joint_velocity_offset = torch.zeros_like(self._feedback_offset)
    self._root_translation = torch.zeros(self.num_envs, 3, device=self.device)
    self._root_rotation = torch.zeros(self.num_envs, 3, device=self.device)
    self._wrench_offset = torch.zeros(self.num_envs, wrench_dim, device=self.device)

    # Force and moment share a block but not a unit, so the scale alternates in
    # sixes. docs/residual-feedback.md#feedback_modalities
    self._wrench_scale = torch.tensor(
      ([cfg.wrench_force_scale] * 3 + [cfg.wrench_torque_scale] * 3)
      * len(self._bridge.layout.input.force_sensors),
      device=self.device,
    )

    print(
      f"[mc_rtc] ResidualFeedback: {self._feedback_dim} channel(s) across "
      f"{list(self._modalities)}"
      f"{'' if cfg.torque_channel else '; torque residual disabled'}."
    )

  def process_actions(self, actions: torch.Tensor) -> None:
    actions = actions.clamp(-1.0, 1.0)
    head = actions[:, : -self._feedback_dim]
    blocks = self._modality_slices(actions[:, -self._feedback_dim :])

    # Gated with the torque residual so a suppressed residual cannot keep lying
    # to the controller about where the robot is.
    gate = self._last_gate.unsqueeze(-1)

    if "joint_position" in blocks:
      gated = blocks["joint_position"] * self.cfg.feedback_scale * gate
      if self._residual_ids is None:
        self._feedback_offset.copy_(gated)
      else:
        self._feedback_offset.zero_()
        self._feedback_offset[:, self._residual_ids] = gated
      self._bridge.set_feedback_offset(self._feedback_offset)

    if "joint_velocity" in blocks:
      gated = blocks["joint_velocity"] * self.cfg.joint_velocity_scale * gate
      if self._residual_ids is None:
        self._joint_velocity_offset.copy_(gated)
      else:
        self._joint_velocity_offset.zero_()
        self._joint_velocity_offset[:, self._residual_ids] = gated
      self._bridge.set_joint_velocity_offset(self._joint_velocity_offset)

    if "wrench" in blocks:
      self._wrench_offset.copy_(blocks["wrench"] * self._wrench_scale * gate)
      self._bridge.set_wrench_offset(self._wrench_offset)

    if "root_pose" in blocks:
      root = blocks["root_pose"] * gate
      self._root_translation.copy_(root[:, :3] * self.cfg.root_translation_scale)
      self._root_rotation.copy_(root[:, 3:] * self.cfg.root_rotation_scale)
      self._bridge.set_root_pose_offset(self._root_translation, self._root_rotation)

    if not self.cfg.torque_channel:
      head = head.clone()
      head[:, : self._residual_action_dim] = 0.0

    super().process_actions(head)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)

    self._feedback_offset[env_ids] = 0.0
    self._joint_velocity_offset[env_ids] = 0.0
    self._root_translation[env_ids] = 0.0
    self._root_rotation[env_ids] = 0.0
    self._wrench_offset[env_ids] = 0.0

  def _modality_slices(self, feedback: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split the trailing feedback block into its modalities, in cfg order."""
    out: dict[str, torch.Tensor] = {}
    start = 0

    for name in self._modalities:
      width = self._modality_dims[name]
      out[name] = feedback[:, start : start + width]
      start += width
    return out
