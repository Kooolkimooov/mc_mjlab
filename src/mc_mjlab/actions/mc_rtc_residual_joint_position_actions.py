"""Residual joint-position action term backed by per-env mc_rtc controllers.

Adds an RL residual on top of each mc_rtc controller's joint-position output
(velocity tracks the raw output). The worker pool, shared-memory I/O, the
one-period-behind dispatch pipeline and the target interpolation all live in
``mc_rtc_residual_action.McRtcResidualActionBase``; this module only maps the
interpolated controller output to position/velocity targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mc_mjlab.actions.mc_rtc_residual_action import (
  McRtcResidualActionBase,
  McRtcResidualActionCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class McRtcResidualJointPositionActionCfg(McRtcResidualActionCfg):
  """Configuration for mc_rtc residual joint position control."""

  def build(self, env: ManagerBasedRlEnv) -> "McRtcResidualJointPositionAction":
    return McRtcResidualJointPositionAction(self, env)


class McRtcResidualJointPositionAction(McRtcResidualActionBase):
  """mc_rtc residual action driving joint position + velocity targets.

  The RL residual is added to the interpolated controller position; velocity
  tracks the raw controller output (no residual).
  """

  cfg: McRtcResidualJointPositionActionCfg

  # Controller outputs consumed, in output-block order (host writes q then alpha).
  output_channels = ("q", "alpha")

  def _seed_interpolation(self, env_ids: torch.Tensor) -> None:
    # Position ramps from the current stance; velocity from zero.
    stance = self._entity.data.joint_pos[:, self._target_ids]
    self._previous_control["q"][env_ids] = stance[env_ids]
    self._next_control["q"][env_ids] = stance[env_ids]
    self._previous_control["alpha"][env_ids] = 0.0
    self._next_control["alpha"][env_ids] = 0.0

  def _apply_control(
    self, interpolated_control: dict[str, torch.Tensor], residual: torch.Tensor
  ) -> None:
    target = interpolated_control["q"] + residual
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
    self._entity.set_joint_velocity_target(
      interpolated_control["alpha"], joint_ids=self._target_ids
    )
