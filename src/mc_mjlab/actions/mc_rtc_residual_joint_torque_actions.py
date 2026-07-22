"""Residual joint-torque action term backed by per-env mc_rtc controllers.

Adds an RL residual to the joint torques the mc_rtc QP computes, reproducing
mc_mujoco's ``--torque-control`` law (``MjRobot::sendControl``): q, alpha and
jointTorque are each interpolated across ``frameskip``, and per joint the
interpolated torque drives the actuator unless it is exactly zero, in which
case the joint falls back to PD tracking of the interpolated q/alpha. mc_rtc
leaves jointTorque at zero for joints its QP does not drive, so the fallback
is what keeps those joints held rather than limp.

Because the term computes that whole law itself, it takes over the entity's
PD: the configured gains (``pd_gains_path`` when given) are copied out at
construction and then zeroed, leaving mjlab's actuators as pass-through
motors fed by ``set_joint_effort_target``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mc_mjlab.actions.mc_rtc_controller_io_binding import read_pd_gains, zero_pd_gains
from mc_mjlab.actions.mc_rtc_residual_action import (
  McRtcResidualActionBase,
  McRtcResidualActionCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class McRtcResidualJointTorqueActionCfg(McRtcResidualActionCfg):
  """Configuration for mc_rtc residual joint torque control."""

  def build(self, env: ManagerBasedRlEnv) -> "McRtcResidualJointTorqueAction":
    return McRtcResidualJointTorqueAction(self, env)


class McRtcResidualJointTorqueAction(McRtcResidualActionBase):
  """mc_rtc residual action driving joint effort (torque) targets.

  The RL residual is added to the commanded torque, for joints under the
  controller's torque command and for joints on the PD fallback alike.
  """

  cfg: McRtcResidualJointTorqueActionCfg

  # Consumed in output-block order (host writes q, then alpha, then tau); the
  # PD fallback needs q/alpha alongside the torque.
  output_channels = ("q", "alpha", "tau")

  def __init__(self, cfg: McRtcResidualJointTorqueActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    # After the base applied `pd_gains_path`, so this copies the real gains.
    self._kp, self._kd = read_pd_gains(
      self._entity, self._target_names, self.num_envs, self.device
    )
    zeroed = zero_pd_gains(self._entity, self._target_names)
    print(
      f"[mc_rtc] torque control: took over the PD law for {zeroed} joint(s); "
      f"mjlab actuators now pass through the commanded effort."
    )

  def _seed_interpolation(self, env_ids: torch.Tensor) -> None:
    # Position ramps from the current stance; velocity and torque from zero.
    # A zero torque seed puts every joint on the PD fallback for the first
    # control period, so the robot holds its stance instead of going limp.
    stance = self._entity.data.joint_pos[:, self._target_ids]
    self._previous_control["q"][env_ids] = stance[env_ids]
    self._next_control["q"][env_ids] = stance[env_ids]
    for channel in ("alpha", "tau"):
      self._previous_control[channel][env_ids] = 0.0
      self._next_control[channel][env_ids] = 0.0

  def _apply_control(
    self, interpolated_control: dict[str, torch.Tensor], residual: torch.Tensor
  ) -> None:
    torque = interpolated_control["tau"]
    pd_torque = self._kp * (
      interpolated_control["q"] - self._entity.data.joint_pos[:, self._target_ids]
    ) + self._kd * (
      interpolated_control["alpha"] - self._entity.data.joint_vel[:, self._target_ids]
    )
    effort = torch.where(torque != 0.0, torque, pd_torque) + residual
    self._entity.set_joint_effort_target(effort, joint_ids=self._target_ids)
