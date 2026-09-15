"""ResidualMPC joint-action to torque-blending action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase
from mc_mjlab.actions.walking_reference_action import (
  AbsoluteWalkingReferenceActionCfg,
  AbsoluteWalkingReferenceMixin,
)
from mc_mjlab.residual_mpc import (
  advance_action_history,
  paper_joint_action_scale,
  paper_torque_blend,
)
from mc_mjlab.residual_safety import project_residual
from utils.pd_gains import read_pd_gains, zero_pd_gains

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ResidualMpcJointTorqueActionCfg(AbsoluteWalkingReferenceActionCfg):
  """Configuration for paper-style joint-action to torque blending."""

  blend_factor: float = 0.1
  """Initial per-environment torque blend factor lambda."""

  action_scale_blend_factor: float = 0.1
  """Lambda used to convert normalized actions into joint radians."""

  def build(self, env: ManagerBasedRlEnv) -> "ResidualMpcJointTorqueAction":
    return ResidualMpcJointTorqueAction(self, env)


class ResidualMpcJointTorqueAction(
  AbsoluteWalkingReferenceMixin, McRtcResidualActionBase
):
  """Blend paper-style leg posture torques onto the mc_rtc nominal effort."""

  cfg: ResidualMpcJointTorqueActionCfg
  output_channels = ("q", "alpha", "tau")
  residual_unit = "Nm"

  def __init__(
    self, cfg: ResidualMpcJointTorqueActionCfg, env: ManagerBasedRlEnv
  ) -> None:
    if not 0.0 <= cfg.blend_factor <= 1.0:
      raise ValueError("blend_factor must be in [0, 1]")
    if cfg.action_scale_blend_factor <= 0.0:
      raise ValueError("action_scale_blend_factor must be positive")
    super().__init__(cfg, env)
    self._kp, self._kd = read_pd_gains(
      self._entity, self._target_names, self.num_envs, self.device
    )
    zeroed = zero_pd_gains(self._entity, self._target_names)
    self.blend_factor = torch.full(
      (self.num_envs,), cfg.blend_factor, device=self.device
    )
    self._residual_mask = torch.zeros(1, self._num_targets, device=self.device)
    ids = self._residual_ids
    self._residual_mask[:, :] = 1.0 if ids is None else 0.0
    if ids is not None:
      self._residual_mask[:, ids] = 1.0
    self._residual_torque = torch.zeros(
      self.num_envs, self._num_targets, device=self.device
    )
    self._blended_residual_torque = torch.zeros_like(self._residual_torque)
    self._nominal_torque = torch.zeros_like(self._residual_torque)
    self._final_effort = torch.zeros_like(self._residual_torque)
    self._effort_sq_sum = torch.zeros_like(self._residual_torque)
    self._effort_substeps = 0
    self._previous_joint_action = torch.zeros_like(self._processed_actions)
    self._second_previous_joint_action = torch.zeros_like(self._processed_actions)
    self.refresh_effort_limits_and_action_scale()
    print(
      f"[mc_rtc] ResidualMPC torque control: took over the PD law for "
      f"{zeroed} joint(s); lambda={cfg.blend_factor:.3f}."
    )

  def refresh_effort_limits_and_action_scale(self) -> None:
    """Sync randomized limits and recompute the paper's per-joint radian scale."""
    lower = self._effort_lower.expand(self.num_envs, -1).clone()
    upper = self._effort_upper.expand(self.num_envs, -1).clone()
    index_of = {name: index for index, name in enumerate(self._target_names)}
    for actuator in self._entity.actuators:
      force_limit = getattr(actuator, "force_limit", None)
      if force_limit is None:
        continue
      for column, name in enumerate(actuator.target_names):
        target = index_of.get(name)
        if target is not None:
          upper[:, target] = force_limit[:, column]
          lower[:, target] = -force_limit[:, column]
    self._effort_lower, self._effort_upper = lower, upper
    ids = self._residual_ids
    effort = upper.abs() if ids is None else upper[:, ids].abs()
    kp = self._kp if ids is None else self._kp[:, ids]
    scale = paper_joint_action_scale(effort, kp, self.cfg.action_scale_blend_factor)
    self._scale = scale
    self._physical_scale.copy_(scale)

  def set_blend_factor(
    self, env_ids: torch.Tensor | slice, value: float | torch.Tensor
  ) -> None:
    """Set lambda for selected environments without resetting controllers."""
    values = torch.as_tensor(value, device=self.device, dtype=self.blend_factor.dtype)
    if not bool(torch.isfinite(values).all()) or bool(
      ((values < 0.0) | (values > 1.0)).any()
    ):
      raise ValueError("blend factor values must be finite and in [0, 1]")
    self.blend_factor[env_ids] = values

  def process_actions(self, actions: torch.Tensor) -> None:
    # A policy step begins, so the previous window's effort accumulator is spent.
    self._effort_sq_sum.zero_()
    self._effort_substeps = 0
    previous, second_previous = advance_action_history(
      self._processed_actions, self._previous_joint_action
    )
    self._previous_joint_action.copy_(previous)
    self._second_previous_joint_action.copy_(second_previous)
    super().process_actions(actions.clamp(-1.0, 1.0))

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      env_ids = slice(None)
    self._processed_actions[env_ids] = 0.0
    self._previous_joint_action[env_ids] = 0.0
    self._second_previous_joint_action[env_ids] = 0.0
    self._residual_torque[env_ids] = 0.0
    self._blended_residual_torque[env_ids] = 0.0
    self._nominal_torque[env_ids] = 0.0
    self._final_effort[env_ids] = 0.0
    self._effort_sq_sum[env_ids] = 0.0

  def _seed_interpolation(self, env_ids: torch.Tensor) -> None:
    stance = self._entity.data.joint_pos_biased[:, self._target_ids]
    self._previous_control["q"][env_ids] = stance[env_ids]
    self._next_control["q"][env_ids] = stance[env_ids]
    for channel in ("alpha", "tau"):
      self._previous_control[channel][env_ids] = 0.0
      self._next_control[channel][env_ids] = 0.0

  def _apply_control(
    self, interpolated_control: dict[str, torch.Tensor], residual: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    joint_action = residual
    q_biased = self._entity.data.joint_pos_biased[:, self._target_ids]
    qd = self._entity.data.joint_vel[:, self._target_ids]
    nominal, residual, blended = paper_torque_blend(
      interpolated_control["q"],
      self._entity.data.default_joint_pos[:, self._target_ids],
      q_biased,
      qd,
      interpolated_control["alpha"],
      interpolated_control["tau"],
      joint_action,
      self._kp,
      self._kd,
      self.blend_factor,
      self._residual_mask,
    )
    effort, executed, projected = project_residual(
      nominal, blended, self._effort_lower, self._effort_upper
    )
    self._nominal_torque.copy_(nominal)
    self._residual_torque.copy_(residual)
    self._blended_residual_torque.copy_(blended)
    self._final_effort.copy_(effort)
    # `torque_l2` scores the whole decimation window, not the last substep:
    # the 500 Hz peaks between policy steps are real applied control.
    self._effort_sq_sum += effort.square()
    self._effort_substeps += 1
    self._entity.set_joint_effort_target(effort, joint_ids=self._target_ids)
    return executed, projected

  @property
  def mean_squared_effort(self) -> torch.Tensor:
    """Mean squared applied effort across the decimation window."""
    if self._effort_substeps == 0:
      return self._final_effort.square()
    return self._effort_sq_sum / self._effort_substeps

  @property
  def requested_joint_action(self) -> torch.Tensor:
    """Current physical leg-joint action in radians."""
    return self._processed_actions

  @property
  def previous_joint_action(self) -> torch.Tensor:
    """Physical leg-joint action from the preceding policy step."""
    return self._previous_joint_action

  @property
  def second_previous_joint_action(self) -> torch.Tensor:
    """Physical leg-joint action from two policy steps ago."""
    return self._second_previous_joint_action

  @property
  def residual_torque(self) -> torch.Tensor:
    """Unblended paper posture residual over every actuated joint."""
    return self._residual_torque

  @property
  def blended_residual_torque(self) -> torch.Tensor:
    """Lambda-scaled residual before effort projection."""
    return self._blended_residual_torque

  @property
  def nominal_torque(self) -> torch.Tensor:
    """Controller torque or PD fallback over every actuated joint."""
    return self._nominal_torque

  @property
  def final_effort(self) -> torch.Tensor:
    """Final effort after hardware-limit projection."""
    return self._final_effort

  @property
  def effort_limit(self) -> torch.Tensor:
    """Active absolute effort limit over every actuated joint."""
    return torch.maximum(self._effort_lower.abs(), self._effort_upper.abs())
