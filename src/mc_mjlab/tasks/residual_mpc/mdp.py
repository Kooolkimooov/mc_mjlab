"""MDP terms specific to the ResidualMPC reproduction task."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from mc_mjlab.actions.residual_mpc_joint_torque_action import (
  ResidualMpcJointTorqueAction,
)
from mc_mjlab.controller_datastore import SUPPORT_FOOT
from mc_mjlab.residual_mpc import contact_phases
from mc_mjlab.tasks import mdp as shared_mdp

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.manager_base import ManagerTermBaseCfg

ACTION_NAME = "mc_rtc_residual"
# ismpc's own getter is a string; the adapter's numeric equivalent is what the
# shared-memory layout can carry. docs/coupling.md
SUPPORT_FOOT_CALLBACK = SUPPORT_FOOT
STEP_TIME_CALLBACK = "ismpc_walking::t"
STEP_DURATION_CALLBACK = "ismpc_walking::get_ts_target"
QP_OBJECTIVE_CALLBACK = "ismpc_walking::qp_objective"
_ROBOT_CFG = SceneEntityCfg("robot")

#: ResidualMPC Fig. 4 bounds the planar and yaw *norms*, not each axis.
#: docs/residual-mpc.md#INITIAL_VELOCITY_RANGE
PLANAR_KICK_SPEED = 0.5
YAW_KICK_RATE = 0.5


def _action(env: ManagerBasedRlEnv) -> ResidualMpcJointTorqueAction:
  """Return the task's typed action term."""
  return cast(ResidualMpcJointTorqueAction, env.action_manager.get_term(ACTION_NAME))


def _asset(env: ManagerBasedRlEnv, cfg: SceneEntityCfg) -> Entity:
  """Return the configured robot entity."""
  return env.scene[cfg.name]


def root_position(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
  """Root position relative to the replicated environment origin."""
  return _asset(env, asset_cfg).data.root_link_pos_w - env.scene.env_origins


def root_quaternion(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
  """Root world quaternion in wxyz order."""
  return _asset(env, asset_cfg).data.root_link_quat_w


def joint_position(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
  """Biased encoder positions for every configured actuated joint."""
  asset = _asset(env, asset_cfg)
  return asset.data.joint_pos_biased[:, asset_cfg.joint_ids]


def joint_velocity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
  """Joint velocities for every configured actuated joint."""
  asset = _asset(env, asset_cfg)
  return asset.data.joint_vel[:, asset_cfg.joint_ids]


def body_linear_velocity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
  """Floating-base linear velocity in the body frame."""
  return _asset(env, asset_cfg).data.root_link_lin_vel_b


def body_angular_velocity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
  """Floating-base angular velocity in the body frame."""
  return _asset(env, asset_cfg).data.root_link_ang_vel_b


def controller_contact_phases(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Four flat-foot phases from live ISMPC timing and support side."""
  action = _action(env)
  return contact_phases(
    action.controller_scalar(STEP_TIME_CALLBACK),
    action.controller_scalar(STEP_DURATION_CALLBACK),
    action.controller_scalar(SUPPORT_FOOT_CALLBACK),
  )


def controller_qp_objective(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Last valid ISMPC QP objective."""
  return _action(env).controller_scalar(QP_OBJECTIVE_CALLBACK).unsqueeze(-1)


def linear_velocity_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """ResidualMPC Table I normalized planar tracking reward."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  velocity = _asset(env, asset_cfg).data.root_link_lin_vel_b[:, :2]
  error = (command[:, :2] - velocity) / (1.0 + command[:, :2].abs())
  return torch.exp(-torch.sum(torch.square(error), dim=1) / sigma)


def angular_velocity_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """ResidualMPC Table I yaw-rate tracking reward."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  yaw_rate = _asset(env, asset_cfg).data.root_link_ang_vel_b[:, 2]
  return torch.exp(-torch.square(command[:, 2] - yaw_rate) / sigma)


def first_action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared first action derivative in physical radians per second."""
  action = _action(env)
  rate = (action.requested_joint_action - action.previous_joint_action) / env.step_dt
  return torch.sum(torch.square(rate), dim=1)


def second_action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared printed second action difference divided by policy dt."""
  action = _action(env)
  rate = (
    action.requested_joint_action
    - 2.0 * action.previous_joint_action
    + action.second_previous_joint_action
  ) / env.step_dt
  return torch.sum(torch.square(rate), dim=1)


def torque_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Mean squared applied effort across the decimation window."""
  return torch.sum(_action(env).mean_squared_effort, dim=1)


def orientation_reward(
  env: ManagerBasedRlEnv,
  sigma: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Exponential projected-gravity orientation reward."""
  gravity_xy = _asset(env, asset_cfg).data.projected_gravity_b[:, :2]
  return torch.exp(-torch.sum(torch.square(gravity_xy), dim=1) / sigma)


def height_reward(
  env: ManagerBasedRlEnv,
  target_height: float,
  sigma: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Exponential nominal root-height reward."""
  height = _asset(env, asset_cfg).data.root_link_pos_w[:, 2]
  return torch.exp(-torch.square(target_height - height) / sigma)


def joint_regularization(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
  """Raw mean squared joint deviation from the nominal stance."""
  asset = _asset(env, asset_cfg)
  nominal = asset.data.default_joint_pos
  assert nominal is not None
  error = asset.data.joint_pos[:, asset_cfg.joint_ids] - nominal[:, asset_cfg.joint_ids]
  return torch.mean(torch.square(error), dim=1)


def self_collision(
  env: ManagerBasedRlEnv, sensor_name: str, force_threshold: float = 10.0
) -> torch.Tensor:
  """Per-environment indicator of any self-contact above threshold."""
  sensor = env.scene[sensor_name]
  assert isinstance(sensor, ContactSensor)
  force = sensor.data.force_history
  if force is not None:
    return (torch.linalg.vector_norm(force, dim=-1) > force_threshold).any(dim=(1, 2))
  found = sensor.data.found
  assert found is not None
  return found.any(dim=1)


def excessive_base_speed(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Terminate when base linear speed exceeds ``limit``."""
  velocity = _asset(env, asset_cfg).data.root_link_lin_vel_w
  return torch.linalg.vector_norm(velocity, dim=1) > limit


def excessive_angular_speed(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Terminate when base angular speed exceeds ``limit``."""
  velocity = _asset(env, asset_cfg).data.root_link_ang_vel_w
  return torch.linalg.vector_norm(velocity, dim=1) > limit


def height_outside(
  env: ManagerBasedRlEnv,
  minimum: float,
  maximum: float,
  asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
  """Terminate when root height leaves the configured interval."""
  height = _asset(env, asset_cfg).data.root_link_pos_w[:, 2]
  return (height < minimum) | (height > maximum)


def controller_failed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Terminate on a controller QP failure."""
  return _action(env).controller_failed


def controller_worker_failed(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Truncate on an exogenous controller worker failure."""
  return _action(env).controller_worker_failed


def refresh_action_scaling(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> None:
  """Sync randomized effort limits and PD gains into the action scale."""
  del env_ids
  _action(env).refresh_effort_limits_and_action_scale()


def projection_fraction(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Fraction of leg actions whose blended torque was projected."""
  return _action(env).projection_mask.float().mean(dim=1)


class survival_kick_curriculum:
  """Move the kick's difficulty on smoothed survival, as Ranjbar's curriculum does."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    kick = env.event_manager.get_term_cfg(
      cfg.params.get("term_name", "initial_base_velocity")
    ).func
    if not isinstance(kick, initial_velocity_kick):
      raise TypeError("the survival curriculum needs an initial_velocity_kick term")
    self._kick = kick
    self.smoothed = float("nan")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    advance_fraction: float = 0.7,
    regress_fraction: float = 0.6,
    increment: float = 0.05,
    scale_range: tuple[float, float] = (0.4, 2.0),
    smoothing: float = 0.02,
    term_name: str = "initial_base_velocity",
  ) -> dict[str, float]:
    """Raise the kick above the advance rate, lower it below the regress rate."""
    del term_name
    if advance_fraction <= regress_fraction:
      raise ValueError("the advance fraction must leave a deadband above regress")
    # `curriculum_manager.compute` runs first in `_reset_idx`, so the termination
    # buffers still describe exactly the envs that just finished.
    if env_ids is not None and not isinstance(env_ids, slice):
      survived = env.termination_manager.get_term("time_out")[env_ids]
      if survived.numel():
        sample = float(survived.to(torch.float32).mean())
        self.smoothed = (
          sample
          if math.isnan(self.smoothed)
          else (1.0 - smoothing) * self.smoothed + smoothing * sample
        )
        if self.smoothed > advance_fraction:
          self._kick.scale = min(scale_range[1], self._kick.scale + increment)
        elif self.smoothed < regress_fraction:
          self._kick.scale = max(scale_range[0], self._kick.scale - increment)
    return {"kick_scale": self._kick.scale, "survival": self.smoothed}


class initial_velocity_kick(shared_mdp.push_and_record):
  """One base-velocity kick per episode, withheld until the FSM is walking."""

  #: Difficulty dial the survival curriculum drives.
  #: docs/residual-feedback.md#survival_kick_curriculum
  scale: float = 1.0

  def __call__(  # type: ignore[override]
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    warmup_s: float = 0.0,
    asset_cfg: SceneEntityCfg | None = None,
  ) -> None:
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    age = env.episode_length_buf[ids]
    # `last_push_step` is global while `age` is per-episode, so this is "the last
    # kick happened inside the current episode" rather than "ever".
    kicked = (env.common_step_counter - self.last_push_step[ids]) < age
    ids = ids[(age * env.step_dt >= warmup_s) & ~kicked]
    if ids.numel() == 0:
      return
    self._kick(env, ids, asset_cfg)

  def _kick(
    self,
    env: ManagerBasedRlEnv,
    ids: torch.Tensor,
    asset_cfg: SceneEntityCfg | None,
  ) -> None:
    """Sample uniformly inside the paper's norm ball and add it to the base."""
    asset = env.scene[(asset_cfg or _ROBOT_CFG).name]
    # Fig. 4 bounds the *norms*, so sampling each axis independently would put
    # 21.6% of draws outside the ball. docs/residual-mpc.md#INITIAL_VELOCITY_RANGE
    vel_w = asset.data.root_link_vel_w[ids]
    delta = torch.zeros_like(vel_w)
    angle = 2.0 * torch.pi * torch.rand(len(ids), device=env.device)
    # sqrt(u) keeps the disc uniform; without it the samples crowd the rim.
    radius = (
      PLANAR_KICK_SPEED
      * self.scale
      * torch.sqrt(torch.rand(len(ids), device=env.device))
    )
    delta[:, 0] = radius * torch.cos(angle)
    delta[:, 1] = radius * torch.sin(angle)
    yaw = YAW_KICK_RATE * self.scale
    delta[:, 5] = torch.empty(len(ids), device=env.device).uniform_(-yaw, yaw)
    # `root_link_vel_w` comes from `cvel`, which MuJoCo does not recompute until
    # the next forward, so a before/after difference would read zero.
    asset.write_root_link_velocity_to_sim(vel_w + delta, env_ids=ids)
    self.last_push_vel[ids] = quat_apply_inverse(
      asset.data.root_link_quat_w[ids], delta[:, :3]
    )
    self.last_push_step[ids] = env.common_step_counter


def forward_speed(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _ROBOT_CFG
) -> torch.Tensor:
  """Base forward speed, so the gait is readable without inferring it from error."""
  return _asset(env, asset_cfg).data.root_link_lin_vel_b[:, 0]


def commanded_speed(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Commanded forward speed; the companion forward_speed is read against it."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command[:, 0]


def maximum_effort_ratio(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Maximum final-effort ratio against the active randomized limit."""
  action = _action(env)
  return (action.final_effort.abs() / action.effort_limit).amax(dim=1)
