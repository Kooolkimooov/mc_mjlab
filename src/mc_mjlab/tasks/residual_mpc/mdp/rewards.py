"""ResidualMPC rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from mc_mjlab.tasks.residual_mpc.mdp.accessors import residual_action

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


from mc_mjlab.tasks.residual_mpc.mdp.accessors import ROBOT_CFG, robot_entity


def linear_velocity_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma: float,
  asset_cfg: SceneEntityCfg = ROBOT_CFG,
) -> torch.Tensor:
  """ResidualMPC Table I normalized planar tracking reward."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  velocity = robot_entity(env, asset_cfg).data.root_link_lin_vel_b[:, :2]
  error = (command[:, :2] - velocity) / (1.0 + command[:, :2].abs())
  return torch.exp(-torch.sum(torch.square(error), dim=1) / sigma)


def angular_velocity_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  sigma: float,
  asset_cfg: SceneEntityCfg = ROBOT_CFG,
) -> torch.Tensor:
  """ResidualMPC Table I yaw-rate tracking reward."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  yaw_rate = robot_entity(env, asset_cfg).data.root_link_ang_vel_b[:, 2]
  return torch.exp(-torch.square(command[:, 2] - yaw_rate) / sigma)


def first_action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared first action derivative in physical radians per second."""
  action = residual_action(env)
  rate = (action.requested_joint_action - action.previous_joint_action) / env.step_dt
  return torch.sum(torch.square(rate), dim=1)


def second_action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared printed second action difference divided by policy dt."""
  action = residual_action(env)
  rate = (
    action.requested_joint_action
    - 2.0 * action.previous_joint_action
    + action.second_previous_joint_action
  ) / env.step_dt
  return torch.sum(torch.square(rate), dim=1)


def torque_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Mean squared applied effort across the decimation window."""
  return torch.sum(residual_action(env).mean_squared_effort, dim=1)


def orientation_reward(
  env: ManagerBasedRlEnv,
  sigma: float,
  asset_cfg: SceneEntityCfg = ROBOT_CFG,
) -> torch.Tensor:
  """Exponential projected-gravity orientation reward."""
  gravity_xy = robot_entity(env, asset_cfg).data.projected_gravity_b[:, :2]
  return torch.exp(-torch.sum(torch.square(gravity_xy), dim=1) / sigma)


def height_reward(
  env: ManagerBasedRlEnv,
  target_height: float,
  sigma: float,
  asset_cfg: SceneEntityCfg = ROBOT_CFG,
) -> torch.Tensor:
  """Exponential nominal root-height reward."""
  height = robot_entity(env, asset_cfg).data.root_link_pos_w[:, 2]
  return torch.exp(-torch.square(target_height - height) / sigma)


def joint_regularization(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = ROBOT_CFG
) -> torch.Tensor:
  """Raw mean squared joint deviation from the nominal stance."""
  asset = robot_entity(env, asset_cfg)
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
