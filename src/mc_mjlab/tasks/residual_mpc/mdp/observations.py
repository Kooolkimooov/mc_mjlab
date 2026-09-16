"""ResidualMPC observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mc_mjlab.residuals.mpc_math import contact_phases

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


from mc_mjlab.tasks.residual_mpc.mdp.accessors import (
  QP_OBJECTIVE_CALLBACK,
  ROBOT_CFG,
  STEP_DURATION_CALLBACK,
  STEP_TIME_CALLBACK,
  SUPPORT_FOOT_CALLBACK,
  residual_action,
  robot_entity,
)


def root_position(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = ROBOT_CFG
) -> torch.Tensor:
  """Root position relative to the replicated environment origin."""
  return robot_entity(env, asset_cfg).data.root_link_pos_w - env.scene.env_origins


def root_quaternion(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = ROBOT_CFG
) -> torch.Tensor:
  """Root world quaternion in wxyz order."""
  return robot_entity(env, asset_cfg).data.root_link_quat_w


def joint_position(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = ROBOT_CFG
) -> torch.Tensor:
  """Biased encoder positions for every configured actuated joint."""
  asset = robot_entity(env, asset_cfg)
  return asset.data.joint_pos_biased[:, asset_cfg.joint_ids]


def joint_velocity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = ROBOT_CFG
) -> torch.Tensor:
  """Joint velocities for every configured actuated joint."""
  asset = robot_entity(env, asset_cfg)
  return asset.data.joint_vel[:, asset_cfg.joint_ids]


def body_linear_velocity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = ROBOT_CFG
) -> torch.Tensor:
  """Floating-base linear velocity in the body frame."""
  return robot_entity(env, asset_cfg).data.root_link_lin_vel_b


def body_angular_velocity(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = ROBOT_CFG
) -> torch.Tensor:
  """Floating-base angular velocity in the body frame."""
  return robot_entity(env, asset_cfg).data.root_link_ang_vel_b


def controller_contact_phases(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Four flat-foot phases from live ISMPC timing and support side."""
  action = residual_action(env)
  return contact_phases(
    action.datastore_scalar_output(STEP_TIME_CALLBACK),
    action.datastore_scalar_output(STEP_DURATION_CALLBACK),
    action.datastore_scalar_output(SUPPORT_FOOT_CALLBACK),
  )


def controller_qp_objective(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Last valid ISMPC QP objective."""
  return (
    residual_action(env).datastore_scalar_output(QP_OBJECTIVE_CALLBACK).unsqueeze(-1)
  )
