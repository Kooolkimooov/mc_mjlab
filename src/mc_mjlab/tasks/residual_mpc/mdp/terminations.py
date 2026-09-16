"""ResidualMPC terminations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


from mc_mjlab.tasks.residual_mpc.mdp.accessors import ROBOT_CFG, robot_entity


def excessive_base_speed(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg = ROBOT_CFG,
) -> torch.Tensor:
  """Terminate when base linear speed exceeds ``limit``."""
  velocity = robot_entity(env, asset_cfg).data.root_link_lin_vel_w
  return torch.linalg.vector_norm(velocity, dim=1) > limit


def excessive_angular_speed(
  env: ManagerBasedRlEnv,
  limit: float,
  asset_cfg: SceneEntityCfg = ROBOT_CFG,
) -> torch.Tensor:
  """Terminate when base angular speed exceeds ``limit``."""
  velocity = robot_entity(env, asset_cfg).data.root_link_ang_vel_w
  return torch.linalg.vector_norm(velocity, dim=1) > limit


def height_outside(
  env: ManagerBasedRlEnv,
  minimum: float,
  maximum: float,
  asset_cfg: SceneEntityCfg = ROBOT_CFG,
) -> torch.Tensor:
  """Terminate when root height leaves the configured interval."""
  height = robot_entity(env, asset_cfg).data.root_link_pos_w[:, 2]
  return (height < minimum) | (height > maximum)
