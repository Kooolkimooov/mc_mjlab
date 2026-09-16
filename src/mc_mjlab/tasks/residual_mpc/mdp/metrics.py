"""ResidualMPC metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mc_mjlab.tasks.residual_mpc.mdp.accessors import residual_action

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


from mc_mjlab.tasks.residual_mpc.mdp.accessors import ROBOT_CFG, robot_entity


def forward_speed(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = ROBOT_CFG
) -> torch.Tensor:
  """Base forward speed, so the gait is readable without inferring it from error."""
  return robot_entity(env, asset_cfg).data.root_link_lin_vel_b[:, 0]


def commanded_speed(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Commanded forward speed; the companion forward_speed is read against it."""
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return command[:, 0]


def active_effort_ratio(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Maximum final-effort ratio against the active randomized limit."""
  action = residual_action(env)
  return (action.final_effort.abs() / action.effort_limit).amax(dim=1)
