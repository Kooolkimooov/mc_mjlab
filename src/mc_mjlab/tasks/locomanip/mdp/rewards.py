"""The task objective: keep the object on the trajectory the controller planned."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mc_mjlab.tasks.locomanip.mdp import accessors

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def object_position_tracking(
  env: ManagerBasedRlEnv, std: float, action_name: str = accessors.ACTION_NAME
) -> torch.Tensor:
  """Gaussian kernel on the object's distance from its commanded position."""
  error = accessors.object_position_error(env, action_name)
  return torch.exp(-error.square().sum(dim=-1) / std**2)


def object_yaw_tracking(
  env: ManagerBasedRlEnv, std: float, action_name: str = accessors.ACTION_NAME
) -> torch.Tensor:
  """Gaussian kernel on the object's angle from its commanded yaw."""
  error = accessors.object_yaw_error(env, action_name)
  return torch.exp(-error.square() / std**2)
