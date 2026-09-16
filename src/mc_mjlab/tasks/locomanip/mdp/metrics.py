"""Length-independent readouts of how the manipulation actually went."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mc_mjlab.tasks.locomanip.mdp import accessors

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def object_position_error(
  env: ManagerBasedRlEnv, action_name: str = accessors.ACTION_NAME
) -> torch.Tensor:
  """Distance from the commanded object position, in metres."""
  return accessors.object_position_error(env, action_name).norm(dim=-1)


def object_yaw_error(
  env: ManagerBasedRlEnv, action_name: str = accessors.ACTION_NAME
) -> torch.Tensor:
  """Absolute angle from the commanded object yaw, in radians."""
  return accessors.object_yaw_error(env, action_name).abs()


def task_complete(
  env: ManagerBasedRlEnv, action_name: str = accessors.ACTION_NAME
) -> torch.Tensor:
  """The controller's completion flag; read it with ``reduce="last"``."""
  term = accessors.residual_action(env, action_name)
  return term.datastore_scalar_output(accessors.COMPLETE)


def hands_released(
  env: ManagerBasedRlEnv, action_name: str = accessors.ACTION_NAME
) -> torch.Tensor:
  """Whether both hands are back in the Free phase, which is label zero."""
  term = accessors.residual_action(env, action_name)
  left = term.datastore_scalar_output(accessors.LEFT_PHASE)
  right = term.datastore_scalar_output(accessors.RIGHT_PHASE)
  return ((left == 0.0) & (right == 0.0)).to(torch.float32)


def cart_mass(
  env: ManagerBasedRlEnv, entity_name: str = accessors.OBJECT_ENTITY
) -> torch.Tensor:
  """The payload this episode drew, in kilograms; read it with ``reduce="last"``."""
  return accessors.object_mass_kg(env, entity_name)
