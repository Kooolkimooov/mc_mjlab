"""What the policy sees of the object it is manipulating."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mc_mjlab.tasks.locomanip.mdp import accessors

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def object_position_error(
  env: ManagerBasedRlEnv, action_name: str = accessors.ACTION_NAME
) -> torch.Tensor:
  """Tracking error of the commanded object position, in the robot's root frame."""
  error = accessors.object_position_error(env, action_name)
  return accessors.to_base_frame(env, error)


def object_yaw_error(
  env: ManagerBasedRlEnv, action_name: str = accessors.ACTION_NAME
) -> torch.Tensor:
  """Tracking error of the commanded object yaw, as one column."""
  return accessors.object_yaw_error(env, action_name).unsqueeze(-1)


def object_pose(
  env: ManagerBasedRlEnv, entity_name: str = accessors.OBJECT_ENTITY
) -> torch.Tensor:
  """Measured object position in the robot's root frame."""
  entity = accessors.object_entity(env, entity_name)
  robot = env.scene[accessors.ROBOT_ENTITY]
  offset = entity.data.root_link_pos_w - robot.data.root_link_pos_w
  return accessors.to_base_frame(env, offset)


def object_velocity(
  env: ManagerBasedRlEnv, entity_name: str = accessors.OBJECT_ENTITY
) -> torch.Tensor:
  """Measured object linear velocity in the robot's root frame."""
  velocity = accessors.object_entity(env, entity_name).data.root_link_lin_vel_w
  return accessors.to_base_frame(env, velocity)


def manipulation_phase(
  env: ManagerBasedRlEnv, action_name: str = accessors.ACTION_NAME
) -> torch.Tensor:
  """Both hands' numeric ManipManager phase labels."""
  term = accessors.residual_action(env, action_name)
  return torch.stack(
    (
      term.datastore_scalar_output(accessors.LEFT_PHASE),
      term.datastore_scalar_output(accessors.RIGHT_PHASE),
    ),
    dim=-1,
  )


def task_complete(
  env: ManagerBasedRlEnv, action_name: str = accessors.ACTION_NAME
) -> torch.Tensor:
  """The controller's own completion flag, as one column."""
  term = accessors.residual_action(env, action_name)
  return term.datastore_scalar_output(accessors.COMPLETE).unsqueeze(-1)


def object_mass(
  env: ManagerBasedRlEnv, entity_name: str = accessors.OBJECT_ENTITY
) -> torch.Tensor:
  """Privileged: the log of the simulated payload, which spans three decades."""
  return accessors.object_mass_kg(env, entity_name).log().unsqueeze(-1)


def hand_contact_force(
  env: ManagerBasedRlEnv, sensor_names: tuple[str, ...] = accessors.HAND_CONTACT_SENSORS
) -> torch.Tensor:
  """Privileged: the true hand-object contact force, not the robot's own sensor."""
  return accessors.hand_contact_forces(env, sensor_names)
