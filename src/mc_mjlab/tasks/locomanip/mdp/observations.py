"""What the Locomanip policy and critic see beyond proprioception."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mc_mjlab.mdp.sensors import GROUND_CONTACT_SENSORS, zmp_sensors
from mc_mjlab.tasks.locomanip.mdp import accessors

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def root_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Root height above the environment's ground plane, in metres."""
  root = env.scene[accessors.ROBOT_ENTITY].data.root_link_pos_w
  return (root[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(-1)


def root_vertical_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
  """World-vertical root velocity, in metres per second."""
  velocity = env.scene[accessors.ROBOT_ENTITY].data.root_link_lin_vel_w
  return velocity[:, 2:3]


def foot_normal_forces(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Nonnegative vertical ground loads under both feet, in kilonewtons."""
  forces = zmp_sensors(
    env, GROUND_CONTACT_SENSORS, accessors.ROBOT_ENTITY
  ).normal_forces(env)
  return forces.clamp(min=0.0) * 0.001


def com_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Privileged CoM position and velocity relative to the ground plane."""
  robot = env.scene[accessors.ROBOT_ENTITY]
  body_id = robot.indexing.root_body_id
  com = env.sim.data.subtree_com[:, body_id] - env.scene.env_origins
  velocity = env.sim.data.subtree_linvel[:, body_id]
  return torch.cat((com, velocity), dim=-1)


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
