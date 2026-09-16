"""Shared Locomanip action, object entity and datastore-callback access."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply_inverse, wrap_to_pi

from mc_mjlab.mdp.sensors import residual_term

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

  from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase


ACTION_NAME = "mc_rtc_residual"


ROBOT_ENTITY = "robot"


#: The scene entity fed to mc_rtc as the ``obj`` robot. docs/locomanip.md#controller_objects
OBJECT_ENTITY = "cart"


#: The cart asset's only body, which the payload event and metric both name.
OBJECT_BODY = "Body"


REQUIRED_CONTROLLER = "LocomanipController"


#: The scene's hand-object contact sensors, in left-then-right order.
HAND_CONTACT_SENSORS = ("left_hand_cart", "right_hand_cart")


OBJECT_REFERENCE_POSITION = "Locomanip::objectReferencePosition"


OBJECT_REFERENCE_RPY = "Locomanip::objectReferenceRpy"


LEFT_PHASE = "Locomanip::leftPhase"


RIGHT_PHASE = "Locomanip::rightPhase"


COMPLETE = "Locomanip::complete"


#: ManipPhaseLabel::Hold, the phase in which a hand carries the object.
HOLD_PHASE = 4.0


def residual_action(
  env: ManagerBasedRlEnv, action_name: str = ACTION_NAME
) -> McRtcResidualActionBase:
  """Return the task's typed residual action term."""
  return residual_term(env, action_name)


def object_entity(env: ManagerBasedRlEnv, entity_name: str = OBJECT_ENTITY) -> Entity:
  """Return the manipulated free-body entity."""
  return env.scene[entity_name]


def holding(env: ManagerBasedRlEnv, action_name: str = ACTION_NAME) -> torch.Tensor:
  """Whether both hands carry the object, as a 0/1 mask."""
  term = residual_action(env, action_name)
  left = term.datastore_scalar_output(LEFT_PHASE)
  right = term.datastore_scalar_output(RIGHT_PHASE)
  return ((left == HOLD_PHASE) & (right == HOLD_PHASE)).to(torch.float32)


def object_position_error(
  env: ManagerBasedRlEnv,
  action_name: str = ACTION_NAME,
  entity_name: str = OBJECT_ENTITY,
) -> torch.Tensor:
  """Commanded object position minus the measured one, in environment-local axes."""
  reference = residual_action(env, action_name).datastore_vector_output(
    OBJECT_REFERENCE_POSITION
  )
  measured = object_entity(env, entity_name).data.root_link_pos_w
  return reference - (measured - env.scene.env_origins)


def object_yaw_error(
  env: ManagerBasedRlEnv,
  action_name: str = ACTION_NAME,
  entity_name: str = OBJECT_ENTITY,
) -> torch.Tensor:
  """Commanded object yaw minus the measured one, wrapped to (-pi, pi]."""
  reference = residual_action(env, action_name).datastore_vector_output(
    OBJECT_REFERENCE_RPY
  )[:, 2]
  quat = object_entity(env, entity_name).data.root_link_quat_w
  _, _, measured = euler_xyz_from_quat(quat)
  return wrap_to_pi(reference - measured)


def object_mass_kg(
  env: ManagerBasedRlEnv, entity_name: str = OBJECT_ENTITY
) -> torch.Tensor:
  """The mass MuJoCo is simulating for the object, in kilograms."""
  body = env.sim.mj_model.body(f"{entity_name}/{OBJECT_BODY}").id
  return torch.as_tensor(env.sim.model.body_mass[:, int(body)])


def hand_contact_forces(
  env: ManagerBasedRlEnv, sensor_names: tuple[str, ...] = HAND_CONTACT_SENSORS
) -> torch.Tensor:
  """Net contact force between each hand and the object, flattened per hand."""
  forces = [env.scene[name].data.force.sum(dim=-2) for name in sensor_names]
  return torch.cat(forces, dim=-1)


def to_base_frame(
  env: ManagerBasedRlEnv, vector: torch.Tensor, entity_name: str = ROBOT_ENTITY
) -> torch.Tensor:
  """Rotate a world-axis vector into the robot's root frame."""
  quat = env.scene[entity_name].data.root_link_quat_w
  return quat_apply_inverse(quat, vector)
