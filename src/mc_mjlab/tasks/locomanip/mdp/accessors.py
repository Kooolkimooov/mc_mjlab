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


OBJECT_REFERENCE_POSITION = "Locomanip::objectReferencePosition"


OBJECT_REFERENCE_RPY = "Locomanip::objectReferenceRpy"


LEFT_PHASE = "Locomanip::leftPhase"


RIGHT_PHASE = "Locomanip::rightPhase"


COMPLETE = "Locomanip::complete"


def residual_action(
  env: ManagerBasedRlEnv, action_name: str = ACTION_NAME
) -> McRtcResidualActionBase:
  """Return the task's typed residual action term."""
  return residual_term(env, action_name)


def object_entity(env: ManagerBasedRlEnv, entity_name: str = OBJECT_ENTITY) -> Entity:
  """Return the manipulated free-body entity."""
  return env.scene[entity_name]


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


def to_base_frame(
  env: ManagerBasedRlEnv, vector: torch.Tensor, entity_name: str = ROBOT_ENTITY
) -> torch.Tensor:
  """Rotate a world-axis vector into the robot's root frame."""
  quat = env.scene[entity_name].data.root_link_quat_w
  return quat_apply_inverse(quat, vector)
