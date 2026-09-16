"""Shared ResidualMPC action, entity and callback access."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from mjlab.managers.scene_entity_config import SceneEntityCfg

from mc_mjlab.actions.residual_mpc_joint_torque_action import (
  ResidualMpcJointTorqueAction,
)
from mc_mjlab.bridge.controller_datastore import SUPPORT_FOOT

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


ACTION_NAME = "mc_rtc_residual"


SUPPORT_FOOT_CALLBACK = SUPPORT_FOOT


STEP_TIME_CALLBACK = "ismpc_walking::t"


STEP_DURATION_CALLBACK = "ismpc_walking::get_ts_target"


QP_OBJECTIVE_CALLBACK = "ismpc_walking::qp_objective"


ROBOT_CFG = SceneEntityCfg("robot")


PLANAR_KICK_SPEED = 0.5


YAW_KICK_RATE = 0.5


def residual_action(env: ManagerBasedRlEnv) -> ResidualMpcJointTorqueAction:
  """Return the task's typed action term."""
  return cast(ResidualMpcJointTorqueAction, env.action_manager.get_term(ACTION_NAME))


def robot_entity(env: ManagerBasedRlEnv, cfg: SceneEntityCfg) -> Entity:
  """Return the configured robot entity."""
  return env.scene[cfg.name]
