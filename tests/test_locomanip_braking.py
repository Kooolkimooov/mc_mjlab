"""Braking-only task contracts protect its action space and safety objective."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest
from conftest import requires_controller
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from mc_mjlab.actions.locomanip_motion_action import (
  MOTION_SCALE_SETTER,
  LocomanipMotionActionCfg,
)
from mc_mjlab.tasks.evaluation import evaluation_for
from mc_mjlab.tasks.locomanip.mdp import accessors
from mc_mjlab.tasks.locomanip.profiles import HRP5P
from mc_mjlab.tasks.locomanip_braking import TASK_IDS
from mc_mjlab.tasks.locomanip_motion import TASK_IDS as JOINT_TASK_IDS

if TYPE_CHECKING:
  from mjlab.rl import RslRlOnPolicyRunnerCfg

pytestmark = requires_controller(HRP5P.mc_rtc_yaml)


@pytest.mark.parametrize("robot", ["HRP5P", "JVRC1"])
@pytest.mark.parametrize("play", [False, True])
def test_only_braking_authority_with_no_placement_reward(
  robot: str, play: bool
) -> None:
  cfg = load_env_cfg(TASK_IDS[robot], play=play)
  action = cfg.actions[accessors.ACTION_NAME]
  assert isinstance(action, LocomanipMotionActionCfg)
  assert action.residual_actuator_names == ()
  assert action.scale == 1.0 and action.offset == 0.0 and action.clip is None
  assert action.datastore_scalar_inputs == (MOTION_SCALE_SETTER,)
  assert not action.datastore_vectors_inputs
  joint_action = load_env_cfg(JOINT_TASK_IDS[robot], play=play).actions[
    accessors.ACTION_NAME
  ]
  assert isinstance(joint_action, LocomanipMotionActionCfg)
  assert action.motion_decision_period_s == joint_action.motion_decision_period_s
  assert set(cfg.rewards) == {
    "contact_dcm_stability",
    "termination_penalty",
    "upright",
    "collapse_risk",
    "rapid_tilt",
    "foot_slip",
    "braking",
    "braking_start",
    "holding",
  }
  assert "actions" not in cfg.observations["actor"].terms
  assert "motion_control" in cfg.observations["actor"].terms
  assert "motion_decision" in cfg.observations
  declaration = evaluation_for(TASK_IDS[robot])
  assert declaration.success == "safe_completion"
  assert "object_position_error" in declaration.metrics
  agent = cast("RslRlOnPolicyRunnerCfg", load_rl_cfg(TASK_IDS[robot]))
  assert agent.experiment_name == TASK_IDS[robot]
  assert agent.actor.class_name == "mc_mjlab.rl.motion_actor:MotionActor"
  original = load_env_cfg(JOINT_TASK_IDS[robot], play=play)
  original_action = original.actions[accessors.ACTION_NAME]
  assert isinstance(original_action, LocomanipMotionActionCfg)
  assert original_action.residual_actuator_names
  assert "residual_magnitude" in original.rewards
