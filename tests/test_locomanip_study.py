"""Study isolation and original-authority contracts."""

from __future__ import annotations

import re

import pytest
from conftest import requires_controller
from mjlab.tasks.registry import load_env_cfg

from mc_mjlab.actions.grouped_residual_action import (
  GroupedResidualJointPositionActionCfg,
  group_of_joint,
)
from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionActionCfg,
)
from mc_mjlab.bridge.config import get_main_robot_name
from mc_mjlab.tasks.locomanip import RESIDUAL_TASK_IDS
from mc_mjlab.tasks.locomanip.locomanip_residual_env_cfg import (
  locomanip_residual_env_cfg,
)
from mc_mjlab.tasks.locomanip.mdp.accessors import ACTION_NAME
from mc_mjlab.tasks.locomanip.profiles import HRP5P, PROFILES
from mc_mjlab.tasks.locomanip_study import CONTROL_TASK_ID, TASK_IDS
from mc_mjlab.tasks.locomanip_study.config import INSTANTANEOUS_TERMS, STUDY_GROUPS

pytestmark = requires_controller(HRP5P.mc_rtc_yaml)


def test_study_control_preserves_production_and_hardware_authority() -> None:
  control = load_env_cfg(CONTROL_TASK_ID)
  original = locomanip_residual_env_cfg(HRP5P, apply_scale_gains=False)
  assert control.actions[ACTION_NAME] == original.actions[ACTION_NAME]
  assert control.rewards == original.rewards
  assert control.observations == original.observations
  assert CONTROL_TASK_ID not in RESIDUAL_TASK_IDS.values()
  for profile in PROFILES:
    production = locomanip_residual_env_cfg(profile)
    robot = get_main_robot_name(profile.mc_rtc_yaml)
    registered = load_env_cfg(RESIDUAL_TASK_IDS[robot])
    assert registered.actions[ACTION_NAME] == production.actions[ACTION_NAME]


def test_study_variants_change_only_the_residual_joints() -> None:
  control = load_env_cfg(CONTROL_TASK_ID)
  original = control.actions[ACTION_NAME]
  assert isinstance(original, McRtcResidualJointPositionActionCfg)
  residual = original.residual_actuator_names
  assert residual is not None
  expected = [
    n for n in residual if any(re.fullmatch(p, n) for p in STUDY_GROUPS.values())
  ]
  assert len(expected) == 13
  for suffix in ("arms-waist", "coordinated"):
    variant = load_env_cfg(TASK_IDS[suffix])
    action = variant.actions[ACTION_NAME]
    assert isinstance(action, McRtcResidualJointPositionActionCfg)
    patterns = action.residual_actuator_names
    assert patterns is not None
    assert [
      n for n in residual if any(re.fullmatch(p, n) for p in patterns)
    ] == expected
    assert action.scale == original.scale
    assert variant.rewards == control.rewards
    assert variant.observations == control.observations
    assert variant.events == control.events
  coordinated = load_env_cfg(TASK_IDS["coordinated"]).actions[ACTION_NAME]
  assert isinstance(coordinated, GroupedResidualJointPositionActionCfg)
  assert group_of_joint(tuple(expected), coordinated.residual_groups) == [
    0 if re.fullmatch(STUDY_GROUPS["arms"], n) else 1 for n in expected
  ]


def test_grouping_rejects_gaps_and_overlaps() -> None:
  names = ("RSP", "LEP", "WP")
  with pytest.raises(ValueError, match="outside"):
    group_of_joint(names, ((STUDY_GROUPS["arms"],),))
  with pytest.raises(ValueError, match="two"):
    group_of_joint(names, ((".*",), ("WP",)))


@pytest.mark.parametrize("frames", [10, 20])
def test_history_variants_share_channels_and_keep_discrete_terms_instant(
  frames: int,
) -> None:
  control = load_env_cfg(CONTROL_TASK_ID)
  variant = load_env_cfg(TASK_IDS[f"history-{frames}"])
  assert variant.actions == control.actions and variant.rewards == control.rewards
  actor = variant.observations["actor"].terms
  critic = variant.observations["critic"].terms
  assert list(actor) == list(control.observations["actor"].terms)
  for name, term in actor.items():
    original = control.observations["actor"].terms[name].history_length
    expected = original if name in INSTANTANEOUS_TERMS else frames
    assert term.history_length == expected, name
    assert critic[name].history_length == term.history_length, name
    assert term.noise == control.observations["actor"].terms[name].noise
  for name in critic.keys() - actor.keys():
    assert critic[name] == control.observations["critic"].terms[name]
