"""Study isolation and original-authority contracts."""

from __future__ import annotations

from conftest import requires_controller
from mjlab.tasks.registry import load_env_cfg

from mc_mjlab.tasks.locomanip import RESIDUAL_TASK_IDS
from mc_mjlab.tasks.locomanip.locomanip_residual_env_cfg import (
  locomanip_residual_env_cfg,
)
from mc_mjlab.tasks.locomanip.mdp.accessors import ACTION_NAME
from mc_mjlab.tasks.locomanip.profiles import HRP5P, PROFILES
from mc_mjlab.tasks.locomanip_study import CONTROL_TASK_ID

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
    robot = production.actions[ACTION_NAME].mc_rtc_robot_name
    registered = load_env_cfg(RESIDUAL_TASK_IDS[robot])
    assert registered.actions[ACTION_NAME] == production.actions[ACTION_NAME]
