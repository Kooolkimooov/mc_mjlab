"""Motion authority preserves the original joint interface and gates each row."""

from __future__ import annotations

from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from conftest import requires_controller
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from probe_locomanip_motion import _checks

from mc_mjlab.actions.locomanip_motion_action import (
  MOTION_LIMIT_GETTER,
  MOTION_SCALE_GETTER,
  MOTION_SCALE_SETTER,
  MOTION_STATE_GETTER,
  LocomanipMotionAction,
  LocomanipMotionActionCfg,
)
from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionAction,
)
from mc_mjlab.tasks.locomanip import RESIDUAL_TASK_IDS
from mc_mjlab.tasks.locomanip.mdp import accessors
from mc_mjlab.tasks.locomanip.profiles import HRP5P
from mc_mjlab.tasks.locomanip_motion import TASK_IDS
from mc_mjlab.tasks.locomanip_motion import mdp as motion_mdp

if TYPE_CHECKING:
  from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
    McRtcResidualJointPositionActionCfg,
  )

pytestmark = requires_controller(HRP5P.mc_rtc_yaml)


def motion_action() -> LocomanipMotionAction:
  """Build only the CPU action buffers needed to exercise dispatch contracts."""
  cfg = cast(
    LocomanipMotionActionCfg,
    load_env_cfg(TASK_IDS["HRP5P"]).actions[accessors.ACTION_NAME],
  )
  action = object.__new__(LocomanipMotionAction)
  action._env = cast(Any, NS(num_envs=5, device="cpu"))
  action.cfg = cfg
  action._action_dim = action._residual_action_dim = 2
  action._raw_actions = torch.zeros(5, 2)
  action._setup_action_extensions(cfg)
  action._datastore_scalar_input_columns = {MOTION_SCALE_SETTER: 0}
  action._datastore_scalar_input_feed = torch.ones(5, 1)
  action._datastore_scalar_outputs = {
    getter: torch.full((5,), accessors.HOLD_PHASE)
    for getter in cfg.motion_phase_getters
  }
  action._datastore_output_fresh = torch.ones(5, dtype=torch.bool)
  action.controller_failed = torch.zeros(5, dtype=torch.bool)
  action.controller_worker_failed = torch.zeros(5, dtype=torch.bool)
  return action


def test_new_task_keeps_joint_task_independent() -> None:
  original = load_env_cfg(RESIDUAL_TASK_IDS["HRP5P"])
  cfg = load_env_cfg(TASK_IDS["HRP5P"])
  action = cfg.actions[accessors.ACTION_NAME]
  assert isinstance(action, LocomanipMotionActionCfg)
  assert MOTION_SCALE_SETTER in action.datastore_scalar_inputs
  assert {MOTION_SCALE_GETTER, MOTION_STATE_GETTER, MOTION_LIMIT_GETTER} <= set(
    action.datastore_scalar_outputs
  )
  original_action = cast(
    "McRtcResidualJointPositionActionCfg", original.actions[accessors.ACTION_NAME]
  )
  assert not original_action.datastore_scalar_inputs
  assert "motion_control" not in original.observations["actor"].terms
  assert "motion_control" in cfg.observations["actor"].terms
  assert cfg.episode_length_s == 2 * original.episode_length_s
  assert load_rl_cfg(TASK_IDS["HRP5P"]).experiment_name == TASK_IDS["HRP5P"]


def test_zero_action_is_nominal_and_positive_action_slows_or_holds(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  action = motion_action()
  monkeypatch.setattr(
    McRtcResidualJointPositionAction,
    "process_actions",
    lambda self, requests: self._raw_actions.copy_(requests),
  )
  requests = torch.tensor(
    [[0.1, 0.2, 0.0], [0.2, 0.1, 0.5], [0, 0, 0.96], [0, 0, -1], [0, 0, 0.91]]
  )
  action.process_actions(requests)
  assert action.action_dim == 3
  assert action.motion_scale_requested.tolist() == pytest.approx([1, 0.5, 0, 1, 0.1])
  torch.testing.assert_close(action.raw_action, requests)
  torch.testing.assert_close(action._raw_actions, requests[:, :2])


def test_dispatch_requires_fresh_bilateral_hold_and_healthy_worker() -> None:
  action = motion_action()
  action._motion_requested.fill_(0.5)
  action._datastore_output_fresh[1] = False
  action._datastore_scalar_outputs[accessors.RIGHT_PHASE][2] = 3.0
  action.controller_failed[3] = True
  action.controller_worker_failed[4] = True
  action._advance_action_extensions()
  assert action.motion_scale_dispatched.tolist() == [0.5, 1, 1, 1, 1]
  assert action.datastore_scalar_input(MOTION_SCALE_SETTER).tolist() == [
    0.5,
    1,
    1,
    1,
    1,
  ]


def test_partial_reset_clears_only_selected_motion_requests() -> None:
  action = motion_action()
  action._motion_raw.fill_(1.0)
  action._motion_requested.zero_()
  action._motion_dispatched.zero_()
  action._reset_action_extensions(torch.tensor([1, 3]))
  assert action.motion_scale_requested.tolist() == [0, 1, 0, 1, 0]
  assert action.motion_scale_dispatched.tolist() == [0, 1, 0, 1, 0]
  assert action._motion_raw.tolist() == [1, 0, 1, 0, 1]
  assert action.datastore_scalar_input(MOTION_SCALE_SETTER).tolist() == [0, 1, 0, 1, 0]


def test_probe_rejects_an_incomplete_hold_and_incorrect_native_delivery() -> None:
  frame = {
    "root_z": 0.75,
    "tilt_rad": 0.1,
    "reference_x": 1.9,
    "robot_x": 1.2,
    "applied_scale": 1.0,
    "motion_state": 0.0,
    "complete": 1.0,
    "controller_failed": 0.0,
    "worker_failed": 0.0,
    "fell_over": 0.0,
    "collapsed": 0.0,
    "native_input_scale": 1.0,
    "dispatched_scale": 1.0,
    "resume_limit": 1.0,
  }
  held = {
    **frame,
    "motion_state": 2.0,
    "reference_x": 1.1,
    "complete": 0.0,
    "applied_scale": 0.0,
    "native_input_scale": 0.0,
    "dispatched_scale": 0.0,
    "resume_limit": 0.5,
  }
  samples = [
    [{**frame, "time_s": 49.0}],
    [{**frame, "time_s": 75.0, "applied_scale": 0.5}],
    [
      {**held, "time_s": 20.0},
      {**held, "time_s": 24.02},
      {**frame, "time_s": 80.0, "resume_limit": 0.5, "applied_scale": 0.5},
    ],
  ]
  assert all(_checks(samples, 4.0).values())
  samples[2][-1]["complete"] = 0.0
  assert not _checks(samples, 4.0)["all_completed"]
  samples[1][-1]["native_input_scale"] = 0.0
  assert not _checks(samples, 4.0)["actual_native_delivery"]


def test_motion_observation_exposes_controller_resume_limit(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  action = motion_action()
  action._datastore_scalar_outputs.update(
    {
      MOTION_SCALE_GETTER: torch.full((5,), 0.5),
      MOTION_STATE_GETTER: torch.zeros(5),
      MOTION_LIMIT_GETTER: torch.full((5,), 0.5),
    }
  )
  monkeypatch.setattr(accessors, "residual_action", lambda env: action)
  observed = motion_mdp.motion_observation(cast(Any, NS()))
  torch.testing.assert_close(observed, torch.tensor([[1, 0.5, 0, 0.5]] * 5))
