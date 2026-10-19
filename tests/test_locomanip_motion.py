"""Motion authority preserves the original joint interface and gates each row."""

from __future__ import annotations

from types import SimpleNamespace as NS
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from conftest import requires_controller
from mjlab.managers.reward_manager import RewardManager
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from probe_locomanip_motion import _checks

from mc_mjlab.actions.locomanip_motion_action import (
  MOTION_COUNT_GETTER,
  MOTION_DELAY_GETTER,
  MOTION_FILTER_GETTER,
  MOTION_SCALE_GETTER,
  MOTION_SCALE_SETTER,
  MOTION_STATE_GETTER,
  MOTION_TARGET_GETTER,
  LocomanipMotionAction,
  LocomanipMotionActionCfg,
)
from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionAction,
)
from mc_mjlab.tasks.locomanip import RESIDUAL_TASK_IDS
from mc_mjlab.tasks.locomanip.locomanip_residual_env_cfg import (
  POSITION_REQUEST_WEIGHT,
  REQUEST_WEIGHT,
)
from mc_mjlab.tasks.locomanip.mdp import accessors
from mc_mjlab.tasks.locomanip.profiles import HRP5P
from mc_mjlab.tasks.locomanip_motion import TASK_IDS
from mc_mjlab.tasks.locomanip_motion import mdp as motion_mdp

if TYPE_CHECKING:
  from mjlab.rl import RslRlOnPolicyRunnerCfg

  from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
    McRtcResidualJointPositionActionCfg,
  )

pytestmark = requires_controller(HRP5P.mc_rtc_yaml)


def motion_action(residual_dim: int = 2) -> LocomanipMotionAction:
  """Build only the CPU action buffers needed to exercise dispatch contracts."""
  cfg = cast(
    LocomanipMotionActionCfg,
    load_env_cfg(TASK_IDS["HRP5P"]).actions[accessors.ACTION_NAME],
  )
  action = object.__new__(LocomanipMotionAction)
  action._env = cast(Any, NS(num_envs=5, device="cpu", step_dt=0.02))
  action.cfg = cfg
  action._action_dim = action._residual_action_dim = residual_dim
  action._raw_actions = torch.zeros(5, residual_dim)
  action._setup_action_extensions(cfg)
  action._datastore_scalar_input_columns = {MOTION_SCALE_SETTER: 0}
  action._datastore_scalar_input_feed = torch.ones(5, 1)
  action._datastore_scalar_outputs = {
    getter: torch.zeros(5) for getter in cfg.datastore_scalar_outputs
  }
  for getter in cfg.motion_phase_getters:
    action._datastore_scalar_outputs[getter].fill_(accessors.HOLD_PHASE)
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
  assert {MOTION_SCALE_GETTER, MOTION_STATE_GETTER} <= set(
    action.datastore_scalar_outputs
  )
  assert "Locomanip::motionScaleLimit" not in action.datastore_scalar_outputs
  original_action = cast(
    "McRtcResidualJointPositionActionCfg", original.actions[accessors.ACTION_NAME]
  )
  assert not original_action.datastore_scalar_inputs
  assert "motion_control" not in original.observations["actor"].terms
  assert "motion_control" in cfg.observations["actor"].terms
  assert cfg.episode_length_s == 2 * original.episode_length_s
  assert load_rl_cfg(TASK_IDS["HRP5P"]).experiment_name == TASK_IDS["HRP5P"]


@pytest.mark.parametrize("residual_dim", [0, 2])
def test_zero_action_is_nominal_and_positive_action_slows_or_holds(
  monkeypatch: pytest.MonkeyPatch, residual_dim: int
) -> None:
  action = motion_action(residual_dim)
  monkeypatch.setattr(
    McRtcResidualJointPositionAction,
    "process_actions",
    lambda self, requests: self._raw_actions.copy_(requests),
  )
  requests = torch.tensor(
    [[0.1, 0.2, 0.0], [0.2, 0.1, 0.5], [0, 0, 0.96], [0, 0, -1], [0, 0, 0.91]]
  )
  if residual_dim == 0:
    requests = requests[:, -1:]
  action.process_actions(requests)
  assert action.action_dim == residual_dim + 1
  assert action.motion_scale_requested.tolist() == pytest.approx([1, 0.5, 0, 1, 0.1])
  torch.testing.assert_close(action.raw_action, requests)
  torch.testing.assert_close(action._raw_actions, requests[:, :residual_dim])


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
  }
  held = {
    **frame,
    "motion_state": 2.0,
    "reference_x": 1.1,
    "complete": 0.0,
    "applied_scale": 0.0,
    "native_input_scale": 0.0,
    "dispatched_scale": 0.0,
  }
  samples = [
    [{**frame, "time_s": 49.0}],
    [{**frame, "time_s": 75.0, "applied_scale": 0.5}],
    [
      {**held, "time_s": 20.0},
      {**held, "time_s": 24.02},
      {**frame, "time_s": 80.0, "applied_scale": 1.0},
    ],
  ]
  assert all(_checks(samples, 4.0).values())
  samples[2][-1]["applied_scale"] = 0.5
  assert not _checks(samples, 4.0)["post_hold_requested_scale_observed"]
  assert all(_checks(samples, 4.0, resume_scale=0.5).values())
  samples[2][-1]["applied_scale"] = 1.0
  samples[2][-1]["complete"] = 0.0
  assert not _checks(samples, 4.0)["all_completed"]
  samples[1][-1]["native_input_scale"] = 0.0
  assert not _checks(samples, 4.0)["actual_native_delivery"]


def test_motion_observation_exposes_native_intent_and_execution(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  action = motion_action()
  action._datastore_scalar_outputs.update(
    {
      MOTION_SCALE_GETTER: torch.full((5,), 0.5),
      MOTION_STATE_GETTER: torch.zeros(5),
      MOTION_TARGET_GETTER: torch.full((5,), 0.5),
      MOTION_FILTER_GETTER: torch.full((5,), 0.6),
      MOTION_DELAY_GETTER: torch.full((5,), 1.5),
    }
  )
  monkeypatch.setattr(accessors, "residual_action", lambda env: action)
  observed = motion_mdp.motion_observation(cast(Any, NS()))
  torch.testing.assert_close(observed, torch.tensor([[1, 0.5, 0, 0.5, 0.6, 0.5]] * 5))


def test_braking_reward_charges_applied_scale_only_during_bilateral_hold(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  action = motion_action()
  action._datastore_scalar_outputs[MOTION_SCALE_GETTER] = torch.tensor(
    [1.0, 0.5, 0.0, 0.0, 0.5]
  )
  action._datastore_scalar_outputs[accessors.RIGHT_PHASE][3] = 3.0
  action._datastore_scalar_outputs[accessors.LEFT_PHASE][4] = 3.0
  monkeypatch.setattr(
    accessors, "residual_action", lambda env, action_name=accessors.ACTION_NAME: action
  )
  cfg = load_env_cfg(TASK_IDS["HRP5P"])
  env = cast(Any, NS(num_envs=5, device="cpu"))
  manager = RewardManager({"braking": cfg.rewards["braking"]}, env)
  torch.testing.assert_close(
    manager.compute(0.02), torch.tensor([0.0, -0.0003, -0.0006, 0.0, 0.0])
  )
  assert action.motion_scale_dispatched.tolist() == [1.0] * 5
  assert "braking" not in load_env_cfg(RESIDUAL_TASK_IDS["HRP5P"]).rewards


def test_native_braking_events_accumulate_once_and_reset_per_row(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  action = motion_action()
  count = action._datastore_scalar_outputs[MOTION_COUNT_GETTER]
  count.fill_(1.0)
  action._advance_action_extensions()
  action._advance_action_extensions()
  torch.testing.assert_close(action.motion_braking_events, torch.ones(5))
  action._datastore_output_fresh[1] = False
  count.fill_(2.0)
  action._advance_action_extensions()
  torch.testing.assert_close(
    action.motion_braking_events, torch.tensor([2, 1, 2, 2, 2]).float()
  )
  action._reset_action_extensions(torch.tensor([0, 3]))
  torch.testing.assert_close(
    action.motion_braking_events, torch.tensor([0, 1, 2, 0, 2]).float()
  )
  torch.testing.assert_close(
    action._motion_count_previous, torch.tensor([0, 1, 2, 0, 2]).float()
  )
  monkeypatch.setattr(accessors, "residual_action", lambda env: action)
  reward = load_env_cfg(TASK_IDS["HRP5P"]).rewards["braking_start"]
  env = cast(Any, NS(num_envs=5, device="cpu", step_dt=0.02))
  manager = RewardManager({"braking_start": reward}, env)
  expected = torch.tensor([0.0, -0.22, -0.44, 0.0, -0.44])
  torch.testing.assert_close(manager.compute(env.step_dt), expected)
  env.step_dt = 0.01
  torch.testing.assert_close(manager.compute(env.step_dt), expected)


def test_holding_cost_excludes_slow_travel_and_inactive_hand_phases(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  action = motion_action()
  action._datastore_scalar_outputs[MOTION_STATE_GETTER] = torch.tensor(
    [0, 1, 2, 2, 0]
  ).float()
  action._datastore_scalar_outputs[accessors.LEFT_PHASE][3] = 3.0
  monkeypatch.setattr(
    accessors, "residual_action", lambda env, action_name=accessors.ACTION_NAME: action
  )
  env = cast(Any, NS(num_envs=5, device="cpu"))
  reward = load_env_cfg(TASK_IDS["HRP5P"]).rewards["holding"]
  manager = RewardManager({"holding": reward}, env)
  torch.testing.assert_close(
    manager.compute(0.02), torch.tensor([0, 0, -0.002, 0, 0]).float()
  )


def test_motion_latches_per_row_while_joints_keep_updating(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  action = motion_action()
  monkeypatch.setattr(
    McRtcResidualJointPositionAction,
    "process_actions",
    lambda self, requests: self._raw_actions.copy_(requests),
  )
  action.process_actions(torch.tensor([[0.1, 0.2, 0.5]] * 5))
  assert action.motion_decisions.tolist() == [1] * 5
  assert action.motion_decision_state[:, 0].tolist() == [0] * 5
  assert action.motion_decision_state[:, 1].tolist() == pytest.approx([0.99] * 5)
  for _ in range(99):
    action.process_actions(torch.tensor([[0.3, 0.4, 0.0]] * 5))
  assert action.motion_scale_requested.tolist() == [0.5] * 5
  assert action._raw_actions.tolist() == pytest.approx(torch.tensor([[0.3, 0.4]] * 5))
  assert action.motion_decision_state[:, 0].tolist() == [1] * 5
  action._reset_action_extensions(torch.tensor([1]))
  assert action.motion_decision_state[:, 2].tolist() == [0.5, 0, 0.5, 0.5, 0.5]
  action.process_actions(torch.zeros(5, 3))
  assert action.motion_scale_requested.tolist() == [1] * 5
  assert action.motion_decisions.tolist() == [2, 1, 2, 2, 2]


def test_phase_exit_clears_latch_and_reentry_selects_immediately(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  action = motion_action()
  monkeypatch.setattr(
    McRtcResidualJointPositionAction, "process_actions", lambda self, requests: None
  )
  action._datastore_scalar_outputs[accessors.LEFT_PHASE][0] = 0
  action._datastore_output_fresh[1] = False
  action.controller_failed[2] = True
  action.controller_worker_failed[3] = True
  assert action.motion_decision_state[:, 0].tolist() == [0, 0, 0, 0, 1]
  action.process_actions(torch.ones(5, 3))
  assert action.motion_scale_requested.tolist() == [1, 1, 1, 1, 0]
  action._datastore_scalar_outputs[accessors.LEFT_PHASE][4] = 0
  action.process_actions(torch.ones(5, 3))
  assert action.motion_scale_requested.tolist() == [1] * 5
  action._datastore_scalar_outputs[accessors.LEFT_PHASE][4] = accessors.HOLD_PHASE
  assert action.motion_decision_state[:, 0].tolist() == [0, 0, 0, 0, 1]
  action.process_actions(torch.tensor([[0, 0, 0.5]] * 5))
  assert action.motion_scale_requested.tolist() == [1, 1, 1, 1, 0.5]


def test_decision_state_is_uncorrupted_and_used_by_actor_and_critic() -> None:
  cfg = load_env_cfg(TASK_IDS["HRP5P"])
  group = cfg.observations["motion_decision"]
  assert not group.enable_corruption
  agent = cast("RslRlOnPolicyRunnerCfg", load_rl_cfg(TASK_IDS["HRP5P"]))
  assert agent.obs_groups == {
    "actor": ("actor", "motion_decision"),
    "critic": ("critic", "motion_decision"),
  }
  assert agent.actor.distribution_cfg is not None
  assert agent.actor.distribution_cfg["init_std"] == 0.1
  assert agent.actor.distribution_cfg["motion_init_std"] == 0.8


def test_partial_reset_preserves_other_rows_decision_clocks() -> None:
  action = motion_action()
  action._motion_steps_remaining.copy_(torch.tensor([20, 40, 60, 80, 99]))
  action._motion_decisions.fill_(3)
  action._reset_action_extensions(torch.tensor([1, 3]))
  assert action._motion_steps_remaining.tolist() == [20, 0, 60, 0, 99]
  assert action.motion_decisions.tolist() == [3, 0, 3, 0, 3]
  assert action.motion_decision_state[:, 0].tolist() == [0, 1, 0, 1, 0]


def test_decision_interval_respects_policy_timestep() -> None:
  action = motion_action()
  action._env = cast(Any, NS(num_envs=5, device="cpu", step_dt=0.01))
  action._setup_action_extensions(action.cfg)
  assert action._motion_decision_steps == 200
  action.cfg.motion_decision_period_s = 0.015
  with pytest.raises(ValueError, match="integer number"):
    action._setup_action_extensions(action.cfg)


def test_retuned_rewards_leave_raw_dcm_metric_and_other_task_weights_independent() -> (
  None
):
  cfg = load_env_cfg(TASK_IDS["HRP5P"])
  original = load_env_cfg(RESIDUAL_TASK_IDS["HRP5P"])
  assert cfg.rewards["contact_dcm_stability"].weight == 0.25
  assert cfg.rewards["termination_penalty"].weight == -400
  assert cfg.rewards["holding"].weight == -0.1
  assert original.rewards["contact_dcm_stability"].weight == 1
  assert original.rewards["termination_penalty"].weight == -200
  for name in ("residual_magnitude", "residual_rate"):
    assert cfg.rewards[name].weight == REQUEST_WEIGHT
    assert original.rewards[name].weight == POSITION_REQUEST_WEIGHT
  assert (
    cfg.metrics["contact_dcm_stability"] == original.metrics["contact_dcm_stability"]
  )
  assert cfg.terminations["controller_worker_failed"].time_out
  assert (
    cfg.rewards["termination_penalty"].func
    is original.rewards["termination_penalty"].func
  )
