"""Force-feedback routing, lifecycle, policy accounting and task contracts."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace as NS
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from conftest import requires_controller
from evaluation.locomanip_authority import (
  CHANNELS,
  AuthorityTrace,
  action_matrix,
  conditions,
)
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import mc_rtc_interface as native_typed
from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionAction,
)
from mc_mjlab.actions.residual_feedback_action import (
  ResidualFeedbackJointPositionAction,
  ResidualFeedbackJointPositionActionCfg,
)
from mc_mjlab.bridge.sim_controller_bridge import SimControllerBridge
from mc_mjlab.mdp.rewards import requested_action_l2, requested_action_rate_l2
from mc_mjlab.rl.effective_training_manifest import (
  canonicalize,
  validate_effective_training_manifest,
)
from mc_mjlab.tasks.evaluation import evaluation_for
from mc_mjlab.tasks.locomanip import FEEDBACK_TASK_IDS, RESIDUAL_TASK_IDS
from mc_mjlab.tasks.locomanip.evaluation import LOCOMANIP_EVALUATION
from mc_mjlab.tasks.locomanip.locomanip_feedback_env_cfg import (
  HAND_FORCE_SENSORS,
  make_locomanip_feedback_env_cfg,
)
from mc_mjlab.tasks.locomanip.locomanip_residual_env_cfg import (
  make_locomanip_residual_env_cfg,
)
from mc_mjlab.tasks.locomanip.mdp import accessors, feedback
from mc_mjlab.tasks.locomanip.profiles import HRP5P, PROFILES

native = cast(Any, native_typed)


def config(**kwargs: Any) -> ResidualFeedbackJointPositionActionCfg:
  """Build an action config without loading a controller or robot asset."""
  return ResidualFeedbackJointPositionActionCfg(
    entity_name="robot",
    actuator_names=(".*",),
    mc_rtc_config_path="unused",
    **{
      "feedback_modalities": ("wrench",),
      "wrench_sensor_names": HAND_FORCE_SENSORS,
      "wrench_force_only": True,
      "gate_scalar_outputs": (accessors.LEFT_PHASE, accessors.RIGHT_PHASE),
      "gate_value": accessors.HOLD_PHASE,
      **kwargs,
    },
  )


@pytest.fixture
def action() -> ResidualFeedbackJointPositionAction:
  """Allocate real action buffers with a reordered native sensor layout."""
  term = object.__new__(ResidualFeedbackJointPositionAction)
  term.cfg = config()
  term._env = NS(num_envs=3, device="cpu")
  term._last_gate = torch.ones(3)
  term._previous_gate = torch.ones(3)
  term._actor_update_gate = torch.ones(3)
  term._num_targets = 0
  term._residual_ids = torch.zeros(0, dtype=torch.long)
  term._action_dim = 0
  term._raw_actions = torch.zeros(3, 0)
  term._residual_raw_actions = term._raw_actions
  term._previous_residual_raw_actions = torch.zeros(3, 0)
  term._processed_actions = torch.zeros(3, 0)
  term._executed_physical = torch.zeros(3, 0)
  term._previous_executed_physical = torch.zeros(3, 0)
  term._scale, term._offset, term._residual_action_dim = 1.0, 0.0, 0
  term._recovery_authority = None
  term._datastore_output_fresh = torch.ones(3, dtype=torch.bool)
  term.controller_failed = torch.zeros(3, dtype=torch.bool)
  term.controller_worker_failed = torch.zeros(3, dtype=torch.bool)
  term._datastore_scalar_outputs = {
    name: torch.full((3,), accessors.HOLD_PHASE)
    for name in (accessors.LEFT_PHASE, accessors.RIGHT_PHASE)
  }
  layout = native.IoLayout()
  layout.input.force_sensors = [
    "RightFootForceSensor",
    "RightHandForceSensor",
    "LeftFootForceSensor",
    "LeftHandForceSensor",
  ]
  bridge = object.__new__(SimControllerBridge)
  bridge.layout = layout
  bridge._clear_state_feedback_offsets()
  term._bridge = bridge
  term._env.sim = NS(mj_model=NS(sensor=lambda _name: NS(dim=[3])))
  term._env.action_manager = NS(get_term=lambda _name: term)
  with patch.object(McRtcResidualJointPositionAction, "_build_bridge"):
    term._build_bridge(term.cfg)
  term._setup_residual_printer(term.cfg)
  return term


@pytest.mark.parametrize("scale", [0, -1, float("inf"), float("nan")])
def test_invalid_scale(scale: float) -> None:
  with pytest.raises(ValueError, match="finite and positive"):
    config(wrench_force_scale=scale)


def test_rejects_unused_or_ambiguous_channels() -> None:
  with pytest.raises(ValueError, match="empty joint residual"):
    config(residual_actuator_names=(".*",))
  with pytest.raises(ValueError, match="distinct"):
    config(wrench_sensor_names=("hand", "hand"))
  with pytest.raises(ValueError, match="unsupported feedback modalities"):
    config(feedback_modalities=("hand_force",))
  with pytest.raises(ValueError, match="recovery authority"):
    config(recovery_detector_path="detector.json")


def test_named_routes_and_sensor_isolation(
  action: ResidualFeedbackJointPositionAction,
) -> None:
  requests = torch.tensor([[0.1, -0.2, 0.3, -0.4, 0.5, 0.6]]).expand(3, 6)
  action.process_actions(requests)
  action._advance_action_extensions()
  assert action.action_dim == 6
  assert action._processed_actions.shape == (3, 0)
  torch.testing.assert_close(action.raw_action, requests)
  torch.testing.assert_close(action._wrench_offset[:, 18:21], requests[:, :3] * 50)
  torch.testing.assert_close(action._wrench_offset[:, 6:9], requests[:, 3:] * 50)

  bridge = action._bridge
  measured = torch.arange(3 * bridge.layout.input_size).reshape(3, -1).double()
  original = measured.clone()
  dispatched = measured.clone()
  bridge._apply_state_feedback_offsets(dispatched)
  off = bridge.layout.input.force_sensors_offset()
  changed = [off + column for column in (18, 19, 20, 6, 7, 8)]
  torch.testing.assert_close(
    dispatched[:, changed], original[:, changed] + requests * 50
  )
  unchanged = [i for i in range(measured.shape[1]) if i not in changed]
  torch.testing.assert_close(dispatched[:, unchanged], original[:, unchanged])
  torch.testing.assert_close(measured, original)

  action.process_actions(torch.zeros_like(requests))
  action._advance_action_extensions()
  baseline = original.clone()
  bridge._apply_state_feedback_offsets(baseline)
  assert torch.equal(baseline, original)


def test_gate_closes_on_phase_staleness_and_failures(
  action: ResidualFeedbackJointPositionAction,
) -> None:
  action.process_actions(torch.ones(3, 6))
  action._datastore_scalar_outputs[accessors.LEFT_PHASE][0] = 0
  action._datastore_output_fresh[1] = False
  action.controller_worker_failed[2] = True
  action._advance_action_extensions()
  assert not action.executed_physical_action.any()
  assert not action._wrench_offset.any()

  action._datastore_scalar_outputs[accessors.LEFT_PHASE][:] = accessors.HOLD_PHASE
  action._datastore_output_fresh[:] = True
  action.controller_worker_failed[:] = False
  action.controller_failed[1] = True
  action._advance_action_extensions()
  torch.testing.assert_close(action.last_gate, torch.tensor([1.0, 0, 1]))
  action._datastore_scalar_outputs[accessors.RIGHT_PHASE][2] = 5
  action._advance_action_extensions()
  torch.testing.assert_close(action.last_gate, torch.tensor([1.0, 0, 0]))


def test_request_costs_history_and_partial_reset(
  action: ResidualFeedbackJointPositionAction,
) -> None:
  action.process_actions(torch.full((3, 6), 2.0))
  action._advance_action_extensions()
  torch.testing.assert_close(feedback.executed_feedback(action._env), torch.ones(3, 6))
  torch.testing.assert_close(
    feedback.feedback_force_rms(action._env), torch.full((3,), 50.0)
  )
  torch.testing.assert_close(requested_action_l2(action._env), torch.full((3,), 6.0))
  assert not requested_action_rate_l2(action._env).any()

  action.process_actions(torch.full((3, 6), -0.5))
  action._advance_action_extensions()
  torch.testing.assert_close(
    action.previous_requested_normalized_action, torch.ones(3, 6)
  )
  torch.testing.assert_close(
    action.previous_executed_normalized_action, torch.ones(3, 6)
  )
  torch.testing.assert_close(
    requested_action_rate_l2(action._env), torch.full((3,), 13.5)
  )
  action._reset_action_extensions(torch.tensor([1]))
  assert not action._wrench_offset[1].any()
  assert not action.raw_action[1].any()
  assert not action.previous_requested_normalized_action[1].any()
  assert not action.previous_executed_normalized_action[1].any()
  assert action.actor_update_gate[1] == 1
  torch.testing.assert_close(
    action.executed_physical_action[0], torch.full((6,), -25.0)
  )
  torch.testing.assert_close(
    action.executed_physical_action[2], torch.full((6,), -25.0)
  )


def test_missing_sensors_fail_before_workers(
  action: ResidualFeedbackJointPositionAction,
) -> None:
  action._bridge.layout.input.force_sensors = ["LeftHandForceSensor"]
  with patch.object(McRtcResidualJointPositionAction, "_build_bridge"):
    with pytest.raises(ValueError, match="RightHandForceSensor"):
      action._build_bridge(action.cfg)


@requires_controller(HRP5P.mc_rtc_yaml)
def test_task_registration_and_environment_parity() -> None:
  assert set(FEEDBACK_TASK_IDS) == set(RESIDUAL_TASK_IDS) == {"HRP5P", "JVRC1"}
  assert not set(FEEDBACK_TASK_IDS.values()) & set(RESIDUAL_TASK_IDS.values())
  for profile in PROFILES:
    old = make_locomanip_residual_env_cfg(profile=profile)
    new = make_locomanip_feedback_env_cfg(profile=profile)
    for name in ("scene", "sim", "rewards", "terminations", "events"):
      assert canonicalize(getattr(new, name)) == canonicalize(getattr(old, name))
    for group in old.observations:
      for name, term in old.observations[group].terms.items():
        if name != "actions":
          assert canonicalize(new.observations[group].terms[name]) == canonicalize(term)
    assert new.episode_length_s == old.episode_length_s
  for task in FEEDBACK_TASK_IDS.values():
    assert evaluation_for(task) is LOCOMANIP_EVALUATION
    assert load_rl_cfg(task).experiment_name == task
    play = load_env_cfg(task, play=True)
    assert play.scene.num_envs == 1
    action_cfg = cast(
      ResidualFeedbackJointPositionActionCfg, play.actions[accessors.ACTION_NAME]
    )
    assert action_cfg.num_workers is None
    assert not play.observations["actor"].enable_corruption


def test_feedback_scale_and_sensor_order_are_checkpoint_interfaces() -> None:
  original = config()
  saved = {"policy_interface": {"actions": canonicalize(original)}}
  validate_effective_training_manifest(saved, saved, full_resume=False)
  for changed in (
    replace(original, wrench_force_scale=25.0),
    replace(
      original, wrench_sensor_names=tuple(reversed(original.wrench_sensor_names))
    ),
  ):
    active = {"policy_interface": {"actions": canonicalize(changed)}}
    with pytest.raises(RuntimeError, match="interface"):
      validate_effective_training_manifest(saved, active, full_resume=False)


def test_probe_exercises_each_force_independently() -> None:
  rows = conditions("feedback", [0.1, 0.5, 1.0], "all")
  env = NS(num_envs=len(rows), device="cpu", action_manager=NS(total_action_dim=6))
  actions = action_matrix(cast(Any, env), rows)
  assert actions.shape == (39, 6)
  assert not actions[:3].any()
  assert torch.all((actions[3:] != 0).sum(dim=1) == 1)
  for index, channel in enumerate(CHANNELS):
    selected = [i for i, row in enumerate(rows) if row["channel"] == channel]
    assert set(actions[selected, index].tolist()) == set(
      torch.tensor([-1, -0.5, -0.1, 0.1, 0.5, 1]).tolist()
    )


def test_probe_retains_terminal_measurement_before_reset(
  action: ResidualFeedbackJointPositionAction,
) -> None:
  env = cast(Any, action._env)
  env.step_dt = 0.02
  env.scene = {
    "robot": NS(data=NS(root_link_pos_w=torch.tensor([[0.0, 0, 0.8]] * 3))),
    "cart": NS(data=NS(root_link_pos_w=torch.tensor([[0.9, 0, 0.0]] * 3))),
  }
  trace = AuthorityTrace(env)
  env.scene["cart"].data.root_link_pos_w[1, 0] = 1.2

  def snapshot(index: int, terminated: bool, truncated: bool) -> dict:
    return {
      "x": float(env.scene["cart"].data.root_link_pos_w[index, 0]),
      "terminated": terminated,
      "truncated": truncated,
    }

  def reset(_env: Any, ids: torch.Tensor) -> None:
    env.scene["cart"].data.root_link_pos_w[ids, 0] = 0.9

  with (
    patch.object(trace, "_snapshot", snapshot),
    patch.object(action, "controller_reference", return_value=torch.zeros(3, 2)),
    patch("evaluation.locomanip_authority.reset_done", reset),
  ):
    trace.update(torch.tensor([False, True, False]), torch.zeros(3, dtype=torch.bool))
    trace.update(torch.zeros(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool))

  result = trace.final[1]
  assert result is not None
  assert result["x"] == pytest.approx(1.2)
  assert result["terminated"]
  assert trace.held[1] == pytest.approx(0.02)
  assert trace.held[0] == pytest.approx(0.04)
