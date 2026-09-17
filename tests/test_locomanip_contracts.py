"""Wiring contracts the trainable locomanip task must not lose."""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

import pytest
from mjlab.envs import ManagerBasedRlEnvCfg
from probe_hand_wrench_adaptation import ARMS

import mc_mjlab.tasks  # noqa: F401
from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionActionCfg,
)
from mc_mjlab.bridge.controller_datastore import CONTROL_COM, PLANNED_ZMP
from mc_mjlab.residuals.authority import TORQUE_FRACTION
from mc_mjlab.tasks.locomanip import DEMO_TASK_ID, RESIDUAL_TASK_ID
from mc_mjlab.tasks.locomanip.cart import cart_nominal_mass_kg
from mc_mjlab.tasks.locomanip.evaluation import LOCOMANIP_EVALUATION
from mc_mjlab.tasks.locomanip.locomanip_residual_env_cfg import (
  CART_MASS_RANGE_KG,
  DECIMATION,
  FORCE_SENSORS,
  FRAMESKIP,
  OBJECT_HISTORY,
  SUCCESS_POSITION_TOLERANCE_M,
  SUCCESS_YAW_TOLERANCE_RAD,
  ZMP_TRACKING_STD,
  make_locomanip_residual_env_cfg,
)
from mc_mjlab.tasks.locomanip.mdp import accessors
from mc_mjlab.tasks.locomanip.mdp.events import mass_alpha_range


def _scales(cfg: ManagerBasedRlEnvCfg) -> dict[str, float]:
  """Return the action's per-actuator scales, which are always a dict here."""
  scale = _action(cfg).scale
  assert isinstance(scale, dict)
  return scale


def _action(cfg: ManagerBasedRlEnvCfg) -> McRtcResidualJointPositionActionCfg:
  """Return the residual action cfg under its own type."""
  return cast(McRtcResidualJointPositionActionCfg, cfg.actions[accessors.ACTION_NAME])


def test_the_patch_exposes_HandWrenchAdaptation() -> None:
  """Verify the probe's arms are knobs the installed controller patch provides."""
  patch = Path(__file__).resolve().parents[1] / "patches/locomanip-unattended.patch"
  text = patch.read_text()
  assert "HandWrenchAdaptation" in text
  for arm in ARMS.values():
    for key in arm:
      assert key in text, key
  # The projection is the part that matters on HRP5P. docs/locomanip.md
  assert ARMS["push-axis"]["forceProjection"] == [1.0, 0.0, 0.0]


def test_ids_are_distinct() -> None:
  """Verify the demo id is a prefix of the trainable one, so matching must be exact."""
  assert RESIDUAL_TASK_ID != DEMO_TASK_ID
  assert RESIDUAL_TASK_ID.startswith(DEMO_TASK_ID)


def test_action_name_matches_the_provenance_key() -> None:
  """Verify the runner's hardcoded action lookup resolves before the first step."""
  cfg = make_locomanip_residual_env_cfg()
  assert accessors.ACTION_NAME in cfg.actions


def test_control_rates_divide() -> None:
  """Verify decimation is a whole number of controller periods."""
  cfg = make_locomanip_residual_env_cfg()
  action = _action(cfg)
  assert cfg.decimation == DECIMATION
  assert action.frameskip == FRAMESKIP
  assert cfg.decimation % action.frameskip == 0


def test_residual_authority_covers_every_actuator() -> None:
  """Verify no actuator can fall through to scale 1.0 with no clip."""
  cfg = make_locomanip_residual_env_cfg()
  action, scales = _action(cfg), _scales(cfg)
  assert action.clip == {k: (-v, v) for k, v in scales.items()}
  assert action.residual_actuator_names
  patterns = {pattern.replace("\\", "") for pattern in scales}
  assert set(action.residual_actuator_names) <= patterns


def test_residual_authority_is_per_joint_hardware_authority() -> None:
  """Verify each joint gets its own share of torque capacity, not one number."""
  cfg = make_locomanip_residual_env_cfg()
  full = _scales(cfg)
  by_joint = {pattern.replace("\\", ""): value for pattern, value in full.items()}
  residual = [by_joint[joint] for joint in _action(cfg).residual_actuator_names or ()]
  assert len(set(residual)) > 1
  halved = _scales(make_locomanip_residual_env_cfg(torque_fraction=TORQUE_FRACTION / 2))
  for pattern, value in halved.items():
    assert value == pytest.approx(full[pattern] / 2)


def test_declared_datastore_outputs_cover_the_terms() -> None:
  """Verify every callback the manager terms read is collected by the action."""
  action = _action(make_locomanip_residual_env_cfg())
  assert set(action.datastore_vectors_outputs) == {
    accessors.OBJECT_REFERENCE_POSITION,
    accessors.OBJECT_REFERENCE_RPY,
    PLANNED_ZMP,
    CONTROL_COM,
  }
  assert set(action.datastore_scalar_outputs) == {
    accessors.LEFT_PHASE,
    accessors.RIGHT_PHASE,
    accessors.COMPLETE,
  }
  assert action.controller_objects == {"obj": accessors.OBJECT_ENTITY}
  assert action.required_controller == accessors.REQUIRED_CONTROLLER


def test_the_zmp_reward_has_the_callbacks_it_compares() -> None:
  """Verify the planned ZMP and its CoM reference are collected for the reward."""
  cfg = make_locomanip_residual_env_cfg()
  outputs = set(_action(cfg).datastore_vectors_outputs)
  assert {PLANNED_ZMP, CONTROL_COM} <= outputs
  assert cfg.rewards["zmp_tracking"].params["std"] == ZMP_TRACKING_STD
  assert set(cfg.metrics) >= {"zmp_error", "zmp_grounded"}


def test_the_critic_sees_more_than_the_actor() -> None:
  """Verify the privileged terms are critic-only and the force sensors are shared."""
  observations = make_locomanip_residual_env_cfg().observations
  actor = set(observations["actor"].terms)
  critic = set(observations["critic"].terms)
  assert actor < critic
  assert critic - actor == {"object_mass", "hand_contact_force"}
  assert set(FORCE_SENSORS) <= actor


def test_the_payload_evidence_carries_history() -> None:
  """Verify one frame is not all the policy gets of the cart it must infer."""
  actor = make_locomanip_residual_env_cfg().observations["actor"].terms
  carried = {"object_velocity", "object_position_error", "object_yaw_error"}
  carried |= set(FORCE_SENSORS)
  for name in carried:
    assert actor[name].history_length == OBJECT_HISTORY, name
  assert actor["manipulation_phase"].history_length == 0


def test_only_the_actor_pays_for_noise() -> None:
  """Verify the policy trains on corrupted observations and the critic does not."""
  observations = make_locomanip_residual_env_cfg().observations
  actor, critic = observations["actor"], observations["critic"]
  assert actor.enable_corruption
  assert not critic.enable_corruption
  noisy = {name for name, term in actor.terms.items() if term.noise is not None}
  assert {"base_lin_vel", "joint_pos", "object_pose", "object_yaw_error"} <= noisy
  assert all(term.noise is None for term in critic.terms.values())


def test_the_encoder_bias_the_actor_suffers_is_applied() -> None:
  """Verify the biased joint reading has the startup event that gives it a bias."""
  cfg = make_locomanip_residual_env_cfg()
  assert cfg.events["encoder_bias"].mode == "startup"
  assert cfg.observations["actor"].terms["joint_pos"].params == {"biased": True}
  assert cfg.observations["critic"].terms["joint_pos"].params == {}


def test_worker_failure_truncates_instead_of_penalising() -> None:
  """Verify infrastructure noise bootstraps rather than paying the fall penalty."""
  terminations = make_locomanip_residual_env_cfg().terminations
  assert terminations["controller_worker_failed"].time_out
  assert not terminations["controller_failed"].time_out


def test_scene_resets_joints_first() -> None:
  """Verify the only term that resets joints leads the reset events."""
  events = make_locomanip_residual_env_cfg().events
  assert next(iter(events)) == "reset_scene_to_default"


def test_cart_pose_hook_is_opt_in() -> None:
  """Verify the pose hook adds its event only when asked for."""
  assert "reset_cart" not in make_locomanip_residual_env_cfg().events
  varied = make_locomanip_residual_env_cfg(cart_pose_range={"x": (-0.05, 0.05)})
  assert "reset_cart" in varied.events


def test_payload_is_drawn_every_episode() -> None:
  """Verify the swept mass range is on by default and resampled at reset."""
  payload = make_locomanip_residual_env_cfg().events["cart_payload"]
  assert payload.mode == "reset"
  assert payload.params["alpha_range"] == mass_alpha_range(
    CART_MASS_RANGE_KG, cart_nominal_mass_kg()
  )
  assert (
    "cart_payload"
    not in make_locomanip_residual_env_cfg(cart_mass_range_kg=None).events
  )


def test_success_is_physical_not_the_controller_schedule() -> None:
  """Verify the verdict is the cart's placement, not the FSM reaching its hold."""
  cfg = make_locomanip_residual_env_cfg()
  success = cfg.metrics["task_success"]
  assert success.reduce == "last"
  assert success.params["position_tolerance_m"] == SUCCESS_POSITION_TOLERANCE_M
  assert success.params["yaw_tolerance_rad"] == SUCCESS_YAW_TOLERANCE_RAD
  assert LOCOMANIP_EVALUATION.success == "task_success"
  # Kept beside it: the gap between the two is how often the schedule outran it.
  assert "task_complete" in cfg.metrics
  assert {"task_complete", "task_success"} <= set(LOCOMANIP_EVALUATION.metrics)


def test_the_payload_metric_names_the_randomized_body() -> None:
  """Verify the drawn mass is reported, from the body the event actually varies."""
  cfg = make_locomanip_residual_env_cfg(cart_mass_range_kg=CART_MASS_RANGE_KG)
  asset_cfg = cfg.events["cart_payload"].params["asset_cfg"]
  assert asset_cfg.body_names == (accessors.OBJECT_BODY,)
  assert cfg.metrics["cart_mass"].reduce == "last"


def test_mass_alpha_range_is_the_log_scale_pseudo_inertia_wants() -> None:
  """Verify the conversion inverts `mass = nominal * exp(2 * alpha)`."""
  low, high = mass_alpha_range((10.0, 40.0), 10.0)
  assert low == 0.0
  assert math.isclose(10.0 * math.exp(2.0 * high), 40.0)
  with pytest.raises(ValueError):
    mass_alpha_range((0.0, 1.0), 10.0)


def test_the_drawn_payload_is_absolute_kilograms() -> None:
  """Verify the draw lands on the swept range whatever the asset's own mass is."""
  nominal = cart_nominal_mass_kg()
  low, high = (
    make_locomanip_residual_env_cfg().events["cart_payload"].params["alpha_range"]
  )
  assert math.isclose(nominal * math.exp(2.0 * low), CART_MASS_RANGE_KG[0])
  assert math.isclose(nominal * math.exp(2.0 * high), CART_MASS_RANGE_KG[1])


def test_sparse_jacobian_is_kept() -> None:
  """Verify the cart's extra velocities stay under MuJoCo Warp's dense limit."""
  assert make_locomanip_residual_env_cfg().sim.mujoco.jacobian == "sparse"
