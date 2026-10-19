"""Wiring contracts the trainable locomanip task must not lose."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from conftest import requires_controller
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

import mc_mjlab.tasks  # noqa: F401
from mc_mjlab import mdp
from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionActionCfg,
)
from mc_mjlab.bridge.controller_datastore import (
  CONTROL_COM,
  CONTROL_COM_VEL,
  PLANNED_ZMP,
)
from mc_mjlab.residuals.authority import TORQUE_FRACTION
from mc_mjlab.tasks.locomanip import DEMO_TASK_IDS, RESIDUAL_TASK_IDS
from mc_mjlab.tasks.locomanip.cart import cart_nominal_mass_kg
from mc_mjlab.tasks.locomanip.evaluation import LOCOMANIP_EVALUATION
from mc_mjlab.tasks.locomanip.locomanip_env_cfg import locomanip_env_cfg
from mc_mjlab.tasks.locomanip.locomanip_residual_env_cfg import (
  CART_MASS_RANGE_KG,
  DECIMATION,
  FORCE_HISTORY,
  FORCE_SENSORS,
  FRAMESKIP,
  HAND_FORCE_SENSORS,
  SAFETY_HISTORY,
  SOLE_VELOCIMETERS,
  SUCCESS_POSITION_TOLERANCE_M,
  SUCCESS_YAW_TOLERANCE_RAD,
  make_locomanip_residual_env_cfg,
)
from mc_mjlab.tasks.locomanip.mdp import accessors, metrics, rewards
from mc_mjlab.tasks.locomanip.mdp import observations as locomanip_observations
from mc_mjlab.tasks.locomanip.mdp.events import mass_alpha_range
from mc_mjlab.tasks.locomanip.profiles import HRP5P, PROFILES

pytestmark = requires_controller(HRP5P.mc_rtc_yaml)


def _scales(cfg: ManagerBasedRlEnvCfg) -> dict[str, float]:
  """Return the action's per-actuator scales, which are always a dict here."""
  scale = _action(cfg).scale
  assert isinstance(scale, dict)
  return scale


def _action(cfg: ManagerBasedRlEnvCfg) -> McRtcResidualJointPositionActionCfg:
  """Return the residual action cfg under its own type."""
  return cast(McRtcResidualJointPositionActionCfg, cfg.actions[accessors.ACTION_NAME])


def test_ids_are_distinct() -> None:
  """Verify the demo id is a prefix of the trainable one, so matching must be exact."""
  assert set(DEMO_TASK_IDS) == set(RESIDUAL_TASK_IDS) == {"HRP5P", "JVRC1"}
  for robot, demo_id in DEMO_TASK_IDS.items():
    assert RESIDUAL_TASK_IDS[robot] != demo_id
    assert RESIDUAL_TASK_IDS[robot].startswith(demo_id)
  # Every id distinct, so a resume can cross neither control mode nor robot.
  assert len(set(DEMO_TASK_IDS.values()) | set(RESIDUAL_TASK_IDS.values())) == 4


def test_every_profile_builds() -> None:
  """Verify each robot's scene resolves, since only registration exercises them."""
  for profile in PROFILES:
    cfg = make_locomanip_residual_env_cfg(profile=profile)
    assert cfg.scene.entities["cart"].init_state.pos[0] == profile.cart_init_x
    assert {s.name for s in cfg.scene.sensors} == {"left_hand_cart", "right_hand_cart"}


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
    CONTROL_COM_VEL,
  }
  assert set(action.datastore_scalar_outputs) == {
    accessors.LEFT_PHASE,
    accessors.RIGHT_PHASE,
    accessors.COMPLETE,
  }
  assert action.controller_objects == {"obj": accessors.OBJECT_ENTITY}
  assert action.required_controller == accessors.REQUIRED_CONTROLLER


def test_balance_reward_uses_every_contact_and_keeps_zmp_diagnostic() -> None:
  """Verify the balance reward sees hand loads and the ZMP metric keeps its plan."""
  cfg = make_locomanip_residual_env_cfg()
  outputs = set(_action(cfg).datastore_vectors_outputs)
  assert {PLANNED_ZMP, CONTROL_COM, CONTROL_COM_VEL} <= outputs
  assert "zmp_tracking" not in cfg.rewards
  assert set(cfg.metrics) >= {"zmp_error", "zmp_grounded"}

  for params in (
    cfg.rewards["contact_dcm_stability"].params,
    cfg.metrics["zmp_error"].params,
  ):
    assert set(HAND_FORCE_SENSORS) <= set(params["contact_sensor_names"])
    assert set(mdp.sensors.GROUND_CONTACT_SENSORS) <= set(
      params["contact_sensor_names"]
    )
    assert params["sensor_names"] == mdp.sensors.GROUND_CONTACT_SENSORS
  assert "contact_sensor_names" not in cfg.metrics["zmp_grounded"].params


def test_the_critic_sees_more_than_the_actor() -> None:
  """Verify every actor term is shared and no simulator-only truth reaches it."""
  cfg = make_locomanip_residual_env_cfg()
  actor = cfg.observations["actor"].terms
  critic = cfg.observations["critic"].terms
  assert set(actor) < set(critic)
  privileged = {
    locomanip_observations.object_mass,
    locomanip_observations.hand_contact_force,
    locomanip_observations.com_state,
    mdp.sensors.measured_zmp_offset,
  }
  assert not privileged & {term.func for term in actor.values()}
  assert not any(name.startswith("object_") for name in actor)
  assert "object_tracking" not in cfg.rewards


def test_observed_evidence_carries_its_history() -> None:
  """Verify each optional precursor or payload sensor the actor sees keeps history."""
  terms = make_locomanip_residual_env_cfg().observations["actor"].terms
  precursors = {
    "base_ang_vel",
    "root_vertical_velocity",
    "actions",
    "controller_pos_error",
    "controller_planned_com_vel",
    "foot_load_share",
    "foot_normal_forces",
    *SOLE_VELOCIMETERS,
  }
  # Which of these the actor observes is a tuning choice; their history is not.
  for name in precursors & set(terms):
    assert terms[name].history_length == SAFETY_HISTORY, name
  for name in set(FORCE_SENSORS) & set(terms):
    assert terms[name].history_length == FORCE_HISTORY, name
  assert terms["manipulation_phase"].history_length == 0


def test_fall_labels_partition_low_and_tilted_states(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A low tilted robot is a topple, and only a low upright robot collapses."""
  low = torch.tensor([False, True, False, True])
  tilted = torch.tensor([False, False, True, True])
  monkeypatch.setattr(
    mdp.terminations.mjlab_terminations, "root_height_below_minimum", lambda *args: low
  )
  monkeypatch.setattr(
    mdp.terminations.mjlab_terminations, "bad_orientation", lambda *args: tilted
  )
  env = cast(ManagerBasedRlEnv, SimpleNamespace())
  collapsed = mdp.terminations.collapsed(env, minimum_height=0.5, limit_angle=0.7)
  assert collapsed.tolist() == [False, True, False, False]
  assert not torch.any(collapsed & tilted)

  for cfg in (make_locomanip_residual_env_cfg(), locomanip_env_cfg(HRP5P)):
    assert cfg.terminations["collapsed"].func is mdp.terminations.collapsed
    assert (
      cfg.terminations["collapsed"].params["limit_angle"]
      == cfg.terminations["fell_over"].params["limit_angle"]
    )


def test_collapse_guard_warns_early_and_rising_is_safer(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A descending low root pays more, while healthy height pays nothing."""
  heights = torch.tensor([[0.72], [0.54], [0.54]])
  speeds = torch.tensor([[-0.4], [0.2], [-0.4]])
  monkeypatch.setattr(rewards, "root_height", lambda env: heights)
  monkeypatch.setattr(rewards, "root_vertical_velocity", lambda env: speeds)
  env = cast(ManagerBasedRlEnv, SimpleNamespace())
  risk = rewards.collapse_risk(
    env, minimum_height=0.5, warning_margin=0.1, downward_speed=0.3
  )
  torch.testing.assert_close(risk, torch.tensor([0.0, 0.36, 0.72]))


def test_safe_completion_excludes_falls_and_controller_faults(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """The safety verdict only accepts a healthy completed cycle."""
  term = SimpleNamespace(
    datastore_scalar_output=lambda name: torch.ones(5),
    controller_failed=torch.tensor([False, False, False, False, True]),
    controller_worker_failed=torch.tensor([False, False, False, True, False]),
  )
  monkeypatch.setattr(accessors, "residual_action", lambda env, name: term)
  labels = {
    "fell_over": torch.tensor([False, True, False, False, False]),
    "collapsed": torch.tensor([False, False, True, False, False]),
  }
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      termination_manager=SimpleNamespace(
        active_terms=list(labels), get_term=lambda name: labels[name]
      )
    ),
  )
  assert metrics.safe_completion(env).tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_only_the_actor_pays_for_noise() -> None:
  """Verify the policy trains on corrupted observations and the critic does not."""
  observations = make_locomanip_residual_env_cfg().observations
  actor, critic = observations["actor"], observations["critic"]
  assert actor.enable_corruption
  assert not critic.enable_corruption
  noisy = {name for name, term in actor.terms.items() if term.noise is not None}
  assert {"base_lin_vel", "joint_pos"} <= noisy
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


def test_completion_ends_the_episode_without_a_penalty() -> None:
  """Verify the FSM's own verdict truncates, so the Exit state is never entered."""
  terminations = make_locomanip_residual_env_cfg().terminations
  assert terminations["task_complete"].time_out


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
  """Keep cart placement diagnostic while scoring safe controller completion."""
  cfg = make_locomanip_residual_env_cfg()
  cart_success = cfg.metrics["task_success"]
  assert cart_success.reduce == "last"
  assert cart_success.params["position_tolerance_m"] == SUCCESS_POSITION_TOLERANCE_M
  assert cart_success.params["yaw_tolerance_rad"] == SUCCESS_YAW_TOLERANCE_RAD
  assert LOCOMANIP_EVALUATION.success == "safe_completion"
  assert cfg.metrics["safe_completion"].reduce == "last"
  assert cfg.metrics["robot_displacement"].reduce == "last"
  assert {"task_complete", "task_success", "robot_displacement"} <= set(
    LOCOMANIP_EVALUATION.metrics
  )


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
