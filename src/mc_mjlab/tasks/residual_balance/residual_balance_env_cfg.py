"""Residual balance task: keep the mc_rtc controller walking under pushes."""

from __future__ import annotations

import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Literal, get_args

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.curriculums import reward_curriculum
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mc_mjlab import MC_RTC_CONFIG_PATH, mdp
from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionActionCfg,
)
from mc_mjlab.actions.mc_rtc_residual_joint_torque_actions import (
  McRtcResidualJointTorqueActionCfg,
)
from mc_mjlab.bridge.controller_datastore import (
  CONTROL_COM,
  CONTROL_COM_VEL,
  PLANNED_ZMP,
)
from mc_mjlab.robots import robot_module as mc_rtc
from mc_mjlab.robots.registry import (
  get_main_robot_spec,
  prepare_cfg_for_mc_rtc,
)
from mc_mjlab.tasks.residual_balance.curriculum_stages import ACHIEVEMENT_STAGES

MC_RTC_YAML = MC_RTC_CONFIG_PATH / "mc_rtc_hrp5_logistic_ismpc_patched.yaml"

# Only values used twice or more live here; the rest sit in the term that uses them.
# Training bands chosen to cover the qualifier's own push magnitudes (0.40 and
# 0.50-0.60 m/s), which the step curriculum never reaches inside a screen budget.
# docs/difficulty.md#stratified_finite_impulse_curriculum
QUALIFICATION_MATCHED_BANDS = ((0.10, 0.25), (0.25, 0.40), (0.40, 0.60))
# One mixture per way the promotion gate can reject `matched`; standing first.
QUALIFICATION_MATCHED_MIXTURES = {
  "matched": (0.20, 0.25, 0.25, 0.30),
  "gait": (0.35, 0.30, 0.20, 0.15),
  "hazard": (0.10, 0.15, 0.25, 0.50),
}
QUALIFICATION_MATCHED_WEIGHTS = QUALIFICATION_MATCHED_MIXTURES["matched"]
#: 1.46x recovery_dcm's own gated median. docs/reward-shaping.md#recovery_dcm
DCM_STD = 0.06
FALL_LIMIT_ANGLE = math.radians(45.0)
TORQUE_MARGIN_WEIGHT = -0.05
SOLE_VELOCIMETERS = ("left_foot_lin_vel", "right_foot_lin_vel")
CONTROLLER_HISTORY = 20
RECOVERY_DETECTOR_PATH = Path(__file__).resolve().parent / "recovery_detector.json"
AuthoritySet = Literal["uniform", "ankle", "ankle_pitch"]
AUTHORITY_SETS = get_args(AuthoritySet)


def select_residual_joints(
  robot_name: str, candidates: tuple[str, ...], authority_set: str
) -> tuple[str, ...]:
  """Select one anatomy-based authority screen in module joint order."""
  if authority_set == "uniform":
    return candidates
  selected: set[str] = set()
  for side in ("left", "right"):
    leg = [
      joint for joint in mc_rtc.get_leg_joints(robot_name, side) if joint in candidates
    ]
    if authority_set == "ankle":
      selected.update(leg[-2:])
    elif authority_set == "ankle_pitch":
      # Ankle pitch without roll; roll degraded every lateral recovery.
      # docs/residual-authority.md#ankle_pitch
      selected.update(leg[-2:-1])
    else:
      raise ValueError(f"unknown authority set {authority_set!r}")
  return tuple(joint for joint in candidates if joint in selected)


def _reference_stiffness(robot_name: str, path: Path) -> dict[str, float]:
  """Read the position gains in ``refJointOrder`` order."""
  rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
  return {
    joint: float(row[0])
    for joint, row in zip(mc_rtc.get_ref_joint_order(robot_name), rows, strict=True)
  }


def _hardware_residual_scales(
  robot_name: str,
  control: str,
  residual_joints: tuple[str, ...],
  pd_gains_path: Path,
) -> dict[str, float]:
  """Build exact per-actuator scales from effort limits and position gains."""
  limits = mc_rtc.get_effort_limits(robot_name)
  stiffness = _reference_stiffness(robot_name, pd_gains_path)
  default = 0.01 if control == "position" else 10.0
  authority: dict[str, float] = {}
  for joint in residual_joints:
    if joint not in limits:
      raise KeyError(f"no effort limit for residual joint {joint}")
    if control == "position":
      if joint not in stiffness or stiffness[joint] <= 0.0:
        raise KeyError(f"no positive PD stiffness for residual joint {joint}")
      authority[joint] = min(default, 0.20 * limits[joint] / stiffness[joint])
    else:
      authority[joint] = min(default, 0.20 * limits[joint])
  return {
    re.escape(joint): authority.get(joint, default)
    for joint in mc_rtc.get_actuated_joints(robot_name)
  }


def make_residual_balance_env_cfg(
  control: Literal["position", "torque"],
  *,
  num_envs: int = 128,
  num_workers: int | None = None,
  # Tracks the *installed* FSM's walk, which a workspace rebuild reverts.
  episode_length_s: float = 90.0,
  # Difficulty dial; the baseline should almost always fail. docs/difficulty.md
  push_velocity: float = 0.4,
  console_output: Literal["none", "single", "all"] = "none",
  print_residual_every: int = 0,
  mc_rtc_yaml: Path = MC_RTC_YAML,
  recovery_detector_path: Path | None = RECOVERY_DETECTOR_PATH,
  disturbance: Literal["finite", "velocity", "none"] = "finite",
  authority_set: AuthoritySet = "uniform",
  randomization_stage: Literal[0, 1] = 0,
) -> ManagerBasedRlEnvCfg:
  """Build the residual balance env cfg for the config's ``MainRobot``."""
  if control not in ("position", "torque"):
    raise ValueError(f"unknown control mode {control!r}")
  if authority_set not in AUTHORITY_SETS:
    raise ValueError(f"unknown authority set {authority_set!r}")
  if randomization_stage not in (0, 1):
    raise ValueError("randomization_stage must be 0 or 1")

  robot_name, robot = get_main_robot_spec(mc_rtc_yaml)
  robot_cfg = prepare_cfg_for_mc_rtc(
    robot.cfg_fn(), names_collision_geoms=robot.names_collision_geoms
  )
  nominal_height = mc_rtc.get_default_root_position(robot_name)[2]
  if randomization_stage:
    if robot_cfg.articulation is None:
      raise ValueError("randomization requires an articulated robot")
    for actuator in robot_cfg.articulation.actuators:
      actuator.delay_max_lag = 2 * randomization_stage

  # Legs only; the task's opinion, not the robot's. docs/residual-authority.md
  upper_body = set(mc_rtc.get_upper_body_joints(robot_name))
  candidates = tuple(j for j in robot.get_residual_joints() if j not in upper_body)
  residual_joints = select_residual_joints(robot_name, candidates, authority_set)

  residual_scale = (
    (0.01 if control == "position" else 10.0)
    if authority_set == "uniform"
    else _hardware_residual_scales(
      robot_name, control, residual_joints, robot.pd_gains_path
    )
  )

  # Must partition *every* actuator: an unmatched joint silently gets scale 1.0
  # and no clip -- 100x the intended authority. docs/residual-authority.md#residual_scales
  residual_scales: dict[str, float] = (
    residual_scale if isinstance(residual_scale, dict) else {".*": residual_scale}
  )
  residual_clip = {pattern: (-v, v) for pattern, v in residual_scales.items()}

  action_cls = (
    McRtcResidualJointPositionActionCfg
    if control == "position"
    else McRtcResidualJointTorqueActionCfg
  )
  actions: dict[str, ActionTermCfg] = {
    "mc_rtc_residual": action_cls(
      entity_name="robot",
      actuator_names=(".*",),
      residual_actuator_names=residual_joints,
      mc_rtc_config_path=str(mc_rtc_yaml),
      mc_rtc_robot_name=robot_name,
      frameskip=2,
      num_workers=num_workers,
      datastore_vectors_outputs=(
        PLANNED_ZMP,
        CONTROL_COM,
        CONTROL_COM_VEL,
      ),
      pd_gains_path=str(robot.pd_gains_path),
      scale=residual_scales,
      clip=residual_clip,
      recovery_detector_path=(
        str(recovery_detector_path) if recovery_detector_path is not None else None
      ),
      console_output=console_output,
      print_residual_every=print_residual_every,
    )
  }

  # Noise is scaled to this robot's measured levels, not mjlab's ~10x-faster ones.
  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.base_lin_vel,
      noise=Unoise(n_min=-0.02, n_max=0.02),
      history_length=5,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel,
      noise=Unoise(n_min=-0.03, n_max=0.03),
      history_length=5,
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    # `biased=True` is what makes the `encoder_bias` startup event take effect.
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      params={"biased": True},
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    "actions": ObservationTermCfg(
      func=mdp.observations.executed_action, history_length=5
    ),
    "controller_ref_vel": ObservationTermCfg(
      func=mdp.observations.controller_reference_velocity,
      history_length=CONTROLLER_HISTORY,
    ),
    "controller_ref_pos": ObservationTermCfg(
      func=mdp.observations.controller_reference_position,
      history_length=CONTROLLER_HISTORY,
    ),
    "controller_pos_error": ObservationTermCfg(
      func=mdp.observations.controller_position_error,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      history_length=CONTROLLER_HISTORY,
    ),
    "controller_planned_zmp": ObservationTermCfg(
      func=mdp.observations.controller_planned_zmp_offset,
      history_length=CONTROLLER_HISTORY,
    ),
    "controller_planned_com_vel": ObservationTermCfg(
      func=mdp.observations.controller_planned_com_velocity,
      history_length=CONTROLLER_HISTORY,
    ),
    # Deployment-compatible support state; datastore timing stays probe-only.
    "foot_load_share": ObservationTermCfg(
      func=mdp.sensors.foot_load_share, history_length=CONTROLLER_HISTORY
    ),
    "gait_phase": ObservationTermCfg(
      func=mdp.sensors.gait_phase,
      params={
        "sensor_names": mdp.sensors.GROUND_CONTACT_SENSORS,
        "asset_cfg": SceneEntityCfg("robot"),
        # Measured |d_dot| rms over the baseline's load difference.
        "rate_ref": 7.1,
      },
      history_length=CONTROLLER_HISTORY,
    ),
    **{
      name: ObservationTermCfg(
        func=envs_mdp.builtin_sensor,
        params={"sensor_name": f"robot/{name}"},
        history_length=CONTROLLER_HISTORY,
      )
      for name in SOLE_VELOCIMETERS
    },
  }

  # Copy the terms rather than rebuilding them from `func`/`params`: a rebuild
  # silently drops every other field, which is how the critic lost its history.
  critic_terms = {
    name: replace(term, params=dict(term.params)) for name, term in actor_terms.items()
  }
  for name in SOLE_VELOCIMETERS:
    actor_terms.pop(name)
  if randomization_stage:
    for term in actor_terms.values():
      term.delay_max_lag = randomization_stage
  critic_terms["joint_pos"] = replace(critic_terms["joint_pos"], params={})
  critic_terms["controller_pos_error"] = replace(
    critic_terms["controller_pos_error"], params={"biased": False}
  )

  # Privileged, critic-only: exogenous or unobservable, and the largest source of
  # return variance. `push_recency` is bounded on purpose -- see `mdp.observations.push_recency`.
  critic_terms |= {
    "push_recency": ObservationTermCfg(func=mdp.observations.push_recency),
    "last_push_velocity": ObservationTermCfg(func=mdp.observations.last_push_velocity),
    "encoder_bias": ObservationTermCfg(func=mdp.sensors.encoder_bias),
    "measured_zmp_offset": ObservationTermCfg(func=mdp.sensors.measured_zmp_offset),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  # Weights are per second: the manager scales each by step_dt.
  rewards = {
    # -2000 was a 1000x gradient-scale outlier that floored the LR.
    "termination_penalty": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
    "upright": RewardTermCfg(func=envs_mdp.flat_orientation_l2, weight=-2.0),
    # The objective: divergence from the *commanded* one, which mc_rtc's
    # plan-matching cannot buy itself. docs/reward-shaping.md#dcm_stability
    "dcm_stability": RewardTermCfg(
      func=mdp.rewards.dcm_stability,
      weight=0.0,
      params={
        "std": DCM_STD,
        "sensor_names": mdp.sensors.GROUND_CONTACT_SENSORS,
        "asset_cfg": SceneEntityCfg("robot"),
        "action_name": "mc_rtc_residual",
      },
    ),
    "recovery_dcm": RewardTermCfg(
      func=mdp.rewards.recovery_dcm,
      weight=4.0,
      params={
        "std": DCM_STD,
        # 2 s rests on a recovery profile taken before the probe was fixed.
        "window_s": 2.0,
        "sensor_names": mdp.sensors.GROUND_CONTACT_SENSORS,
        "asset_cfg": SceneEntityCfg("robot"),
        "push_term_name": "push_robot",
        "action_name": "mc_rtc_residual",
      },
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.rewards.angular_momentum_l2, weight=-0.005
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.rewards.foot_slip,
      weight=-1.0,
      params={
        "sensor_names": mdp.sensors.GROUND_CONTACT_SENSORS,
        "asset_cfg": SceneEntityCfg("robot"),
        "velocimeter_names": SOLE_VELOCIMETERS,
      },
    ),
    "torque_margin": RewardTermCfg(
      func=mdp.rewards.torque_margin,
      weight=TORQUE_MARGIN_WEIGHT,
      params={"soft_ratio": 0.8, "action_name": "mc_rtc_residual"},
    ),
    "residual_magnitude": RewardTermCfg(
      func=mdp.rewards.requested_action_l2,
      weight=-0.1,
      params={"action_name": "mc_rtc_residual"},
    ),
    "residual_rate": RewardTermCfg(
      func=mdp.rewards.requested_action_rate_l2,
      weight=-0.1,
      params={"action_name": "mc_rtc_residual"},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation, params={"limit_angle": FALL_LIMIT_ANGLE}
    ),
    # Crouch-collapse keeps the trunk upright, so `fell_over` misses it.
    "collapsed": TerminationTermCfg(
      func=mdp.terminations.collapsed,
      params={
        "minimum_height": 0.7 * nominal_height,
        "limit_angle": FALL_LIMIT_ANGLE,
      },
    ),
    "controller_failed": TerminationTermCfg(
      func=mdp.terminations.controller_failed, params={"action_name": "mc_rtc_residual"}
    ),
    # `time_out=True` is the point: exogenous, so bootstrap, and no fall penalty.
    "controller_worker_failed": TerminationTermCfg(
      func=mdp.terminations.controller_worker_failed,
      params={"action_name": "mc_rtc_residual"},
      time_out=True,
    ),
  }

  events: dict[str, EventTermCfg] = {
    # Must come first and must not be dropped: declaring `events` at all replaces
    # mjlab's default, and this is the only term that resets *joints*. Without it
    # every episode starts from qpos0 and the FSM never reaches walking.
    "reset_scene_to_default": EventTermCfg(
      func=envs_mdp.reset_scene_to_default, mode="reset"
    ),
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        # Safe only because the reset teleport reconciles the controller's frame
        # every episode. docs/coupling.md#reset-pose-seeding
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
        # Not an option: `reset()` takes encoders and a pose but no velocity, so
        # the controller would start believing something false.
        "velocity_range": {},
      },
    ),
    "encoder_bias": EventTermCfg(
      func=dr.encoder_bias,
      mode="startup",
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
    ),
  }
  if randomization_stage:
    physics_scale = 0.05 * randomization_stage
    friction_scale = 0.10 * randomization_stage
    events |= {
      "randomize_friction": EventTermCfg(
        func=dr.geom_friction,
        mode="startup",
        params={
          "ranges": (1.0 - friction_scale, 1.0 + friction_scale),
          "operation": "scale",
          "asset_cfg": SceneEntityCfg("robot"),
        },
      ),
      "randomize_pd_gains": EventTermCfg(
        func=mdp.disturbances.randomize_current_pd_gains,
        mode="startup",
        params={
          "scale_range": (1.0 - physics_scale, 1.0 + physics_scale),
          "asset_cfg": SceneEntityCfg("robot"),
        },
      ),
      "randomize_strength": EventTermCfg(
        func=dr.effort_limits,
        mode="startup",
        params={
          "effort_limit_range": (1.0 - physics_scale, 1.0 + physics_scale),
          "asset_cfg": SceneEntityCfg("robot"),
        },
      ),
    }
  if disturbance == "velocity":
    events["push_robot"] = EventTermCfg(
      func=mdp.disturbances.push_and_record,
      mode="interval",
      interval_range_s=(5.0, 7.0),
      params={
        "velocity_range": {
          "x": (-push_velocity, push_velocity),
          "y": (-push_velocity, push_velocity),
          "roll": (0.0, 0.0),
          "pitch": (0.0, 0.0),
        },
        "warmup_s": 10.0,
      },
    )
  else:
    events["push_robot"] = EventTermCfg(
      func=mdp.disturbances.finite_impulse_curriculum,
      mode="step",
      params={
        "enabled": disturbance == "finite",
        "interval_range_s": (5.0, 7.0),
        "warmup_s": 10.0,
        "duration_range_s": (0.08, 0.20),
        "height_range_m": (0.0, 0.25),
        "stages": (
          (0, (0.10, 0.25)),
          (48_000, (0.10, 0.40)),
          (96_000, (0.10, 0.50)),
        ),
        "asset_cfg": SceneEntityCfg("robot"),
      },
    )

  # Read tracking quality as `zmp_error / zmp_grounded`, never zmp_error alone.
  metric_params = {
    "sensor_names": mdp.sensors.GROUND_CONTACT_SENSORS,
    "asset_cfg": SceneEntityCfg("robot"),
    "action_name": "mc_rtc_residual",
  }
  # Ramped so early training does not charge mc_rtc alone. docs/reward-shaping.md
  curriculum = {
    "torque_margin_weight": CurriculumTermCfg(
      func=reward_curriculum,
      params={
        "reward_name": "torque_margin",
        "stages": [
          {"step": 0, "weight": TORQUE_MARGIN_WEIGHT},
          {"step": 48_000, "weight": TORQUE_MARGIN_WEIGHT * 4},
          {"step": 96_000, "weight": TORQUE_MARGIN_WEIGHT * 10},
        ],
      },
    )
  }

  metrics = {
    "zmp_error": MetricsTermCfg(func=mdp.metrics.zmp_error, params=dict(metric_params)),
    "zmp_grounded": MetricsTermCfg(
      func=mdp.metrics.zmp_grounded, params=dict(metric_params)
    ),
    "gate_mean": MetricsTermCfg(func=mdp.metrics.gate_mean),
    "detector_score": MetricsTermCfg(func=mdp.metrics.detector_score, reduce="max"),
    "inactive_residual_violation": MetricsTermCfg(
      func=mdp.metrics.inactive_residual_violation, reduce="max"
    ),
    "projection_fraction": MetricsTermCfg(func=mdp.metrics.projection_fraction),
    "near_bound_fraction": MetricsTermCfg(func=mdp.metrics.near_bound_fraction),
    "executed_residual_l2": MetricsTermCfg(func=mdp.rewards.action_l2),
    "impulse_speed": MetricsTermCfg(func=mdp.metrics.impulse_speed),
    "requested_residual_l2": MetricsTermCfg(func=mdp.rewards.requested_action_l2),
    "requested_residual_rate_l2": MetricsTermCfg(
      func=mdp.rewards.requested_action_rate_l2
    ),
    "max_effort_ratio": MetricsTermCfg(
      func=mdp.metrics.nominal_effort_ratio,
      params={"action_name": "mc_rtc_residual"},
      reduce="max",
      per_substep=True,
    ),
    "com_velocity_error": MetricsTermCfg(
      func=mdp.metrics.com_velocity_error,
      params={"asset_cfg": SceneEntityCfg("robot"), "action_name": "mc_rtc_residual"},
    ),
    "dcm_error": MetricsTermCfg(func=mdp.metrics.dcm_error, params=dict(metric_params)),
    "recovery_dcm_error": MetricsTermCfg(
      func=mdp.metrics.recovery_dcm_error,
      params={**metric_params, "window_s": 2.0, "push_term_name": "push_robot"},
    ),
    # Detector recall inside the scored window: read over `recovery_active`.
    "recovery_authority_coverage": MetricsTermCfg(
      func=mdp.metrics.recovery_authority_coverage,
      params={**metric_params, "window_s": 2.0, "push_term_name": "push_robot"},
    ),
    "recovery_active": MetricsTermCfg(
      func=mdp.metrics.recovery_active,
      params={
        "sensor_names": mdp.sensors.GROUND_CONTACT_SENSORS,
        "asset_cfg": SceneEntityCfg("robot"),
        "window_s": 2.0,
        "push_term_name": "push_robot",
      },
    ),
    "foot_slip": MetricsTermCfg(
      func=mdp.rewards.foot_slip,
      params={
        "sensor_names": mdp.sensors.GROUND_CONTACT_SENSORS,
        "asset_cfg": SceneEntityCfg("robot"),
        "velocimeter_names": SOLE_VELOCIMETERS,
      },
    ),
  }

  # Solver settings follow mc_mujoco's HRP5Pmain.xml, as in the demo.
  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=num_envs,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot_cfg},
    ),
    observations=observations,
    actions=actions,
    curriculum=curriculum,
    rewards=rewards,
    terminations=terminations,
    events=events,
    metrics=metrics,
    decimation=20,
    episode_length_s=episode_length_s,
    sim=SimulationCfg(
      # A hard per-world cap: past it MuJoCo drops rows and simulates the fall
      # wrong rather than failing. Budget for a sprawled robot, not a stance.
      njmax=1500,
      nconmax=100,
      mujoco=MujocoCfg(
        timestep=0.001,
        integrator="euler",
        solver="newton",
        iterations=50,
        tolerance=1e-10,
        jacobian="dense",
      ),
    ),
  )


def _apply_play_overrides(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Retune a training cfg for a viewer session."""
  cfg.scene.num_envs = 1  # each env is its own ~70 MB controller, built serially
  cfg.episode_length_s = 1e10  # only a fall should end an episode
  cfg.observations["actor"].enable_corruption = False
  return cfg


# "single" not "all": keeps env 0 readable if `--num-envs` grows.
PLAY_CONSOLE_OUTPUT = "single"

# 50 Hz is unreadable; every 10th step is 5 Hz. `MC_MJLAB_PRINT_RESIDUAL` retunes.
PLAY_PRINT_RESIDUAL_EVERY = 10


def residual_balance_position_env_cfg(
  play: bool = False,
  authority_set: AuthoritySet = "uniform",
  randomization_stage: Literal[0, 1] = 0,
) -> ManagerBasedRlEnvCfg:
  """Residual on the controller's joint *position* targets."""
  cfg = make_residual_balance_env_cfg(
    control="position",
    console_output=PLAY_CONSOLE_OUTPUT if play else "none",
    print_residual_every=PLAY_PRINT_RESIDUAL_EVERY if play else 0,
    authority_set=authority_set,
    randomization_stage=randomization_stage,
  )
  if play:
    _apply_play_overrides(cfg)
  return cfg


def residual_balance_position_matched_impulse_env_cfg(
  play: bool = False,
  mixture: Literal["matched", "gait", "hazard"] = "matched",
  authority_set: Literal["ankle", "ankle_pitch"] = "ankle",
) -> ManagerBasedRlEnvCfg:
  """Build the task whose pushes span the qualifier's range at one authority."""
  cfg = residual_balance_position_env_cfg(play=play, authority_set=authority_set)
  push = cfg.events["push_robot"]
  push.func = mdp.disturbances.stratified_finite_impulse_curriculum
  push.params["bands"] = QUALIFICATION_MATCHED_BANDS
  push.params["band_weights"] = QUALIFICATION_MATCHED_MIXTURES[mixture]
  union = (
    QUALIFICATION_MATCHED_BANDS[0][0],
    QUALIFICATION_MATCHED_BANDS[-1][1],
  )
  push.params["stages"] = ((0, union),)
  return cfg


def residual_balance_position_achievement_curriculum_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the ankle-authority achievement-gated training task."""
  cfg = residual_balance_position_env_cfg(play=play, authority_set="ankle")
  push = cfg.events["push_robot"]
  push.func = mdp.disturbances.achievement_finite_impulse_curriculum
  push.params["stages"] = tuple(
    (index, stage.training_velocity_range)
    for index, stage in enumerate(ACHIEVEMENT_STAGES)
  )
  push.params["rehearsal_weights"] = tuple(
    stage.rehearsal_weights for stage in ACHIEVEMENT_STAGES
  )
  push.params["initial_stage"] = len(ACHIEVEMENT_STAGES) - 1 if play else 0
  if play:
    weights = [0.0] * (len(ACHIEVEMENT_STAGES) + 1)
    weights[-1] = 1.0
    push.params["rehearsal_weights"] = tuple(
      tuple(weights) for _stage in ACHIEVEMENT_STAGES
    )
  cfg.curriculum = {
    "achievement_stage": CurriculumTermCfg(
      func=mdp.curricula.achievement_curriculum_state
    )
  }
  return cfg


def residual_balance_torque_env_cfg(
  play: bool = False,
  authority_set: AuthoritySet = "uniform",
  randomization_stage: Literal[0, 1] = 0,
) -> ManagerBasedRlEnvCfg:
  """Residual on the controller's joint *torques*."""
  cfg = make_residual_balance_env_cfg(
    control="torque",
    console_output=PLAY_CONSOLE_OUTPUT if play else "none",
    print_residual_every=PLAY_PRINT_RESIDUAL_EVERY if play else 0,
    authority_set=authority_set,
    randomization_stage=randomization_stage,
  )
  if play:
    _apply_play_overrides(cfg)
  return cfg
