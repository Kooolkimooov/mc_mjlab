"""Residual balance task: keep the mc_rtc controller walking under pushes."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Literal

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

from mc_mjlab import MC_RTC_YAML_PATH
from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionActionCfg,
)
from mc_mjlab.actions.mc_rtc_residual_joint_torque_actions import (
  McRtcResidualJointTorqueActionCfg,
)
from mc_mjlab.robots import mc_rtc_robot_configuration as mc_rtc
from mc_mjlab.robots.robots_registry import (
  get_main_robot_spec,
  prepare_cfg_for_mc_rtc,
)
from mc_mjlab.tasks import mdp

# Only values used twice or more live here; the rest sit in the term that uses them.
DCM_STD = 0.05
FALL_LIMIT_ANGLE = math.radians(45.0)
TORQUE_MARGIN_WEIGHT = -0.05
SOLE_VELOCIMETERS = ("left_foot_lin_vel", "right_foot_lin_vel")
CONTROLLER_HISTORY = 20


def _make_env_cfg(
  control: str,
  num_envs: int = 128,
  num_workers: int | None = None,
  residual_scale: float | dict[str, float] | None = None,
  # Tracks the *installed* FSM's walk, which a workspace rebuild reverts.
  episode_length_s: float = 90.0,
  # Difficulty dial; the baseline should almost always fail. docs/difficulty.md
  push_velocity: float = 0.4,
  push_angular_velocity: float = 0.0,
  console_output: Literal["none", "single", "all"] = "none",
  print_residual_every: int = 0,
  mc_rtc_yaml: Path = MC_RTC_YAML_PATH,
) -> ManagerBasedRlEnvCfg:
  """Build the residual balance env cfg for the config's ``MainRobot``."""
  robot_name, robot = get_main_robot_spec(mc_rtc_yaml)
  robot_cfg = prepare_cfg_for_mc_rtc(
    robot.cfg_fn(), names_collision_geoms=robot.names_collision_geoms
  )
  nominal_height = mc_rtc.get_default_root_position(robot_name)[2]

  # Legs only; the task's opinion, not the robot's. docs/residual-authority.md
  upper_body = set(mc_rtc.get_upper_body_joints(robot_name))
  residual_joints = tuple(j for j in robot.get_residual_joints() if j not in upper_body)

  # User-requested authority sweep; re-measure before treating 0.20 rad as safe.
  if residual_scale is None:
    residual_scale = 0.20 if control == "position" else 10.0

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
      controller_vectors=("planned_zmp", "control_com", "control_com_vel"),
      pd_gains_path=str(robot.pd_gains_path),
      scale=residual_scales,
      clip=residual_clip,
      # 0.0: attenuation stayed flat over 1500 iterations and the win vanished.
      gate_strength=0.0,
      gate_alpha_ref=0.5,
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
    "actions": ObservationTermCfg(func=mdp.executed_action, history_length=5),
    "controller_ref_vel": ObservationTermCfg(
      func=mdp.controller_reference_velocity, history_length=CONTROLLER_HISTORY
    ),
    "controller_ref_pos": ObservationTermCfg(
      func=mdp.controller_reference_position, history_length=CONTROLLER_HISTORY
    ),
    "controller_pos_error": ObservationTermCfg(
      func=mdp.controller_position_error,
      noise=Unoise(n_min=-0.01, n_max=0.01),
      history_length=CONTROLLER_HISTORY,
    ),
    "controller_planned_zmp": ObservationTermCfg(
      func=mdp.controller_planned_zmp_offset, history_length=CONTROLLER_HISTORY
    ),
    "controller_planned_com_vel": ObservationTermCfg(
      func=mdp.controller_planned_com_velocity, history_length=CONTROLLER_HISTORY
    ),
    # Gait phase, sim-side: mc_rtc's walking plan is unreachable through the
    # bindings, so support state and sole motion stand in for it.
    "foot_load_share": ObservationTermCfg(
      func=mdp.foot_load_share, history_length=CONTROLLER_HISTORY
    ),
    "gait_phase": ObservationTermCfg(
      func=mdp.gait_phase,
      params={
        "sensor_names": mdp.GROUND_CONTACT_SENSORS,
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
  critic_terms["joint_pos"] = replace(critic_terms["joint_pos"], params={})
  critic_terms["controller_pos_error"] = replace(
    critic_terms["controller_pos_error"], params={"biased": False}
  )

  # Privileged, critic-only: exogenous or unobservable, and the largest source of
  # return variance. `push_recency` is bounded on purpose -- see `mdp.push_recency`.
  critic_terms |= {
    "push_recency": ObservationTermCfg(func=mdp.push_recency),
    "last_push_velocity": ObservationTermCfg(func=mdp.last_push_velocity),
    "encoder_bias": ObservationTermCfg(func=mdp.encoder_bias),
    "measured_zmp_offset": ObservationTermCfg(func=mdp.measured_zmp_offset),
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
      func=mdp.dcm_stability,
      weight=1.0,
      params={
        "std": DCM_STD,
        "sensor_names": mdp.GROUND_CONTACT_SENSORS,
        "asset_cfg": SceneEntityCfg("robot"),
        "action_name": "mc_rtc_residual",
      },
    ),
    "recovery_dcm": RewardTermCfg(
      func=mdp.recovery_dcm,
      weight=1.0,
      params={
        "std": DCM_STD,
        # 2 s rests on a recovery profile taken before the probe was fixed.
        "window_s": 2.0,
        "sensor_names": mdp.GROUND_CONTACT_SENSORS,
        "asset_cfg": SceneEntityCfg("robot"),
        "push_term_name": "push_robot",
        "action_name": "mc_rtc_residual",
      },
    ),
    "angular_momentum": RewardTermCfg(func=mdp.angular_momentum_l2, weight=-0.005),
    "foot_slip": RewardTermCfg(
      func=mdp.foot_slip,
      weight=-1.0,
      params={
        "sensor_names": mdp.GROUND_CONTACT_SENSORS,
        "asset_cfg": SceneEntityCfg("robot"),
        "velocimeter_names": SOLE_VELOCIMETERS,
      },
    ),
    "torque_margin": RewardTermCfg(
      func=mdp.torque_margin,
      weight=TORQUE_MARGIN_WEIGHT,
      params={"soft_ratio": 1.0, "action_name": "mc_rtc_residual"},
    ),
    "residual_magnitude": RewardTermCfg(func=mdp.action_l2, weight=-0.1),
    "residual_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation, params={"limit_angle": FALL_LIMIT_ANGLE}
    ),
    # Crouch-collapse keeps the trunk upright, so `fell_over` misses it.
    "collapsed": TerminationTermCfg(
      func=mdp.collapsed,
      params={
        "minimum_height": 0.7 * nominal_height,
        "limit_angle": FALL_LIMIT_ANGLE,
      },
    ),
    "controller_failed": TerminationTermCfg(
      func=mdp.controller_failed, params={"action_name": "mc_rtc_residual"}
    ),
    # `time_out=True` is the point: exogenous, so bootstrap, and no fall penalty.
    "controller_worker_failed": TerminationTermCfg(
      func=mdp.controller_worker_failed,
      params={"action_name": "mc_rtc_residual"},
      time_out=True,
    ),
  }

  events = {
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
    "push_robot": EventTermCfg(
      # Records *when* it fired; `recovery_tracking` gates on that, and
      # `steps_since_push` raises if this is swapped for the plain term.
      func=mdp.push_and_record,
      mode="interval",
      interval_range_s=(5.0, 7.0),
      params={
        "velocity_range": {
          "x": (-push_velocity, push_velocity),
          "y": (-push_velocity, push_velocity),
          "roll": (-push_angular_velocity, push_angular_velocity),
          "pitch": (-push_angular_velocity, push_angular_velocity),
        },
        "warmup_s": 10.0,
      },
    ),
    "encoder_bias": EventTermCfg(
      func=dr.encoder_bias,
      mode="startup",
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
    ),
  }

  # Read tracking quality as `zmp_error / zmp_grounded`, never zmp_error alone.
  metric_params = {
    "sensor_names": mdp.GROUND_CONTACT_SENSORS,
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
    "zmp_error": MetricsTermCfg(func=mdp.zmp_error, params=dict(metric_params)),
    "zmp_grounded": MetricsTermCfg(func=mdp.zmp_grounded, params=dict(metric_params)),
    "gate_mean": MetricsTermCfg(func=mdp.gate_mean),
    "com_velocity_error": MetricsTermCfg(
      func=mdp.com_velocity_error,
      params={"asset_cfg": SceneEntityCfg("robot"), "action_name": "mc_rtc_residual"},
    ),
    "dcm_error": MetricsTermCfg(func=mdp.dcm_error, params=dict(metric_params)),
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


def residual_balance_position_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Residual on the controller's joint *position* targets."""
  cfg = _make_env_cfg(
    control="position",
    console_output=PLAY_CONSOLE_OUTPUT if play else "none",
    print_residual_every=PLAY_PRINT_RESIDUAL_EVERY if play else 0,
  )
  if play:
    _apply_play_overrides(cfg)
  return cfg


def residual_balance_torque_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Residual on the controller's joint *torques*."""
  cfg = _make_env_cfg(
    control="torque",
    console_output=PLAY_CONSOLE_OUTPUT if play else "none",
    print_residual_every=PLAY_PRINT_RESIDUAL_EVERY if play else 0,
  )
  if play:
    _apply_play_overrides(cfg)
  return cfg
