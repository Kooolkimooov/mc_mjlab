"""ResidualMPC reproduction task for HRP5P and LogisticController_ismpc."""

from __future__ import annotations

import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg

from mc_mjlab import MC_RTC_YAML_PATH
from mc_mjlab.actions.residual_mpc_joint_torque_action import (
  ResidualMpcJointTorqueActionCfg,
)
from mc_mjlab.robots import mc_rtc_robot_configuration as mc_rtc
from mc_mjlab.robots.robots_registry import (
  get_main_robot_spec,
  prepare_cfg_for_mc_rtc,
)
from mc_mjlab.tasks import mdp as shared_mdp
from mc_mjlab.tasks.residual_mpc import mdp

POLICY_DT = 0.01
BLEND_FACTOR = 0.1
SELF_COLLISION_SENSOR = "self_collision"
COMMAND_NAME = "twist"
#: Commanded (vx, vy, wz) box. The paper's claim is about commands the MPC prior
#: cannot track, so this must straddle its failure boundary, not sit inside it.
#: docs/residual-mpc.md#COMMAND_RANGES
COMMAND_RANGES = ((0.0, 0.50), (0.0, 0.0), (0.0, 0.0))
#: Sized so the bare ISMPC scores about two thirds at the top of the box,
#: leaving a third for the residual. The paper does not fix sigma.
#: docs/residual-mpc.md#LINEAR_TRACKING_SIGMA
LINEAR_TRACKING_SIGMA = 0.06
#: Baseline-relative offset for the planner's cruise speed (installed default
#: 0.1 m/s). Not yet wired: the pool fails during configure with an empty
#: payload when it is. docs/residual-mpc.md#mean_speed
MEAN_SPEED_OFFSET = 0.20
MEAN_SPEED_COMMANDS = (
  ("footsteps_planner::get_mean_speed", "footsteps_planner::set_mean_speed"),
)


def residual_mpc_env_cfg(
  play: bool = False,
  num_envs: int = 128,
  num_workers: int | None = None,
  randomization: bool = True,
  pushes: bool = True,
  fixed_twist: tuple[float, float, float] | None = None,
  command_ranges: tuple[
    tuple[float, float], tuple[float, float], tuple[float, float]
  ] = COMMAND_RANGES,
  console_output: Literal["none", "single", "all"] = "none",
  mc_rtc_yaml: Path = MC_RTC_YAML_PATH,
) -> ManagerBasedRlEnvCfg:
  """Build the forward-only paper-style ResidualMPC task."""
  robot_name, robot = get_main_robot_spec(mc_rtc_yaml)
  if robot_name != "HRP5P":
    raise ValueError(f"ResidualMPC reproduction requires HRP5P, got {robot_name}")
  robot_cfg = prepare_cfg_for_mc_rtc(
    robot.cfg_fn(), names_collision_geoms=robot.names_collision_geoms
  )
  if robot_cfg.articulation is None:
    raise ValueError("ResidualMPC requires an articulated robot")
  for actuator in robot_cfg.articulation.actuators:
    actuator.delay_max_lag = 2 if randomization else 0

  actuated = mc_rtc.get_actuated_joints(robot_name)
  leg_joints = mc_rtc.get_leg_joints(robot_name)
  if len(leg_joints) != 12:
    raise ValueError(f"expected HRP5P's 12 leg joints, got {len(leg_joints)}")
  nominal_height = mc_rtc.get_default_root_position(robot_name)[2]
  joint_cfg = SceneEntityCfg("robot", joint_names=tuple(map(re.escape, actuated)))

  actions: dict[str, ActionTermCfg] = {
    mdp.ACTION_NAME: ResidualMpcJointTorqueActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      residual_actuator_names=leg_joints,
      mc_rtc_config_path=str(mc_rtc_yaml),
      mc_rtc_robot_name=robot_name,
      frameskip=2,
      num_workers=num_workers,
      pd_gains_path=str(robot.pd_gains_path),
      scale=1.0,
      blend_factor=BLEND_FACTOR,
      action_scale_blend_factor=BLEND_FACTOR,
      controller_scalars=(
        mdp.QP_OBJECTIVE_CALLBACK,
        mdp.STEP_TIME_CALLBACK,
        mdp.STEP_DURATION_CALLBACK,
        mdp.SUPPORT_FOOT_CALLBACK,
      ),
      walking_velocity_command_name=COMMAND_NAME,
      # Read back so a run shows the reference the controller actually holds.
      controller_vectors=("walking_ref_vel",),
      console_output="single" if play else console_output,
      print_residual_every=10 if play else 0,
    )
  }

  # A held twist is how the envelope sweep asks for one command per episode; the
  # box is otherwise resampled. docs/residual-mpc.md#COMMAND_RANGES
  vx, vy, wz = (
    command_ranges
    if fixed_twist is None
    else tuple((value, value) for value in fixed_twist)
  )
  commands: dict[str, CommandTermCfg] = {
    COMMAND_NAME: UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.0 if fixed_twist is not None else 0.1,
      heading_command=False,
      debug_vis=play,
      ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=vx, lin_vel_y=vy, ang_vel_z=wz),
    )
  }

  observation_terms = {
    "root_position": ObservationTermCfg(func=mdp.root_position),
    "root_quaternion": ObservationTermCfg(func=mdp.root_quaternion),
    "joint_position": ObservationTermCfg(
      func=mdp.joint_position, params={"asset_cfg": joint_cfg}
    ),
    "joint_velocity": ObservationTermCfg(
      func=mdp.joint_velocity, params={"asset_cfg": joint_cfg}
    ),
    "body_linear_velocity": ObservationTermCfg(func=mdp.body_linear_velocity),
    "body_angular_velocity": ObservationTermCfg(func=mdp.body_angular_velocity),
    "contact_phases": ObservationTermCfg(func=mdp.controller_contact_phases),
    "mpc_objective": ObservationTermCfg(func=mdp.controller_qp_objective),
  }
  observations = {
    "actor": ObservationGroupCfg(
      terms={
        name: replace(term, params=dict(term.params))
        for name, term in observation_terms.items()
      },
      concatenate_terms=True,
    ),
    "critic": ObservationGroupCfg(
      terms={
        name: replace(term, params=dict(term.params))
        for name, term in observation_terms.items()
      },
      concatenate_terms=True,
    ),
  }

  rewards = {
    "linear_tracking": RewardTermCfg(
      func=mdp.linear_velocity_tracking,
      weight=10.0,
      params={"command_name": COMMAND_NAME, "sigma": LINEAR_TRACKING_SIGMA},
    ),
    "angular_tracking": RewardTermCfg(
      func=mdp.angular_velocity_tracking,
      weight=5.0,
      params={"command_name": COMMAND_NAME, "sigma": 0.5},
    ),
    "first_action_rate": RewardTermCfg(func=mdp.first_action_rate, weight=-1.0e-3),
    "second_action_rate": RewardTermCfg(func=mdp.second_action_rate, weight=-1.0e-4),
    "torque_l2": RewardTermCfg(func=mdp.torque_l2, weight=-1.0e-4),
    "orientation": RewardTermCfg(
      func=mdp.orientation_reward, weight=1.0, params={"sigma": 0.5}
    ),
    "height": RewardTermCfg(
      func=mdp.height_reward,
      weight=1.0,
      params={"target_height": nominal_height, "sigma": 0.5},
    ),
    # Table I prints positive raw MSE with +1, which rewards deviation.
    # docs/residual-mpc.md
    "joint_regularization": RewardTermCfg(
      func=mdp.joint_regularization,
      weight=-1.0,
      params={"asset_cfg": joint_cfg},
    ),
    "self_collision": RewardTermCfg(
      func=mdp.self_collision,
      weight=-1.0,
      params={"sensor_name": SELF_COLLISION_SENSOR},
    ),
    "termination": RewardTermCfg(func=envs_mdp.is_terminated, weight=-100.0),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "self_collision": TerminationTermCfg(
      func=mdp.self_collision, params={"sensor_name": SELF_COLLISION_SENSOR}
    ),
    "base_speed": TerminationTermCfg(
      func=mdp.excessive_base_speed, params={"limit": 10.0}
    ),
    "angular_speed": TerminationTermCfg(
      func=mdp.excessive_angular_speed, params={"limit": 5.0}
    ),
    "orientation": TerminationTermCfg(
      func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(45.0)}
    ),
    "height": TerminationTermCfg(
      func=mdp.height_outside,
      params={"minimum": 0.7 * nominal_height, "maximum": 1.3 * nominal_height},
    ),
    "controller_failed": TerminationTermCfg(func=mdp.controller_failed),
    "controller_worker_failed": TerminationTermCfg(
      func=mdp.controller_worker_failed, time_out=True
    ),
  }

  events: dict[str, EventTermCfg] = {
    "reset_scene_to_default": EventTermCfg(
      func=envs_mdp.reset_scene_to_default, mode="reset"
    ),
    "encoder_bias": EventTermCfg(
      func=dr.encoder_bias,
      mode="startup",
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
    ),
  }
  if randomization:
    events |= {
      "randomize_friction": EventTermCfg(
        func=dr.geom_friction,
        mode="startup",
        params={
          "ranges": (0.9, 1.1),
          "operation": "scale",
          "asset_cfg": SceneEntityCfg("robot"),
        },
      ),
      "randomize_pd_gains": EventTermCfg(
        func=shared_mdp.randomize_current_pd_gains,
        mode="startup",
        params={"scale_range": (0.95, 1.05), "action_name": mdp.ACTION_NAME},
      ),
      "randomize_effort": EventTermCfg(
        func=dr.effort_limits,
        mode="startup",
        params={
          "effort_limit_range": (0.95, 1.05),
          "operation": "scale",
          "asset_cfg": SceneEntityCfg("robot"),
        },
      ),
      "refresh_action_scaling": EventTermCfg(
        func=mdp.refresh_action_scaling, mode="startup"
      ),
    }
  events["push_robot"] = EventTermCfg(
    func=shared_mdp.finite_impulse_curriculum,
    mode="step",
    params={
      "enabled": pushes,
      "interval_range_s": (5.0, 7.0),
      "warmup_s": 10.0,
      "duration_range_s": (0.08, 0.20),
      "height_range_m": (0.0, 0.25),
      "stages": ((0, (0.10, 0.25)),),
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  self_collision_sensor = ContactSensorCfg(
    name=SELF_COLLISION_SENSOR,
    primary=ContactMatch(
      mode="subtree",
      pattern=re.escape(mc_rtc.get_root_body(robot_name)),
      entity="robot",
    ),
    secondary=ContactMatch(
      mode="subtree",
      pattern=re.escape(mc_rtc.get_root_body(robot_name)),
      entity="robot",
    ),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=10,
  )
  metrics = {
    "projection_fraction": MetricsTermCfg(func=mdp.projection_fraction),
    "maximum_effort_ratio": MetricsTermCfg(
      func=mdp.maximum_effort_ratio, reduce="max", per_substep=True
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=1 if play else num_envs,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot_cfg},
      sensors=(self_collision_sensor,),
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    rewards=rewards,
    terminations=terminations,
    events=events,
    metrics=metrics,
    decimation=10,
    episode_length_s=30.0,
    sim=SimulationCfg(
      njmax=1500,
      nconmax=100,
      mujoco=MujocoCfg(
        timestep=0.001,
        integrator="euler",
        solver="newton",
        iterations=50,
        tolerance=1.0e-10,
        jacobian="dense",
      ),
    ),
  )
