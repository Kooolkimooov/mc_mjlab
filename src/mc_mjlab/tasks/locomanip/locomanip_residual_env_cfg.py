"""Trainable locomanip task: an RL residual on the cart-pushing controller."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from mc_mjlab import mdp
from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionActionCfg,
)
from mc_mjlab.bridge.controller_datastore import CONTROL_COM, PLANNED_ZMP
from mc_mjlab.robots import robot_module as mc_rtc
from mc_mjlab.robots.registry import (
  RobotSpec,
  get_main_robot_spec,
  prepare_cfg_for_mc_rtc,
)
from mc_mjlab.tasks.locomanip import mdp as locomanip_mdp
from mc_mjlab.tasks.locomanip.cart import (
  cart_cfg,
  cart_floor_contact,
  hand_cart_contact_sensors,
)
from mc_mjlab.tasks.locomanip.locomanip_env_cfg import MC_RTC_YAML

#: 1 kHz physics, 500 Hz controller, 50 Hz policy; must divide by `FRAMESKIP`.
DECIMATION = 20
FRAMESKIP = 2

# One Locomanip controller per environment, built serially, plus its own cart:
# memory-bound, not GPU-bound. Raise it once you have measured the machine.
NUM_ENVS = 256
NUM_WORKERS = 64

# The installed DemoFSM's reach-push-release cycle takes 52.2 s. docs/locomanip.md
EPISODE_LENGTH_S = 60.0

#: Position residual bound, in radians. docs/locomanip.md#residual_scale
RESIDUAL_SCALE = 0.01

#: Kernel width of the object-tracking reward, in metres. docs/locomanip.md#object_tracking_std
OBJECT_TRACKING_STD = 0.10

#: Kernel width of the ZMP-tracking reward, in metres. docs/locomanip.md#zmp_tracking_std
ZMP_TRACKING_STD = 0.05

#: The wrench channels the policy gets: both feet, both hands.
FORCE_SENSORS = (
  "LeftFootForceSensor_fsensor",
  "RightFootForceSensor_fsensor",
  "LeftHandForceSensor_fsensor",
  "RightHandForceSensor_fsensor",
)

ZMP_PARAMS = {
  "sensor_names": mdp.sensors.GROUND_CONTACT_SENSORS,
  "asset_cfg": SceneEntityCfg("robot"),
}

#: The cart asset's own mass, which is what mc_rtc's model keeps believing.
CART_NOMINAL_MASS_KG = 10.0

#: The masses the mc_mujoco sweep characterised. docs/locomanip.md#cart_mass_range_kg
CART_MASS_RANGE_KG = (1.0, 1000.0)

FALL_LIMIT_ANGLE = math.radians(45.0)


def mass_alpha_range(
  mass_range_kg: tuple[float, float],
  nominal_mass_kg: float = CART_NOMINAL_MASS_KG,
) -> tuple[float, float]:
  """Convert a mass range to `pseudo_inertia`'s log scale, where mass is e^(2a)."""
  low, high = mass_range_kg
  if low <= 0.0 or high < low:
    raise ValueError(f"invalid cart mass range {mass_range_kg}")
  return (
    0.5 * math.log(low / nominal_mass_kg),
    0.5 * math.log(high / nominal_mass_kg),
  )


def locomanip_residual_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the trainable locomanip residual task, or its viewer variant."""
  cfg = make_locomanip_residual_env_cfg(
    console_output="single" if play else "none",
    print_residual_every=PLAY_PRINT_RESIDUAL_EVERY if play else 0,
  )
  if play:
    cfg.scene.num_envs = 1
    cfg.observations["actor"].enable_corruption = False
  return cfg


def make_locomanip_residual_env_cfg(
  *,
  num_envs: int = NUM_ENVS,
  num_workers: int = NUM_WORKERS,
  episode_length_s: float = EPISODE_LENGTH_S,
  residual_scale: float = RESIDUAL_SCALE,
  cart_pose_range: dict[str, tuple[float, float]] | None = None,
  cart_mass_range_kg: tuple[float, float] | None = CART_MASS_RANGE_KG,
  console_output: Literal["none", "single", "all"] = "none",
  print_residual_every: int = 0,
  mc_rtc_yaml: Path = MC_RTC_YAML,
) -> ManagerBasedRlEnvCfg:
  """Assemble the managers from the builders below, one builder per manager."""
  robot_name, robot = get_main_robot_spec(mc_rtc_yaml)
  robot_cfg = prepare_cfg_for_mc_rtc(
    robot.cfg_fn(), names_collision_geoms=robot.names_collision_geoms
  )

  return ManagerBasedRlEnvCfg(
    scene=_scene(robot_cfg, num_envs),
    actions=_actions(
      robot_name,
      robot,
      mc_rtc_yaml,
      residual_scale,
      num_workers,
      console_output,
      print_residual_every,
    ),
    observations=_observations(),
    rewards=_rewards(),
    terminations=_terminations(robot_name),
    events=_events(cart_pose_range, cart_mass_range_kg),
    metrics=_metrics(),
    decimation=DECIMATION,
    episode_length_s=episode_length_s,
    sim=_sim(),
  )


def _scene(robot_cfg: EntityCfg, num_envs: int) -> SceneCfg:
  """The robot, the cart it pushes, and the ground they share."""
  return SceneCfg(
    num_envs=num_envs,
    terrain=TerrainEntityCfg(terrain_type="plane"),
    entities={"robot": robot_cfg, locomanip_mdp.accessors.OBJECT_ENTITY: cart_cfg()},
    # The cart's own directional friction pair needs the scene's ground plane.
    spec_fn=cart_floor_contact,
    sensors=hand_cart_contact_sensors(),
    env_spacing=5.0,
  )


def _actions(
  robot_name: str,
  robot: RobotSpec,
  mc_rtc_yaml: Path,
  residual_scale: float,
  num_workers: int | None,
  console_output: Literal["none", "single", "all"],
  print_residual_every: int,
) -> dict[str, ActionTermCfg]:
  """One position residual on top of the controller's joint targets."""
  return {
    locomanip_mdp.accessors.ACTION_NAME: McRtcResidualJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      residual_actuator_names=robot.get_residual_joints(),
      mc_rtc_config_path=str(mc_rtc_yaml),
      mc_rtc_robot_name=robot_name,
      required_controller=locomanip_mdp.accessors.REQUIRED_CONTROLLER,
      frameskip=FRAMESKIP,
      num_workers=num_workers,
      pd_gains_path=str(robot.pd_gains_path),
      controller_objects={"obj": locomanip_mdp.accessors.OBJECT_ENTITY},
      datastore_vectors_outputs=(
        locomanip_mdp.accessors.OBJECT_REFERENCE_POSITION,
        locomanip_mdp.accessors.OBJECT_REFERENCE_RPY,
        # The ZMP reward compares these two. docs/coupling.md#planned_zmp
        PLANNED_ZMP,
        CONTROL_COM,
      ),
      datastore_scalar_outputs=(
        locomanip_mdp.accessors.LEFT_PHASE,
        locomanip_mdp.accessors.RIGHT_PHASE,
        locomanip_mdp.accessors.COMPLETE,
      ),
      # A float covers every actuator; a partial dict leaves the joints it does
      # not match at scale 1.0 and no clip. docs/residual-authority.md#residual_scales
      scale=residual_scale,
      clip={".*": (-residual_scale, residual_scale)},
      console_output=console_output,
      print_residual_every=print_residual_every,
    )
  }


def _observations() -> dict[str, ObservationGroupCfg]:
  """Proprioception and object state for both, plus what only the critic may see."""
  terms = {
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.observations.executed_action),
    "object_pose": ObservationTermCfg(func=locomanip_mdp.observations.object_pose),
    "object_velocity": ObservationTermCfg(
      func=locomanip_mdp.observations.object_velocity
    ),
    "object_position_error": ObservationTermCfg(
      func=locomanip_mdp.observations.object_position_error
    ),
    "object_yaw_error": ObservationTermCfg(
      func=locomanip_mdp.observations.object_yaw_error
    ),
    "manipulation_phase": ObservationTermCfg(
      func=locomanip_mdp.observations.manipulation_phase
    ),
    "task_complete": ObservationTermCfg(func=locomanip_mdp.observations.task_complete),
    **{
      name: ObservationTermCfg(
        func=envs_mdp.builtin_sensor, params={"sensor_name": f"robot/{name}"}
      )
      for name in FORCE_SENSORS
    },
  }

  # Privileged: exogenous, and the single largest source of return variance here.
  critic_terms = dict(terms) | {
    "object_mass": ObservationTermCfg(func=locomanip_mdp.observations.object_mass),
    "hand_contact_force": ObservationTermCfg(
      func=locomanip_mdp.observations.hand_contact_force
    ),
  }
  return {
    "actor": ObservationGroupCfg(terms=terms, concatenate_terms=True),
    "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True),
  }


def _rewards() -> dict[str, RewardTermCfg]:
  """One task term and the regularizers every residual task carries."""
  return {
    "object_tracking": RewardTermCfg(
      func=locomanip_mdp.rewards.object_position_tracking,
      weight=1.0,
      params={"std": OBJECT_TRACKING_STD},
    ),
    "zmp_tracking": RewardTermCfg(
      func=locomanip_mdp.rewards.zmp_tracking,
      weight=1.0,
      params={**ZMP_PARAMS, "std": ZMP_TRACKING_STD},
    ),
    "termination_penalty": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
    "upright": RewardTermCfg(func=envs_mdp.flat_orientation_l2, weight=-2.0),
    "residual_magnitude": RewardTermCfg(
      func=mdp.rewards.requested_action_l2, weight=-0.1
    ),
    "residual_rate": RewardTermCfg(
      func=mdp.rewards.requested_action_rate_l2, weight=-0.1
    ),
  }


def _terminations(robot_name: str) -> dict[str, TerminationTermCfg]:
  """Falls end the episode; a dead worker only truncates it."""
  nominal_height = mc_rtc.get_default_root_position(robot_name)[2]
  return {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation, params={"limit_angle": FALL_LIMIT_ANGLE}
    ),
    "collapsed": TerminationTermCfg(
      func=mdp.terminations.collapsed,
      params={
        "minimum_height": 0.7 * nominal_height,
        "limit_angle": FALL_LIMIT_ANGLE,
      },
    ),
    "controller_failed": TerminationTermCfg(func=mdp.terminations.controller_failed),
    # Exogenous, so bootstrap instead of charging the fall penalty to the policy.
    # docs/coupling.md#worker-failure-is-a-truncation
    "controller_worker_failed": TerminationTermCfg(
      func=mdp.terminations.controller_worker_failed, time_out=True
    ),
  }


def _events(
  cart_pose_range: dict[str, tuple[float, float]] | None,
  cart_mass_range_kg: tuple[float, float] | None,
) -> dict[str, EventTermCfg]:
  """Reset the scene, and vary the cart whose mass the controller cannot see."""
  # Must come first and must not be dropped: declaring `events` replaces mjlab's
  # default, and this is the only term that resets joints.
  events = {
    "reset_scene_to_default": EventTermCfg(
      func=envs_mdp.reset_scene_to_default, mode="reset"
    )
  }
  if cart_pose_range is not None:
    events["reset_cart"] = EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": cart_pose_range,
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg(locomanip_mdp.accessors.OBJECT_ENTITY),
      },
    )
  if cart_mass_range_kg is not None:
    events["cart_payload"] = EventTermCfg(
      func=dr.pseudo_inertia,
      mode="reset",
      params={
        # Uniform in the log scale is log-uniform in mass, which is how the
        # sweep spaced its samples. docs/locomanip.md#cart_mass_range_kg
        "alpha_range": mass_alpha_range(cart_mass_range_kg),
        "asset_cfg": SceneEntityCfg(
          locomanip_mdp.accessors.OBJECT_ENTITY,
          body_names=(locomanip_mdp.accessors.OBJECT_BODY,),
        ),
      },
    )
  return events


def _metrics() -> dict[str, MetricsTermCfg]:
  """Length-independent readouts; episode reward sums cannot replace these."""
  return {
    "object_position_error": MetricsTermCfg(
      func=locomanip_mdp.metrics.object_position_error
    ),
    # Read tracking quality as `zmp_error / zmp_grounded`, never zmp_error alone.
    "zmp_error": MetricsTermCfg(func=mdp.metrics.zmp_error, params=dict(ZMP_PARAMS)),
    "zmp_grounded": MetricsTermCfg(
      func=mdp.metrics.zmp_grounded, params=dict(ZMP_PARAMS)
    ),
    "object_yaw_error": MetricsTermCfg(func=locomanip_mdp.metrics.object_yaw_error),
    "task_complete": MetricsTermCfg(
      func=locomanip_mdp.metrics.task_complete, reduce="last"
    ),
    "hands_released": MetricsTermCfg(
      func=locomanip_mdp.metrics.hands_released, reduce="last"
    ),
    # Every readout above has to be stratified by this. docs/locomanip.md#cart_mass_range_kg
    "cart_mass": MetricsTermCfg(func=locomanip_mdp.metrics.cart_mass, reduce="last"),
  }


def _sim() -> SimulationCfg:
  """mc_mujoco's solver settings, with the Jacobian the cart forces."""
  return SimulationCfg(
    njmax=1500,
    nconmax=100,
    mujoco=MujocoCfg(
      timestep=0.001,
      integrator="euler",
      solver="newton",
      iterations=50,
      tolerance=1e-10,
      # HRP5P plus the free cart is 65 velocities, over Warp's dense limit of 60.
      jacobian="sparse",
    ),
  )


# 50 Hz is unreadable; every 10th step is 5 Hz. `MC_MJLAB_PRINT_RESIDUAL` retunes.
PLAY_PRINT_RESIDUAL_EVERY = 10
