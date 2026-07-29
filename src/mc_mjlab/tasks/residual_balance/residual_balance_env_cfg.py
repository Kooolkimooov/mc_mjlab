"""Residual balance task: keep the mc_rtc controller walking under pushes.

The mc_rtc controller is the base policy and is *not* commanded by the RL side
-- this coupling feeds it state, not velocity references -- so the learnable
part is a residual that keeps reality close to the controller's plan when the
world disagrees with the controller's model: pushes, friction and CoM
randomization, reset noise.

Two things this task has to get right, both learned the hard way from a run
whose robot never walked at all:

*The env must hand mc_rtc a robot it recognizes.* Declaring an ``events`` dict
replaces mjlab's default, which is where ``reset_scene_to_default`` -- the only
term that resets joints -- lives. Without it every episode starts from the
model's qpos0 (legs straight) instead of the half-sitting stance, mc_rtc
initializes against that posture and its FSM never reaches the walking state.
The robot then stands still for the whole episode no matter what the residual
does, which reads exactly like a policy that has learned to freeze the gait.

*The reward must pay for walking, not merely for surviving.* Walking is the
risky activity, so an alive-only reward prefers a policy that stops the gait.
Payment is therefore mostly for the controller still generating a gait and the
robot still covering ground -- measured on both sides, since the controller's
intent and the robot's motion can be frozen independently -- and the residual
is hard-clipped to an authority that cannot cancel a swing trajectory.

Rates: the sim runs at 1 kHz and the controller at 500 Hz (``frameskip=2``, the
mc_mujoco pairing), while the policy acts at 50 Hz (``decimation=20``); the
residual is therefore held across 10 controller periods.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
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

# How hard the robot is shoved: the task's difficulty dial. Both directions
# ruin training -- too gentle and mc_rtc never falls, so the best residual is
# no residual; too hard and the robot falls whatever the residual does, which
# plateaus the policy at a fraction of an episode.
#
# 0.1 m/s reads timid next to mjlab's velocity task (+/-0.5), but a *walking*
# robot is far easier to topple than a standing one: measured over 65 s x 16
# envs, the zero-residual baseline already loses about half its episodes here
# (fell_over + collapsed vs time_out). That is the headroom the residual is
# meant to recover, so re-measure before raising this -- the number to keep an
# eye on is the baseline's survival rate, not the push magnitude. They live
# here rather than as CLI flags because they sit inside an event term's
# ``velocity_range`` dict, which tyro does not flatten.
PUSH_VELOCITY = 0.1
PUSH_ANGULAR_VELOCITY = 0.0

# How long the base controller actually walks, and therefore how long an
# episode is worth running. With the *installed* LogisticController_ismpc
# config the FSM settles the posture for ~4 s, walks 1 m in ~12 s and then
# stands still for good (its remaining transitions are commented out), so
# everything past ~16 s would train the residual on a stationary robot -- the
# opposite of this task. Enabling the endless-walk override shipped in
# `etc/mc_rtc_controllers/` removes that ceiling; raise this with
# `--env.episode-length-s` when you do. Measured with a zero residual.
WALK_WINDOW_S = 16.0

# A viewer default, not a training one: every env is its own mc_rtc controller
# (~70 MB, ~570 ms to construct, built serially), so replaying at the training
# env count would spend minutes and gigabytes before the first frame.
PLAY_NUM_ENVS = 1


def _make_env_cfg(
  control: str,
  num_envs: int = 128,
  num_workers: int | None = None,
  residual_scale: float | None = None,
  episode_length_s: float = WALK_WINDOW_S,
  push_velocity: float = PUSH_VELOCITY,
  push_angular_velocity: float = PUSH_ANGULAR_VELOCITY,
  console_output: Literal["none", "single", "all"] = "none",
  mc_rtc_yaml: Path = MC_RTC_YAML_PATH,
) -> ManagerBasedRlEnvCfg:
  """Build the residual balance env cfg for the config's ``MainRobot``."""
  robot_name, robot = get_main_robot_spec(mc_rtc_yaml)
  robot_cfg = prepare_cfg_for_mc_rtc(
    robot.cfg_fn(), names_collision_geoms=robot.names_collision_geoms
  )
  # The stance height the reward aims at, taken from the controller's own
  # default attitude rather than a constant here: it is the height mc_rtc holds,
  # so the reward target and the base policy cannot drift apart.
  nominal_height = mc_rtc.get_default_root_position(robot_name)[2]

  # The residual acts on the legs only: balancing is what this task rewards, and
  # the upper body does not hold the robot up. This is the task's opinion, so it
  # lives here -- a locomotion or manipulation task would want the arms, and the
  # robot's own `get_residual_joints` stays the place for exclusions that hold
  # whatever the task is (a joint mc_rtc models as fixed can carry no residual
  # anywhere, and is dropped there). Filtering that set rather than taking the
  # legs directly keeps the robot's carve-outs and refJointOrder ordering.
  upper_body = set(mc_rtc.get_upper_body_joints(robot_name))
  residual_joints = tuple(j for j in robot.get_residual_joints() if j not in upper_body)

  ##
  # Actions: the residual on top of mc_rtc.
  ##

  # Residual authority, in the control channel's own unit (rad for position,
  # Nm for torque -- one number cannot serve both). ``scale`` maps the policy's
  # ~unit output into that authority and ``clip`` makes it a hard bound: a
  # residual able to outvote the controller is how the policy learns to freeze
  # the gait instead of stabilizing it (a swing trajectory is ~0.5 rad;
  # rejecting a push needs far less).
  if residual_scale is None:
    residual_scale = 0.1 if control == "position" else 10.0

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
      pd_gains_path=str(robot.pd_gains_path),
      scale=residual_scale,
      clip={".*": (-residual_scale, residual_scale)},
      console_output=console_output,
    )
  }

  ##
  # Observations.
  ##

  actor_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
    ),
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5)
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
    # What the controller wants vs. reality, and where its gait is headed:
    # without these the policy cannot phase its residual with the plan it is
    # meant to protect. The error sees encoder-level noise; the reference
    # velocity is controller-internal and known exactly.
    "controller_error": ObservationTermCfg(
      func=mdp.controller_position_error, noise=Unoise(n_min=-0.01, n_max=0.01)
    ),
    "controller_ref_vel": ObservationTermCfg(func=mdp.controller_reference_velocity),
  }

  # The critic sees the same signals without observation noise.
  critic_terms = {
    name: ObservationTermCfg(func=term.func, params=dict(term.params))
    for name, term in actor_terms.items()
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True, enable_corruption=True
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  ##
  # Rewards. Weights are per second: the manager scales them by step_dt.
  #
  # Structured so that full payment requires walking: `alive` alone pays less
  # than `tracking` + `plan_motion`, both of which a frozen robot (or a
  # controller disturbed into standing) forfeits. The residual penalties only
  # break ties toward minimal intervention; the action term's hard clip is
  # what actually bounds the residual.
  ##

  rewards = {
    "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=1.0),
    # The two halves of "it is still walking", and together the bulk of the
    # payment: the controller is still generating a gait, and the robot is
    # really covering ground. A policy that stops either one keeps only
    # `alive`, which on its own is worth less than what it gave up.
    #
    # `scale`/`speed` are set from the zero-residual baseline so both read as
    # switches rather than gradients: its reference velocity sits at 0.43 rad/s
    # median while walking against 0.003 rad/s standing, and it travels at
    # ~0.09 m/s. Saturating there pays for walking at all, not for shaking the
    # controller harder or outrunning its gait.
    "plan_motion": RewardTermCfg(
      func=mdp.controller_reference_motion, weight=1.5, params={"scale": 0.5}
    ),
    "progress": RewardTermCfg(
      func=mdp.base_progress_tanh, weight=1.5, params={"speed": 0.1}
    ),
    # There is deliberately no root-height term next to this one: the
    # controller dips to z~0.75 while walking (stance is 0.79), so a fixed
    # height target would pay the policy to stand tall -- i.e. to stop the gait.
    "upright": RewardTermCfg(func=envs_mdp.flat_orientation_l2, weight=-2.0),
    "residual_magnitude": RewardTermCfg(func=mdp.action_l2, weight=-0.1),
    "residual_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.1),
  }

  ##
  # Terminations.
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(45.0)}
    ),
    # Crouch-collapse keeps the trunk upright, so `fell_over` misses it. This
    # is the termination that actually fires most (27 of 37 baseline failures
    # over 65 s x 16 envs); walking dips to z~0.75, so the threshold has room.
    "collapsed": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.7 * nominal_height},
    ),
    "controller_failed": TerminationTermCfg(
      func=mdp.controller_failed, params={"action_name": "mc_rtc_residual"}
    ),
  }

  ##
  # Events: the disturbances the residual has to earn its keep against.
  ##

  events = {
    # Must come first, and must not be dropped: this is mjlab's default reset
    # event, and defining an `events` dict at all replaces that default. It is
    # the only term that resets *joints*; without it every episode starts from
    # the model's qpos0 (legs straight, z~0.86) rather than the controller's
    # half-sitting stance, mc_rtc initializes against that posture, and its FSM
    # never reaches the walking state -- the robot stands still all episode
    # with no residual able to change it. `reset_base` below then offsets from
    # the default root state this writes, so the order matters too.
    "reset_scene_to_default": EventTermCfg(
      func=envs_mdp.reset_scene_to_default, mode="reset"
    ),
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        # No z offset: dropping the robot injects a transient the controller
        # has to absorb before the episode even starts.
        "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-math.pi, math.pi)},
        "velocity_range": {},
      },
    ),
    "push_robot": EventTermCfg(
      func=envs_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(2.0, 5.0),
      params={
        "velocity_range": {
          "x": (-push_velocity, push_velocity),
          "y": (-push_velocity, push_velocity),
          "roll": (-push_angular_velocity, push_angular_velocity),
          "pitch": (-push_angular_velocity, push_angular_velocity),
        }
      },
    ),
    "encoder_bias": EventTermCfg(
      func=dr.encoder_bias,
      mode="startup",
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
    ),
  }

  ##
  # Assemble. Solver settings follow mc_mujoco's HRP5Pmain.xml, as in the demo.
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=num_envs,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot_cfg},
    ),
    observations=observations,
    actions=actions,
    rewards=rewards,
    terminations=terminations,
    events=events,
    decimation=20,
    episode_length_s=episode_length_s,
    sim=SimulationCfg(
      # A standing robot needs a handful of constraint rows, so the default
      # heuristic (sized off the demo's stance) overflows the moment one falls
      # -- and `njmax` is a hard per-world cap: past it MuJoCo drops rows and
      # simulates the fall wrong rather than failing. Falls are most of this
      # task, so budget for a sprawled robot; these match mjlab's own humanoid
      # locomotion task.
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
  cfg.scene.num_envs = PLAY_NUM_ENVS
  cfg.episode_length_s = 1e10  # only a fall should end an episode
  cfg.observations["actor"].enable_corruption = False
  return cfg


# A viewer session is one controller, so silencing it hides the only thing worth
# watching; "single" rather than "all" keeps env 0 readable if `--num-envs` grows.
PLAY_CONSOLE_OUTPUT = "single"


def residual_balance_position_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Residual on the controller's joint *position* targets."""
  cfg = _make_env_cfg(
    control="position", console_output=PLAY_CONSOLE_OUTPUT if play else "none"
  )
  if play:
    _apply_play_overrides(cfg)
  return cfg


def residual_balance_torque_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Residual on the controller's joint *torques*."""
  cfg = _make_env_cfg(
    control="torque", console_output=PLAY_CONSOLE_OUTPUT if play else "none"
  )
  if play:
    _apply_play_overrides(cfg)
  return cfg


# RL config.


def residual_balance_ppo_cfg(
  max_iterations: int = 500, experiment_name: str = "mc_rtc_residual_balance"
) -> RslRlOnPolicyRunnerCfg:
  """PPO settings, following mjlab's locomotion configs."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        # The residual should start near zero: a unit init_std would hand the
        # controller a huge random offset on step one and knock it over.
        "init_std": 0.2,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=max_iterations,
    logger="wandb",
  )
