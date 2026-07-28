"""Residual balance task: keep the mc_rtc controller upright under pushes.

The mc_rtc controller is the base policy and is *not* commanded by the RL side
-- this coupling feeds it state, not velocity references -- so the learnable
part is a residual that keeps the robot standing when the world disagrees with
the controller's model: pushes, friction and CoM randomization, reset noise.
The reward pays for staying alive, upright and at the controller's stance
height, and charges for the residual itself, so the policy is pushed to depart
from mc_rtc only where it must.

Rates: the sim runs at 1 kHz and the controller at 500 Hz (``frameskip=2``, the
mc_mujoco pairing), while the policy acts at 50 Hz (``decimation=20``); the
residual is therefore held across 10 controller periods.
"""

from __future__ import annotations

import math
from pathlib import Path

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
# plateaus the policy at a fraction of an episode. These match mjlab's own
# velocity task (+/-0.5 m/s, +/-0.5 rad/s): treat them as the calibrated
# starting point and raise them only as far as the baseline still survives most
# episodes. They live here rather than as CLI flags because they sit inside an
# event term's ``velocity_range`` dict, which tyro does not flatten.
PUSH_VELOCITY = 0.5
PUSH_ANGULAR_VELOCITY = 0.5

# A viewer default, not a training one: every env is its own mc_rtc controller
# (~70 MB, ~570 ms to construct, built serially), so replaying at the training
# env count would spend minutes and gigabytes before the first frame.
PLAY_NUM_ENVS = 1


def _make_env_cfg(
  control: str,
  num_envs: int = 128,
  num_workers: int | None = None,
  residual_scale: float = 0.1,
  episode_length_s: float = 10.0,
  push_velocity: float = PUSH_VELOCITY,
  push_angular_velocity: float = PUSH_ANGULAR_VELOCITY,
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
  ##

  rewards = {
    "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=2.0),
    "upright": RewardTermCfg(func=envs_mdp.flat_orientation_l2, weight=-2.0),
    "stance_height": RewardTermCfg(
      func=mdp.root_height_l2,
      weight=-20.0,
      params={"target_height": nominal_height},
    ),
    # Keep the policy near mc_rtc: pay for the residual and for jerk in it.
    "residual_magnitude": RewardTermCfg(func=mdp.action_l2, weight=-0.01),
    "residual_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.005),
  }

  ##
  # Terminations.
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation, params={"limit_angle": math.radians(60.0)}
    ),
    "dropped": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": nominal_height - 0.25},
    ),
    "controller_failed": TerminationTermCfg(
      func=mdp.controller_failed, params={"action_name": "mc_rtc_residual"}
    ),
  }

  ##
  # Events: the disturbances the residual has to earn its keep against.
  ##

  events = {
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


def residual_balance_position_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Residual on the controller's joint *position* targets."""
  cfg = _make_env_cfg(control="position")
  if play:
    _apply_play_overrides(cfg)
  return cfg


def residual_balance_torque_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Residual on the controller's joint *torques*."""
  cfg = _make_env_cfg(control="torque")
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
