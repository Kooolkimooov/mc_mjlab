"""Zero-residual task: the mc_rtc controller driving the robot on its own.

The env the demo used to build inline. There is no reward and no termination
here on purpose -- nothing is being learned, the point is to watch the
controller hold the robot up with the RL residual left at zero. mjlab's ``play
--agent zero`` supplies that zero action.

This is a *play* task, and ``train`` on it is not supported. With no reward
term there is nothing to optimise, and the env is built to be driven by a zero
action: ``decimation=2`` means a policy would emit a fresh sample every 2 ms,
across all controlled joints, with no termination to reset a robot it wrecks.
Under random actions the contact set then grows past ``njmax`` and MuJoCo-Warp
faults inside ``sim.forward()``. Train the residual balance task instead.

The observation group and ``episode_length_s`` below are not for learning
either -- they keep the env well-formed (rsl_rl rejects an empty observation
set, and a zero ``max_episode_length`` cannot be sampled).

Rates: the sim runs at 1 kHz and the controller at 500 Hz (``frameskip=2``, the
mc_mujoco pairing), with ``decimation=2`` so one env step is one controller
period.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg

from mc_mjlab import MC_RTC_YAML_PATH
from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionActionCfg,
)
from mc_mjlab.actions.mc_rtc_residual_joint_torque_actions import (
  McRtcResidualJointTorqueActionCfg,
)
from mc_mjlab.robots.robots_registry import (
  get_main_robot_spec,
  prepare_cfg_for_mc_rtc,
)

# Every env is its own mc_rtc controller (~70 MB, ~570 ms to construct, built
# serially), so the env count is memory- and startup-bound rather than GPU-bound.
# Two is what a viewer session wants; raise it to sweep throughput.
PLAY_NUM_ENVS = 1
NUM_ENVS = 420

# An hour of sim time: effectively unbounded for a demo (there is no
# termination term either, so a run ends when you stop it) while staying a sane
# step count. Zero is not equivalent -- it leaves ``max_episode_length`` at 0,
# which rsl_rl cannot sample against -- and a huge sentinel like 1e10 is worse:
# it overflows the episode-length buffer's dtype and corrupts GPU memory.
EPISODE_LENGTH_S = 3600.0


def _make_env_cfg(
  control: str,
  num_envs: int = NUM_ENVS,
  num_workers: int | None = None,
  console_output: Literal["none", "single", "all"] = "none",
  mc_rtc_yaml: Path = MC_RTC_YAML_PATH,
) -> ManagerBasedRlEnvCfg:
  """Build the zero-residual env cfg for the config's ``MainRobot``."""
  robot_name, robot = get_main_robot_spec(mc_rtc_yaml)
  robot_cfg = prepare_cfg_for_mc_rtc(
    robot.cfg_fn(), names_collision_geoms=robot.names_collision_geoms
  )

  action_cls = (
    McRtcResidualJointPositionActionCfg
    if control == "position"
    else McRtcResidualJointTorqueActionCfg
  )
  actions: dict[str, ActionTermCfg] = {
    "robot_joints": action_cls(
      entity_name="robot",
      actuator_names=(".*",),
      residual_actuator_names=robot.get_residual_joints(),
      mc_rtc_config_path=str(mc_rtc_yaml),
      mc_rtc_robot_name=robot_name,
      frameskip=2,
      num_workers=num_workers,
      pd_gains_path=str(robot.pd_gains_path),
      console_output=console_output,
    )
  }

  # Nothing reads these -- the residual is zero and there is no reward. They
  # exist because an env with no observation group is not a well-formed RL env:
  # rsl_rl refuses to build an algorithm against an empty set, so `train` would
  # die on this task after paying the full controller-construction cost. Cheap
  # insurance, and they make the robot's state visible while playing.
  terms = {
    "base_lin_vel": ObservationTermCfg(func=envs_mdp.base_lin_vel),
    "base_ang_vel": ObservationTermCfg(func=envs_mdp.base_ang_vel),
    "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
  }
  observations = {
    "actor": ObservationGroupCfg(terms=dict(terms), concatenate_terms=True),
    "critic": ObservationGroupCfg(terms=dict(terms), concatenate_terms=True),
  }

  return ManagerBasedRlEnvCfg(
    # A terrain is required for a ground plane; it also spreads the env origins.
    scene=SceneCfg(
      num_envs=num_envs,
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot_cfg},
    ),
    observations=observations,
    actions=actions,
    decimation=2,
    episode_length_s=EPISODE_LENGTH_S,
    # Solver/integrator settings from mc_mujoco's HRP5Pmain.xml.
    sim=SimulationCfg(
      # A standing robot needs few constraint rows, so the default heuristic
      # (sized off exactly this stance) overflows the moment one sprawls -- and
      # `njmax` is a hard per-world cap: past it MuJoCo writes out of bounds
      # rather than failing cleanly, which surfaces as an async CUDA illegal
      # access. The controller can lose the robot here too, so budget for it.
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


def zero_residual_position_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """mc_rtc's joint *position* targets, with no residual on top."""
  cfg = _make_env_cfg(control="position")
  if play:
    cfg.scene.num_envs = PLAY_NUM_ENVS
  return cfg


def zero_residual_torque_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """mc_rtc's joint *torques*, with no residual on top."""
  cfg = _make_env_cfg(control="torque")
  if play:
    cfg.scene.num_envs = PLAY_NUM_ENVS
  return cfg


def zero_residual_rl_cfg() -> RslRlOnPolicyRunnerCfg:
  """A placeholder runner cfg: registration wants one, no policy is ever built."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(32,),
      activation="elu",
      # Without a distribution the actor cannot report a log-prob and the
      # runner dies on its first step; a placeholder still has to be valid.
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.2,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(hidden_dims=(32,), activation="elu"),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.0,
      num_learning_epochs=1,
      num_mini_batches=1,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="mc_rtc_zero_residual",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=1,
    # Local only: a demo that is not learning anything has no business opening
    # a W&B run, which the default would do.
    logger="tensorboard",
  )
