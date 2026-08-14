"""Zero-residual task: the mc_rtc controller driving the robot on its own."""

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

# An hour of sim time: effectively unbounded for a demo.
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

  # Unused: the residual is zero and nothing learns, but the managers require
  # at least one term each.
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
      # Budget for a sprawled robot: `njmax` is a hard cap, and past it MuJoCo
      # drops rows and simulates the fall wrong rather than failing.
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
  """Retune a cfg for a viewer session."""
  cfg.scene.num_envs = PLAY_NUM_ENVS
  return cfg


# A viewer session is one controller, so silencing it hides the only thing worth
# watching; "single" rather than "all" keeps env 0 readable if `--num-envs` grows.
PLAY_CONSOLE_OUTPUT = "single"


def zero_residual_position_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """mc_rtc's joint *position* targets, with no residual on top."""
  cfg = _make_env_cfg(
    control="position", console_output=PLAY_CONSOLE_OUTPUT if play else "none"
  )
  if play:
    _apply_play_overrides(cfg)
  return cfg


def zero_residual_torque_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """mc_rtc's joint *torques*, with no residual on top."""
  cfg = _make_env_cfg(
    control="torque", console_output=PLAY_CONSOLE_OUTPUT if play else "none"
  )
  if play:
    _apply_play_overrides(cfg)
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
