"""Locomanip's payload task with six hand-force feedback actions."""

from __future__ import annotations

from dataclasses import fields
from typing import Literal, cast

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg

from mc_mjlab.actions.residual_feedback_action import (
  ResidualFeedbackJointPositionActionCfg,
)
from mc_mjlab.tasks.locomanip import mdp
from mc_mjlab.tasks.locomanip.locomanip_residual_env_cfg import (
  CART_MASS_RANGE_KG,
  EPISODE_LENGTH_S,
  NUM_ENVS,
  NUM_WORKERS,
  PLAY_PRINT_RESIDUAL_EVERY,
  make_locomanip_residual_env_cfg,
)
from mc_mjlab.tasks.locomanip.profiles import HRP5P, LocomanipProfile

#: The two hands the controller grasps the cart with, in policy order.
HAND_FORCE_SENSORS = ("LeftHandForceSensor", "RightHandForceSensor")


def locomanip_feedback_env_cfg(
  profile: LocomanipProfile, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Build a registered hand-force feedback task or its viewer variant."""
  cfg = make_locomanip_feedback_env_cfg(
    profile=profile,
    console_output="single" if play else "none",
    print_residual_every=PLAY_PRINT_RESIDUAL_EVERY if play else 0,
  )
  if play:
    cfg.scene.num_envs = 1
    action = cast(
      ResidualFeedbackJointPositionActionCfg, cfg.actions[mdp.accessors.ACTION_NAME]
    )
    action.num_workers = None
    cfg.observations["actor"].enable_corruption = False
  return cfg


def make_locomanip_feedback_env_cfg(
  *,
  profile: LocomanipProfile = HRP5P,
  num_envs: int = NUM_ENVS,
  num_workers: int = NUM_WORKERS,
  episode_length_s: float = EPISODE_LENGTH_S,
  force_scale: float = 50.0,
  cart_pose_range: dict[str, tuple[float, float]] | None = None,
  cart_mass_range_kg: tuple[float, float] | None = CART_MASS_RANGE_KG,
  console_output: Literal["none", "single", "all"] = "none",
  print_residual_every: int = 0,
) -> ManagerBasedRlEnvCfg:
  """Replace only the residual interface and add feedback diagnostics."""
  cfg = make_locomanip_residual_env_cfg(
    profile=profile,
    num_envs=num_envs,
    num_workers=num_workers,
    episode_length_s=episode_length_s,
    cart_pose_range=cart_pose_range,
    cart_mass_range_kg=cart_mass_range_kg,
    console_output=console_output,
    print_residual_every=print_residual_every,
  )
  base = cfg.actions[mdp.accessors.ACTION_NAME]
  carried = {f.name: getattr(base, f.name) for f in fields(base)}
  carried.update(residual_actuator_names=(), scale=1.0, offset=0.0, clip=None)
  cfg.actions[mdp.accessors.ACTION_NAME] = ResidualFeedbackJointPositionActionCfg(
    **carried,
    feedback_modalities=("wrench",),
    wrench_sensor_names=HAND_FORCE_SENSORS,
    wrench_force_only=True,
    wrench_force_scale=force_scale,
    gate_scalar_outputs=(mdp.accessors.LEFT_PHASE, mdp.accessors.RIGHT_PHASE),
    gate_value=mdp.accessors.HOLD_PHASE,
  )
  for group in cfg.observations.values():
    group.terms["actions"] = ObservationTermCfg(func=mdp.feedback.executed_feedback)
  cfg.metrics.update(
    feedback_active=MetricsTermCfg(func=mdp.feedback.feedback_active),
    feedback_force_rms=MetricsTermCfg(func=mdp.feedback.feedback_force_rms),
    feedback_saturation=MetricsTermCfg(func=mdp.feedback.feedback_saturation),
  )
  return cfg
