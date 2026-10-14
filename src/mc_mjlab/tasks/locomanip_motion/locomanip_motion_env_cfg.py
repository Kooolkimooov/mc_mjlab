"""Locomanip joint residual with one additional slowdown and hold action."""

from __future__ import annotations

from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg

from mc_mjlab.actions.locomanip_motion_action import LocomanipMotionActionCfg
from mc_mjlab.tasks.locomanip.locomanip_residual_env_cfg import (
  locomanip_residual_env_cfg,
)
from mc_mjlab.tasks.locomanip.mdp import accessors
from mc_mjlab.tasks.locomanip.profiles import HRP5P, LocomanipProfile
from mc_mjlab.tasks.locomanip_motion import mdp

MOTION_EPISODE_SCALE = 2.0


def locomanip_motion_env_cfg(
  profile: LocomanipProfile = HRP5P, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Build joint motion authority with time for a slower route to finish."""
  cfg = locomanip_residual_env_cfg(profile, play=play)
  cfg.episode_length_s *= MOTION_EPISODE_SCALE
  base = cfg.actions[accessors.ACTION_NAME]
  carried = {field.name: getattr(base, field.name) for field in fields(base)}
  cfg.actions[accessors.ACTION_NAME] = LocomanipMotionActionCfg(
    **carried,
    motion_phase_getters=(accessors.LEFT_PHASE, accessors.RIGHT_PHASE),
    motion_hold_phase=accessors.HOLD_PHASE,
  )
  for group in cfg.observations.values():
    group.terms["motion_control"] = ObservationTermCfg(func=mdp.motion_observation)
  cfg.metrics.update(
    motion_scale=MetricsTermCfg(func=mdp.motion_scale),
    motion_holding=MetricsTermCfg(func=mdp.motion_holding),
  )
  return cfg
