"""Locomanip safety using only controller-side braking authority."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

from mjlab.envs import ManagerBasedRlEnvCfg

from mc_mjlab.actions.locomanip_motion_action import LocomanipMotionActionCfg
from mc_mjlab.tasks.locomanip.mdp import accessors
from mc_mjlab.tasks.locomanip.profiles import HRP5P, LocomanipProfile
from mc_mjlab.tasks.locomanip_motion.locomanip_motion_env_cfg import (
  locomanip_motion_env_cfg,
)


def locomanip_braking_env_cfg(
  profile: LocomanipProfile = HRP5P, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Keep safety and braking costs while every joint tracks the base controller."""
  cfg = locomanip_motion_env_cfg(profile, play=play)
  action = cast(LocomanipMotionActionCfg, cfg.actions[accessors.ACTION_NAME])
  cfg.actions[accessors.ACTION_NAME] = replace(
    action,
    residual_actuator_names=(),
    scale=1.0,
    offset=0.0,
    clip=None,
    print_residual_every=0,
  )
  for name in ("residual_magnitude", "residual_rate"):
    del cfg.rewards[name]
  for group in cfg.observations.values():
    group.terms.pop("actions", None)
  return cfg
