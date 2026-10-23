"""Locomanip joint residual with one additional slowdown and hold action."""

from __future__ import annotations

from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from mc_mjlab.actions.locomanip_motion_action import LocomanipMotionActionCfg
from mc_mjlab.rl.motion_actor import MOTION_DECISION_GROUP
from mc_mjlab.tasks.locomanip.locomanip_residual_env_cfg import (
  REQUEST_WEIGHT,
  locomanip_residual_env_cfg,
)
from mc_mjlab.tasks.locomanip.mdp import accessors
from mc_mjlab.tasks.locomanip.profiles import HRP5P, LocomanipProfile
from mc_mjlab.tasks.locomanip_motion import mdp

MOTION_EPISODE_SCALE = 2.0
BRAKING_COST = 0.03
BRAKING_START_COST = 0.22
HOLDING_COST = 0.10
DCM_REWARD_WEIGHT = 0.25
FAILURE_COST = 400.0


def locomanip_motion_env_cfg(
  profile: LocomanipProfile = HRP5P, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Build joint motion authority with time for a slower route to finish."""
  # Its rewards were retuned against this price. docs/locomanip-motion.md#DCM_REWARD_WEIGHT
  cfg = locomanip_residual_env_cfg(profile, play=play, request_weight=REQUEST_WEIGHT)
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
  cfg.observations[MOTION_DECISION_GROUP] = ObservationGroupCfg(
    terms={"state": ObservationTermCfg(func=mdp.motion_decision_state)},
    enable_corruption=False,
  )
  cfg.rewards["contact_dcm_stability"].weight = DCM_REWARD_WEIGHT
  cfg.rewards["termination_penalty"].weight = -FAILURE_COST
  cfg.rewards["braking"] = RewardTermCfg(func=mdp.braking, weight=-BRAKING_COST)
  cfg.rewards["braking_start"] = RewardTermCfg(
    func=mdp.braking_start, weight=-BRAKING_START_COST
  )
  cfg.rewards["holding"] = RewardTermCfg(func=mdp.holding, weight=-HOLDING_COST)
  cfg.metrics.update(
    motion_scale=MetricsTermCfg(func=mdp.motion_scale),
    motion_holding=MetricsTermCfg(func=mdp.motion_holding),
    braking_starts=MetricsTermCfg(func=mdp.braking_count, reduce="last"),
    motion_decisions=MetricsTermCfg(func=mdp.motion_decisions, reduce="last"),
  )
  return cfg
