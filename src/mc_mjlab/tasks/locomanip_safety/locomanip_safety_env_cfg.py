"""HRP5P hand-feedback safety task at the measured cart-mass transition."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from mc_mjlab import MC_RTC_CONFIG_PATH
from mc_mjlab.actions.residual_feedback_action import (
  ResidualFeedbackJointPositionActionCfg,
)
from mc_mjlab.tasks.locomanip.cart import cart_nominal_mass_kg
from mc_mjlab.tasks.locomanip.locomanip_feedback_env_cfg import (
  make_locomanip_feedback_env_cfg,
)
from mc_mjlab.tasks.locomanip.locomanip_residual_env_cfg import (
  PLAY_PRINT_RESIDUAL_EVERY,
  add_cart_mass_printout,
)
from mc_mjlab.tasks.locomanip.mdp import accessors, payload_mixture
from mc_mjlab.tasks.locomanip.profiles import HRP5P

SAFETY_CONFIG = MC_RTC_CONFIG_PATH / "mc_rtc_hrp5_locomanip_safety.yaml"
TRANSITION_MIDPOINT_KG = 537.5
MAX_ITERATIONS = 1000


def locomanip_safety_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the frozen safety task or its one-environment viewer variant."""
  profile = replace(HRP5P, mc_rtc_yaml=Path(SAFETY_CONFIG))
  cfg = make_locomanip_feedback_env_cfg(
    profile=profile,
    console_output="single" if play else "none",
    print_residual_every=PLAY_PRINT_RESIDUAL_EVERY if play else 0,
  )
  cfg.rewards.pop("contact_dcm_stability")
  cfg.rewards["is_alive"] = RewardTermCfg(func=envs_mdp.is_alive, weight=1.0)
  payload = cfg.events["cart_payload"]
  cfg.events["cart_payload"] = EventTermCfg(
    func=payload_mixture.transition_payload,
    mode="reset",
    params={
      "midpoint_kg": TRANSITION_MIDPOINT_KG,
      "nominal_mass_kg": cart_nominal_mass_kg(),
      "asset_cfg": payload.params["asset_cfg"],
    },
  )
  if play:
    cfg.scene.num_envs = 1
    action = cast(
      ResidualFeedbackJointPositionActionCfg, cfg.actions[accessors.ACTION_NAME]
    )
    action.num_workers = None
    cfg.observations["actor"].enable_corruption = False
    add_cart_mass_printout(cfg)
  return cfg
