"""HRP5P hand-feedback safety task at the measured cart-mass transition."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from mc_mjlab.tasks.locomanip.cart import cart_nominal_mass_kg
from mc_mjlab.tasks.locomanip.locomanip_feedback_env_cfg import (
  locomanip_feedback_env_cfg,
)
from mc_mjlab.tasks.locomanip.mdp import payload_mixture
from mc_mjlab.tasks.locomanip.profiles import HRP5P

SAFETY_CONFIG = HRP5P.mc_rtc_yaml
PASSING_MASS_KG = 525.0
FAILING_MASS_KG = 550.0
TRANSITION_MIDPOINT_KG = (PASSING_MASS_KG + FAILING_MASS_KG) / 2
MAX_ITERATIONS = 1000


def locomanip_safety_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Build the frozen safety task or its one-environment viewer variant."""
  cfg = locomanip_feedback_env_cfg(profile=HRP5P, play=play)
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
  return cfg
