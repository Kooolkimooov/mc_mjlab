"""Motion exploration independent of the fast joint residual channels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mc_mjlab.rl.motion_actor import MOTION_DECISION_GROUP
from mc_mjlab.tasks.locomanip.locomanip_ppo_cfg import locomanip_ppo_cfg

if TYPE_CHECKING:
  from mjlab.rl import RslRlOnPolicyRunnerCfg


def locomanip_motion_ppo_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
  """Keep shared PPO settings and configure the independently masked motion actor."""
  cfg = locomanip_ppo_cfg(experiment_name=experiment_name)
  cfg.actor.class_name = "mc_mjlab.rl.motion_actor:MotionActor"
  cfg.actor.distribution_cfg = {
    **(cfg.actor.distribution_cfg or {}),
    "class_name": "mc_mjlab.rl.motion_distribution:MotionGaussianDistribution",
    "motion_init_std": 0.8,
    "motion_std_range": (0.5, 1.0),
  }
  cfg.obs_groups = {
    "actor": ("actor", MOTION_DECISION_GROUP),
    "critic": ("critic", MOTION_DECISION_GROUP),
  }
  return cfg
