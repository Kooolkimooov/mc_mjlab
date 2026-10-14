"""Read motion decision eligibility from the same observations PPO stores."""

from __future__ import annotations

from typing import Any, cast

import torch

from mc_mjlab.rl.motion_distribution import MotionGaussianDistribution
from mc_mjlab.rl.zero_init_actor import ZeroInitMLPModel

MOTION_DECISION_GROUP = "motion_decision"


class MotionActor(ZeroInitMLPModel):
  """Apply a stateless decision mask during rollout and shuffled PPO updates."""

  def get_latent(
    self, obs: Any, masks: torch.Tensor | None = None, hidden_state: Any = None
  ) -> torch.Tensor:
    """Read raw eligibility before normalization and build the ordinary MLP input."""
    distribution = cast(MotionGaussianDistribution, self.distribution)
    distribution.set_motion_active(obs[MOTION_DECISION_GROUP][..., 0])
    return super().get_latent(obs, masks, hidden_state)
