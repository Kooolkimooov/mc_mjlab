"""ResidualMPC curricula."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.manager_base import ManagerTermBaseCfg


from mc_mjlab.tasks.residual_mpc.mdp.disturbances import initial_velocity_kick


class survival_kick_curriculum:
  """Move the kick's difficulty on smoothed survival, as Ranjbar's curriculum does."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    kick = env.event_manager.get_term_cfg(
      cfg.params.get("term_name", "initial_base_velocity")
    ).func
    if not isinstance(kick, initial_velocity_kick):
      raise TypeError("the survival curriculum needs an initial_velocity_kick term")
    self._kick = kick
    self.smoothed = float("nan")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    advance_fraction: float = 0.7,
    regress_fraction: float = 0.6,
    increment: float = 0.05,
    scale_range: tuple[float, float] = (0.4, 2.0),
    smoothing: float = 0.02,
    term_name: str = "initial_base_velocity",
  ) -> dict[str, float]:
    """Raise the kick above the advance rate, lower it below the regress rate."""
    del term_name
    if advance_fraction <= regress_fraction:
      raise ValueError("the advance fraction must leave a deadband above regress")
    # `curriculum_manager.compute` runs first in `_reset_idx`, so the termination
    # buffers still describe exactly the envs that just finished.
    if env_ids is not None and not isinstance(env_ids, slice):
      survived = env.termination_manager.get_term("time_out")[env_ids]
      if survived.numel():
        sample = float(survived.to(torch.float32).mean())
        self.smoothed = (
          sample
          if math.isnan(self.smoothed)
          else (1.0 - smoothing) * self.smoothed + smoothing * sample
        )
        if self.smoothed > advance_fraction:
          self._kick.scale = min(scale_range[1], self._kick.scale + increment)
        elif self.smoothed < regress_fraction:
          self._kick.scale = max(scale_range[0], self._kick.scale - increment)
    return {"kick_scale": self._kick.scale, "survival": self.smoothed}
