"""Curricula that move the disturbance mixture on measured training progress."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mc_mjlab.mdp.disturbances import (
  _push_term,
  achievement_finite_impulse_curriculum,
  stratified_finite_impulse_curriculum,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.manager_base import ManagerTermBaseCfg


class episode_length_impulse_curriculum:
  """Move a stratified impulse mixture on smoothed terminal episode length."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    push = _push_term(env, cfg.params.get("term_name", "push_robot"))
    if not isinstance(push, stratified_finite_impulse_curriculum):
      raise TypeError("episode-length curriculum needs a stratified impulse term")

    self._push: stratified_finite_impulse_curriculum = push
    self._mixtures = tuple(
      push.validated_weights(mixture) for mixture in cfg.params["mixtures"]
    )
    if len(self._mixtures) < 2:
      raise ValueError("an episode-length ladder needs at least two mixtures")

    self.stage = 0
    self.smoothed = float("nan")
    self._push.set_band_weights(self._mixtures[0])

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    mixtures: tuple[tuple[float, ...], ...],
    advance_fraction: float = 0.65,
    regress_fraction: float = 0.45,
    smoothing: float = 0.02,
    term_name: str = "push_robot",
  ) -> dict[str, float]:
    """Smooth the terminating cohort's length and step the ladder across it."""
    del mixtures, term_name
    if advance_fraction <= regress_fraction:
      raise ValueError("the advance fraction must leave a deadband above regress")

    # `curriculum_manager.compute` runs first in `_reset_idx`, so these buffers
    # still hold the terminal lengths of exactly the envs that just finished.
    if env_ids is not None and not isinstance(env_ids, slice):
      lengths = env.episode_length_buf[env_ids]
      if lengths.numel():
        sample = float(lengths.to(torch.float32).mean()) / env.max_episode_length
        self.smoothed = (
          sample
          if math.isnan(self.smoothed)
          else (1.0 - smoothing) * self.smoothed + smoothing * sample
        )
        self._step_ladder(advance_fraction, regress_fraction)

    return {
      "stage": float(self.stage),
      "smoothed_length_fraction": self.smoothed,
      "top_band_share": self._mixtures[self.stage][-1],
    }

  def _step_ladder(self, advance_fraction: float, regress_fraction: float) -> None:
    """Take at most one ladder step, with a deadband against oscillation."""
    if self.smoothed >= advance_fraction and self.stage + 1 < len(self._mixtures):
      self.stage += 1
    elif self.smoothed < regress_fraction and self.stage > 0:
      self.stage -= 1
    else:
      return

    self._push.set_band_weights(self._mixtures[self.stage])
    # Re-smoothed from the new mixture: the old level is not evidence about it.
    self.smoothed = float("nan")


def achievement_curriculum_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  term_name: str = "push_robot",
) -> dict[str, float | int]:
  """Report the achievement event's reset-cohort mixture."""
  del env_ids
  term = _push_term(env, term_name)
  if not isinstance(term, achievement_finite_impulse_curriculum):
    raise TypeError(f"event term {term_name!r} is not achievement gated")

  return term.curriculum_state()
