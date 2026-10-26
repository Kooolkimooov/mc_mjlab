"""Original-authority control for the safe-completion study."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mc_mjlab.tasks.locomanip.locomanip_residual_env_cfg import (
  locomanip_residual_env_cfg,
)
from mc_mjlab.tasks.locomanip.profiles import HRP5P

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg


def study_control_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Keep production intact while restoring original hardware authority."""
  return locomanip_residual_env_cfg(HRP5P, play=play, apply_scale_gains=False)
