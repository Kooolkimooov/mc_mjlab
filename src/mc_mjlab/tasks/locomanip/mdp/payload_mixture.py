"""Log-uniform cart payload bands around a measured controller transition."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from mc_mjlab.tasks.locomanip.mdp.events import (
  CART_MASS_RANGE_KG,
  LOG_MASS,
  mass_alpha_range,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def mass_bands(
  midpoint_kg: float, bounds_kg: tuple[float, float] = CART_MASS_RANGE_KG
) -> tuple[tuple[float, float], ...]:
  """Return central, lower and upper payload bands within 10–1000 kg."""
  lower, upper = bounds_kg
  if not math.isfinite(midpoint_kg) or not lower < midpoint_kg < upper:
    raise ValueError("transition midpoint must be finite and inside mass bounds")
  low, high = max(lower, midpoint_kg - 75.0), min(upper, midpoint_kg + 75.0)
  return ((low, high), (lower, low), (high, upper))


def apply_mass_ranges(
  env: ManagerBasedRlEnv,
  assignments: list[tuple[torch.Tensor, tuple[float, float]]],
  nominal_mass_kg: float,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Apply each nonempty mass group through pseudo-inertia once."""
  for selected, mass_range in assignments:
    if selected.numel() == 0:
      continue
    dr.pseudo_inertia(
      env,
      selected,
      asset_cfg=asset_cfg,
      alpha_range=mass_alpha_range(mass_range, nominal_mass_kg),
      distribution=LOG_MASS,
    )


@requires_model_fields(
  "body_mass",
  "body_ipos",
  "body_inertia",
  "body_iquat",
  recompute=RecomputeLevel.set_const,
)
def transition_payload(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  midpoint_kg: float,
  nominal_mass_kg: float,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Draw 60/20/20 percent central/lower/upper mass bands at each reset."""
  bands = mass_bands(midpoint_kg)
  ids = (
    torch.arange(env.num_envs, device=env.device)
    if env_ids is None
    else env_ids.to(env.device)
  )
  if ids.numel() == 0:
    return
  active = [
    (band, weight)
    for band, weight in zip(bands, (0.6, 0.2, 0.2), strict=True)
    if band[0] < band[1]
  ]
  weights = torch.tensor([weight for _, weight in active], device=env.device)
  boundaries = (weights / weights.sum()).cumsum(0)[:-1]
  draw = torch.rand(ids.numel(), device=env.device)
  band_ids = torch.searchsorted(boundaries, draw, right=True)
  assignments = [
    (ids[band_ids == index], band) for index, (band, _) in enumerate(active)
  ]
  apply_mass_ranges(env, assignments, nominal_mass_kg, asset_cfg)
