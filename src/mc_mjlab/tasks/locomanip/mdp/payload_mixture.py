"""Log-uniform cart payload bands around a measured controller transition."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from mc_mjlab.tasks.locomanip.mdp.events import mass_alpha_range

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def mass_bands(midpoint_kg: float) -> tuple[tuple[float, float], ...]:
  """Return central, lower and upper payload bands within 10–1000 kg."""
  if not math.isfinite(midpoint_kg) or not 10 < midpoint_kg < 1000:
    raise ValueError("transition midpoint must be finite and inside (10, 1000) kg")
  low, high = max(10.0, midpoint_kg - 75.0), min(1000.0, midpoint_kg + 75.0)
  return ((low, high), (10.0, low), (high, 1000.0))


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
  draw = torch.rand(ids.numel(), device=env.device)
  band_ids = torch.where(draw < 0.6, 0, torch.where(draw < 0.8, 1, 2))
  for index, band in enumerate(bands):
    selected = ids[band_ids == index]
    if selected.numel() == 0:
      continue
    dr.pseudo_inertia(
      env,
      selected,
      asset_cfg=asset_cfg,
      alpha_range=mass_alpha_range(band, nominal_mass_kg),
    )
