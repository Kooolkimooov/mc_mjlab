"""Explicit condition identifiers for matching payloads and encoder draws."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from mc_mjlab.tasks.locomanip.mdp.payload_mixture import apply_mass_ranges

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def condition_id(seed: int, mass: float, repeat: int) -> str:
  """Name a matched input draw independently of row ordering or batch size."""
  return f"seed={seed}/mass={mass:g}/repeat={repeat}"


def draw_encoder_bias(identifier: str, width: int, enabled: bool) -> torch.Tensor:
  """Generate a stable CPU draw shared by every arm of one condition."""
  seed = int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:8], "little")
  generator = torch.Generator().manual_seed(seed)
  return (torch.rand(width, generator=generator) * 0.02 - 0.01) * enabled


def assert_payload_masses(actual: torch.Tensor, rows: list[dict[str, Any]]) -> None:
  """Reject a condition when the applied cart mass differs from its label."""
  requested = torch.tensor([row["mass_kg"] for row in rows], device=actual.device)
  if torch.allclose(actual, requested, atol=0.05, rtol=1e-5):
    return
  error = (actual - requested).abs().max().item()
  raise RuntimeError(f"cart payload differs from requested mass by {error:g} kg")


def matched_encoder_bias(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  rows: list[dict[str, Any]],
  enabled: bool,
) -> None:
  """Install condition-specific encoder draws before the first controller dispatch."""
  ids = list(range(env.num_envs)) if env_ids is None else env_ids.tolist()
  bias = env.scene["robot"].data.encoder_bias
  for index in ids:
    bias[index] = draw_encoder_bias(
      rows[index]["condition_id"], bias.shape[1], enabled
    ).to(env.device)


@requires_model_fields(
  "body_mass",
  "body_ipos",
  "body_inertia",
  "body_iquat",
  recompute=RecomputeLevel.set_const,
)
def fixed_payloads(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  rows: list[dict[str, Any]],
  nominal_mass: float,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Apply each condition's exact density scale, including partial resets."""
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  masses = torch.tensor([row["mass_kg"] for row in rows], device=env.device)
  assignments = [
    (ids[masses[ids] == mass], (mass, mass)) for mass in masses[ids].unique().tolist()
  ]
  apply_mass_ranges(env, assignments, nominal_mass, asset_cfg)
