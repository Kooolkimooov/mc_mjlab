"""The cart payload draw: kilograms to `pseudo_inertia`'s log scale, and its laws."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.dr import Distribution, uniform
from mjlab.utils.lab_api.math import sample_uniform

from mc_mjlab.tasks.locomanip.mdp.accessors import OBJECT_ENTITY, object_mass_kg

CART_MASS_RANGE_KG = (1.0, 700.0)

# Laws in kilograms over a `mass_alpha_range`. `pseudo_inertia` draws all ten of its
# parameters with one, and a (0, 0) range still gives exactly 0 under either.
LOG_MASS = Distribution(name="log_mass", sample=uniform.sample)
FLAT_MASS = Distribution(
  name="flat_mass",
  sample=lambda lo, hi, shape, device: (
    0.5 * sample_uniform(torch.exp(2.0 * lo), torch.exp(2.0 * hi), shape, device).log()
  ),
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def mass_alpha_range(
  mass_range_kg: tuple[float, float], nominal_mass_kg: float
) -> tuple[float, float]:
  """Convert a mass range to `pseudo_inertia`'s log scale, where mass is e^(2a)."""
  low, high = mass_range_kg
  if low <= 0.0 or high < low:
    raise ValueError(f"invalid mass range {mass_range_kg}")
  return (
    0.5 * math.log(low / nominal_mass_kg),
    0.5 * math.log(high / nominal_mass_kg),
  )


def print_object_mass(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  entity_name: str = OBJECT_ENTITY,
) -> None:
  """Print each reset row's drawn payload; the viewers show no event or action term."""
  # Registered after `cart_payload`, so the draw it reports is this episode's.
  mass = object_mass_kg(env, entity_name)
  rows = range(env.num_envs) if env_ids is None else env_ids.tolist()
  drawn = ", ".join(f"env {int(row)}: {float(mass[int(row)]):.1f} kg" for row in rows)
  print(f"[cart] {drawn}", flush=True)
