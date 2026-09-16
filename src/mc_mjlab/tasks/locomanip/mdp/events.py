"""Converting a payload range into the log scale `pseudo_inertia` samples in."""

from __future__ import annotations

import math


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
