"""Typed, batch-independent inputs for the locomanip completion study."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from evaluation.locomanip_inputs import condition_id

MASS_GRID = (400, 450, 500, 550, 600, 625, 650, 675, 700, 725, 750, 775, 800, 850, 900)
TRANSITION_GRID = tuple(range(600, 801, 25))


@dataclass(frozen=True)
class StudyCohort:
  """Independent input draws shared across arms and lifecycle cohorts."""

  masses: tuple[float, ...] = MASS_GRID
  seeds: tuple[int, ...] = (1001, 1002)
  draws: int = 8
  lifecycles: tuple[Literal["fresh", "recycled"], ...] = ("fresh", "recycled")
  nominal: bool = False

  def __post_init__(self) -> None:
    for values in (self.masses, self.seeds, self.lifecycles):
      if not values or len(values) != len(set(values)):
        raise ValueError("cohort axes must be nonempty and unique")
    if self.draws < 1 or any(not math.isfinite(m) or m <= 0 for m in self.masses):
      raise ValueError("positive draws and finite positive masses required")
    if not set(self.lifecycles) <= {"fresh", "recycled"}:
      raise ValueError("unknown lifecycle")

  def conditions(self) -> list[dict]:
    """Keep nominal repeats in one statistical cluster rather than inflating n."""
    rows = []
    for mass in self.masses:
      for seed in self.seeds:
        for draw in range(self.draws):
          input_id = condition_id(seed, mass, draw)
          cluster = f"nominal/mass={mass:g}" if self.nominal else input_id
          for lifecycle in self.lifecycles:
            rows.append(
              {
                "condition_id": f"{input_id}/lifecycle={lifecycle}",
                "input_id": input_id,
                "cluster_id": cluster,
                "seed": seed,
                "repeat": draw,
                "mass_kg": mass,
                "lifecycle": lifecycle,
                "nominal": self.nominal,
              }
            )
    return rows
