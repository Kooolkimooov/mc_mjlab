"""Episode records for distinct evaluation designs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QualificationEpisode:
  """One fixed-schedule episode and its accumulated outputs."""

  checkpoint: str
  scenario: str
  seed: int
  env_id: int
  pair: int
  arm: str
  length: int
  terminations: dict[str, int]
  rewards: dict[str, float]
  metrics: dict[str, float]
