"""Fixed-cohort comparison records and summary statistics."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field


def describe(values: Sequence[float]) -> dict[str, float]:
  """Mean/spread/quartiles for one per-episode quantity."""
  n = len(values)
  if n == 0:
    return dict.fromkeys(
      ("n", "mean", "std", "sem", "min", "q1", "median", "q3", "max"), float("nan")
    ) | {"n": 0}
  mean = statistics.fmean(values)
  std = statistics.stdev(values) if n > 1 else 0.0
  # numpy/pandas convention, so these paste into anything else.
  q1, med, q3 = (
    statistics.quantiles(values, n=4, method="inclusive")
    if n > 1
    else (values[0], values[0], values[0])
  )
  return {
    "n": n,
    "mean": mean,
    "std": std,
    "sem": std / math.sqrt(n) if n else 0.0,
    "min": min(values),
    "q1": q1,
    "median": med,
    "q3": q3,
    "max": max(values),
  }


def wilson(k: int, n: int) -> tuple[float, float]:
  """95% interval for a proportion; behaves at 0/n and n/n, unlike the normal one."""
  if n == 0:
    return (float("nan"), float("nan"))
  z = 1.959963985
  p = k / n
  d = 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return (max(0.0, c - h), min(1.0, c + h))


def two_proportion_p(k1: int, n1: int, k2: int, n2: int) -> float:
  """Two-sided p for equal proportions, pooled-variance normal approximation."""
  if n1 == 0 or n2 == 0:
    return float("nan")
  p = (k1 + k2) / (n1 + n2)
  se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
  if se == 0.0:
    return 1.0
  return math.erfc(abs(k1 / n1 - k2 / n2) / se / math.sqrt(2))


def welch_p(a: dict[str, float], b: dict[str, float]) -> float:
  """Two-sided p that two means differ, from their standard errors."""
  se = math.hypot(a["sem"], b["sem"])
  if se == 0.0 or a["n"] < 2 or b["n"] < 2:
    return float("nan")
  return math.erfc(abs(a["mean"] - b["mean"]) / se / math.sqrt(2))


@dataclass
class ComparisonEpisode:
  env_id: int
  nth: int
  length: int
  terms: dict[str, int]
  rewards: dict[str, float]


@dataclass
class Arm:
  label: str
  #: Every env this arm owns, including ones that never finished an episode.
  env_ids: tuple[int, ...] = ()
  episodes: list[ComparisonEpisode] = field(default_factory=list)
  steps: int = 0

  def trimmed(self) -> tuple[list[ComparisonEpisode], int]:
    """The first K episodes of every env, K set by the env that finished fewest."""
    # K over `env_ids`, not over envs that finished: the latter drops survivors.
    # docs/evaluation.md#fixed-episodes-per-env-not-everything-that-finished
    per_env: dict[int, int] = dict.fromkeys(self.env_ids, -1)
    for e in self.episodes:
      per_env[e.env_id] = max(per_env.get(e.env_id, -1), e.nth)
    if not per_env:
      return [], 0
    k = min(per_env.values()) + 1  # nth is 0-based
    return [e for e in self.episodes if e.nth < k], k


def survival(eps: Sequence[ComparisonEpisode]) -> tuple[int, int]:
  return sum(e.terms.get("time_out", 0) for e in eps), len(eps)
