"""Deterministic qualification contracts."""

from __future__ import annotations

import math

import torch
from evaluation.qualification import (
  cluster_stats,
  clusters_for_confidence,
  promotion,
  summarize,
  t_critical,
)
from evaluation.qualification_strata import classify_strata
from evaluation.records import QualificationEpisode as Episode


def _recovery_summary(mean: float, sem: float, clusters: float) -> dict:
  """Build the one qualifier summary the recovery gate reads."""
  return {
    "finite_impulse": {
      "recovery_dcm_error": {
        "baseline": 0.131,
        "relative": mean / 0.131,
        "paired": {
          "mean": mean,
          "sem": sem,
          "ci_high": mean + t_critical(int(clusters)) * sem,
          "clusters": clusters,
        },
      },
      "hazard": {"baseline": 0.09375, "policy": 0.0625},
      "worker_failure": {"baseline": 0.0, "policy": 0.0},
      "max_effort_ratio": {"policy": 0.5},
      "projection_fraction": {"policy": 0.0},
      "near_bound_fraction": {"policy": 0.0},
      "residual_rms": {"policy": 0.01},
    }
  }


def _paired_episode(seed: int, env_id: int, arm: str, recovery: float) -> Episode:
  """Build one qualifier episode carrying only the summarized metrics."""
  return Episode(
    checkpoint="model.pt",
    scenario="finite_impulse",
    seed=seed,
    env_id=env_id,
    pair=0,
    arm=arm,
    length=100,
    terminations={"controller_worker_failed": 0},
    rewards={},
    metrics={
      "recovery_dcm_error": recovery,
      "recovery_active": 1.0,
      "hazard": 0.0,
      "dcm_error": recovery,
      "com_velocity_error": 0.1,
      "zmp_error": 0.03,
      "foot_slip": 0.0,
      "projection_fraction": 0.0,
      "near_bound_fraction": 0.0,
      "max_effort_ratio": 0.5,
      "gate_mean": 0.02,
      "executed_residual_l2": 1.0e-4,
    },
  )


def test_paired_clustering() -> None:
  """Check a second seed adds clusters instead of collapsing them to two."""
  episodes = []
  for seed in (42, 43):
    for env_id in range(16):
      baseline = 0.13 + 0.002 * env_id
      episodes.append(_paired_episode(seed, env_id, "baseline", baseline))
      episodes.append(_paired_episode(seed, env_id, "policy", baseline - 0.010))
  both = summarize(episodes)["recovery_dcm_error"]
  assert both["cluster_level"] == "seed-environment"
  assert both["paired"]["clusters"] == 32.0
  assert both["paired"]["ci_high"] < 0.0
  one = summarize([e for e in episodes if e.seed == 42])["recovery_dcm_error"]
  assert one["paired"]["clusters"] == 16.0


def test_qualifier_power() -> None:
  """Check the Student-t interval, the one-cluster hole, and power reporting."""
  assert t_critical(1) == math.inf
  assert math.isclose(t_critical(2), 12.7062)
  assert math.isclose(t_critical(16), 2.1314)
  assert all(t_critical(count) > t_critical(count + 1) for count in range(2, 60))
  assert t_critical(400) > 1.959963985

  single = cluster_stats([-0.01])
  assert single["sem"] == math.inf and single["ci_high"] == math.inf
  assert single["clusters"] == 1.0
  assert not single["ci_high"] < 0.0

  # The 16-environment paired rejection of `standard/model_180`.
  assert clusters_for_confidence(-0.01010, 0.005760, 16) == 23.0
  assert math.isnan(clusters_for_confidence(0.01, 0.005, 16))
  assert math.isnan(clusters_for_confidence(-0.01, 0.005, 1))

  # An invalidated scenario must not also produce a substantive policy verdict.
  broken = _recovery_summary(-0.00131, 0.0002, 16.0)
  broken["finite_impulse"]["worker_failure"] = {"baseline": 0.0, "policy": 1.0}
  verdict = promotion(broken)
  assert not verdict["eligible"]
  assert verdict["invalidated_scenarios"] == ["finite_impulse"]
  assert any("invalidated the run" in reason for reason in verdict["reasons"])
  assert any("unreadable" in reason for reason in verdict["reasons"])
  assert not any("below 5%" in reason for reason in verdict["reasons"])
  assert not any("hazard ratio" in reason for reason in verdict["reasons"])

  unresolved = promotion(_recovery_summary(-0.01010, 0.005760, 16.0))
  assert unresolved["invalidated_scenarios"] == []
  assert not unresolved["eligible"]
  assert any("unresolved by 16 clusters" in reason for reason in unresolved["reasons"])
  assert any("23 would resolve it" in reason for reason in unresolved["reasons"])
  small = promotion(_recovery_summary(-0.00131, 0.0002, 16.0))
  assert any("below 5%" in reason for reason in small["reasons"])
  resolved = promotion(_recovery_summary(-0.01010, 0.003, 32.0))
  assert not any("recovery DCM" in reason for reason in resolved["reasons"])


def test_qualification_strata() -> None:
  """Check startup, sustained, and body-frame recovery direction classification."""
  episode_steps = torch.tensor([500, 600, 501, 600, 600, 600])
  push_age = torch.tensor([1 << 30, 1 << 30, 1, 50, 100, 100])
  push_velocity = torch.tensor(
    [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.4, 0.1, 0.0],
      [-0.4, 0.1, 0.0],
      [0.1, 0.4, 0.0],
      [0.1, -0.4, 0.0],
    ]
  )
  strata = classify_strata(episode_steps, push_age, push_velocity, 500, 100)
  assert strata.tolist() == [0, 1, 2, 3, 4, 5]
