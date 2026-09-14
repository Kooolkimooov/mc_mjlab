"""Deterministic qualification contracts."""

from __future__ import annotations

import contextlib
import math
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

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

from mc_mjlab.tasks.residual_balance import qualification_sidecar


@contextlib.contextmanager
def _expect_error(fragment: str) -> Iterator[None]:
  """Assert the block raises a RuntimeError naming ``fragment``."""
  try:
    yield
  except RuntimeError as error:
    assert fragment in str(error), f"{fragment!r} not in {error}"
  else:
    raise AssertionError(f"expected a RuntimeError naming {fragment!r}")


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
      "grounded_fraction": {
        "baseline": 0.98,
        "policy": 0.98,
        "paired": {"mean": 0.0, "sem": 0.001, "ci_high": 0.002, "clusters": 16.0},
      },
      "residual_rms": {"policy": 0.01},
    }
  }


def _nominal_summary() -> dict:
  """Build a passing nominal-scenario summary for the gates that read it."""
  clean = {
    "baseline": 0.03,
    "policy": 0.03,
    "paired": {"mean": 0.0, "sem": 0.0001, "ci_high": 0.0002, "clusters": 16.0},
  }
  return {
    "nominal": {
      "hazard": {"baseline": 0.10, "policy": 0.05},
      "worker_failure": {"baseline": 0.0, "policy": 0.0},
      "max_effort_ratio": {"policy": 0.5},
      "projection_fraction": {"policy": 0.0},
      "near_bound_fraction": {"policy": 0.0},
      "grounded_fraction": {
        "baseline": 0.98,
        "policy": 0.98,
        "paired": {"mean": 0.0, "sem": 0.001, "ci_high": 0.002, "clusters": 16.0},
      },
      "gate_duty": {"policy": 0.02},
      "com_velocity_error": dict(clean, baseline=0.1),
      "zmp_error": dict(clean),
      "foot_slip": dict(clean),
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
    terminations={
      "controller_worker_failed": 0,
      "fell_over": 0,
      "collapsed": 0,
      "controller_failed": 0,
    },
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
      "zmp_grounded": 0.98,
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


def test_qualifier_not_measured() -> None:
  """Check a criterion with no measurement reports NOT MEASURED, never PASS."""
  healthy = promotion(_recovery_summary(-0.01010, 0.003, 32.0))
  assert healthy["eligible"] and not healthy["not_measured"]
  assert all(item["verdict"] == "pass" for item in healthy["criteria"])

  # Every gate, one at a time: an absent measurement must block promotion and
  # must not be reported as a policy failure.
  gates = (
    ("max_effort_ratio", "policy"),
    ("projection_fraction", "policy"),
    ("near_bound_fraction", "policy"),
    ("worker_failure", "baseline"),
  )
  for name, arm in gates:
    summary = _recovery_summary(-0.01010, 0.003, 32.0)
    summary["finite_impulse"][name][arm] = float("nan")
    verdict = promotion(summary)
    assert not verdict["eligible"], name
    assert name in verdict["not_measured"], name
    record = next(
      item
      for item in verdict["criteria"]
      if item["name"] == name and item["verdict"] == "not_measured"
    )
    assert "not measured" in record["detail"], name

  # The defect that mattered most: an absent recovery metric used to divide a
  # zero default and score as a perfect recovery.
  missing = _recovery_summary(-0.01010, 0.003, 32.0)
  missing["finite_impulse"]["recovery_dcm_error"]["paired"]["mean"] = float("nan")
  verdict = promotion(missing)
  assert not verdict["eligible"]
  assert "recovery_dcm_error" in verdict["not_measured"]

  # A relative gate whose baseline is zero has no measurable ratio. It used to
  # `continue` past the gate in silence; it now says so.
  nominal = _nominal_summary()
  assert promotion(nominal)["eligible"]
  zero_base = _nominal_summary()
  zero_base["nominal"]["zmp_error"]["baseline"] = 0.0
  verdict = promotion(zero_base)
  assert not verdict["eligible"]
  assert "zmp_error" in verdict["not_measured"]

  # A zero baseline hazard with no policy hazard still ranks as a ratio of 1.0,
  # which is the pre-existing verdict and must not have moved.
  hazard_free = _nominal_summary()
  hazard_free["nominal"]["hazard"] = {"baseline": 0.0, "policy": 0.0}
  assert promotion(hazard_free)["hazard_ratio"] == 1.0

  # The contact-gate escape: less time grounded than the paired baseline arm.
  escaped = _recovery_summary(-0.01010, 0.003, 32.0)
  escaped["finite_impulse"]["grounded_fraction"]["paired"]["ci_high"] = -0.05
  verdict = promotion(escaped)
  assert not verdict["eligible"]
  assert any("grounded fraction" in reason for reason in verdict["reasons"])
  assert "grounded_fraction" not in verdict["not_measured"]


def test_qualification_sidecar() -> None:
  """Check a full resume refuses a checkpoint no sweep has qualified."""
  with tempfile.TemporaryDirectory() as directory:
    checkpoint = Path(directory) / "model_100.pt"
    checkpoint.write_bytes(b"policy")
    passing = {"eligible": True, "reasons": [], "not_measured": []}

    # No verdict beside it: actor-only warns, a full resume refuses.
    qualification_sidecar.enforce(checkpoint, full_resume=False)
    with _expect_error("sweep it first"):
      qualification_sidecar.enforce(checkpoint, full_resume=True)

    qualification_sidecar.write(checkpoint, passing, {"seeds": [42]})
    qualification_sidecar.enforce(checkpoint, full_resume=True)

    # A verdict written for different bytes is not a verdict about this file.
    checkpoint.write_bytes(b"a different policy")
    with _expect_error("different bytes"):
      qualification_sidecar.enforce(checkpoint, full_resume=True)

    failing = {
      "eligible": False,
      "reasons": ["nominal: authority duty exceeds 5%"],
      "not_measured": ["foot_slip"],
    }
    qualification_sidecar.write(checkpoint, failing, {"seeds": [42]})
    with _expect_error("1 criteria not measured"):
      qualification_sidecar.enforce(checkpoint, full_resume=True)

    os.environ[qualification_sidecar.ALLOW_UNQUALIFIED_ENV] = "1"
    try:
      qualification_sidecar.enforce(checkpoint, full_resume=True)
    finally:
      del os.environ[qualification_sidecar.ALLOW_UNQUALIFIED_ENV]


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
