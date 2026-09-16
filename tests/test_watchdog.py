"""Deterministic watchdog contracts."""

from __future__ import annotations

import math

from mc_mjlab.tasks.residual_balance.training_watchdog import (
  HealthObservation,
  WatchdogThresholds,
  decide_health,
  intervention_for,
  qualification_baseline,
  qualification_regressions,
)


def test_training_watchdog() -> None:
  """Check escalation, preserve-before-stop, and post-attach regression guards."""
  thresholds = WatchdogThresholds()
  healthy = HealthObservation(0.0, 0.0, 12_000.0, 0, 0, 0)
  assert decide_health(healthy, thresholds).level == "ok"
  worker_degraded = HealthObservation(0.0, 0.0, 12_000.0, 0, 3, 0)
  assert decide_health(worker_degraded, thresholds).level == "preserve"
  near_oom = HealthObservation(0.0, 0.0, 512.0, 2, 0, 0)
  assert decide_health(near_oom, thresholds).level == "stop"
  assert intervention_for("stop", "stop", False, False) == "preserve"
  assert intervention_for("stop", "stop", True, False) == "stop"
  assert intervention_for("stop", "warn", False, False) == "warn"
  assert intervention_for("preserve", "stop", True, False) is None

  samples = [
    {"eligible": True, "hazard_ratio": 0.8, "recovery_gain": 0.10},
    {"eligible": True, "hazard_ratio": 0.9, "recovery_gain": 0.12},
  ]
  baseline = qualification_baseline(samples)
  assert math.isclose(baseline["hazard_ratio"], 0.85)
  regressions = qualification_regressions(
    baseline,
    {"eligible": False, "hazard_ratio": 1.0, "recovery_gain": 0.04},
  )
  assert len(regressions) == 3
