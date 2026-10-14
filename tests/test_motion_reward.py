"""Braking costs use applied authority and preserve terminal reward scaling."""

from __future__ import annotations

from typing import Any

import pytest
from evaluation.motion_reward import score_braking


def frames(arm: str, rates: list[float], scale: float = 1.0) -> list[dict[str, Any]]:
  """Build a fully observed first episode with timestep-scaled reward rates."""
  return [
    {
      "arm": arm,
      "time_s": (i + 1) * 0.02,
      "trigger_time_s": 0.0 if arm != "baseline" else None,
      "reward": rate * 0.02,
      "reward_rates": {"test": rate},
      "applied_scale": scale,
      "dispatched_scale": 1.0,
      "holding": 1.0,
      "motion_state": 2.0 if scale == 0.0 else 0.0,
      "worker_failed": 0.0,
      "controller_failed": 0.0,
      "fell_over": 0.0,
      "collapsed": 0.0,
      "complete": float(i == len(rates) - 1),
      "terminated": i == len(rates) - 1,
      "truncated": False,
    }
    for i, rate in enumerate(rates)
  ]


def test_discounted_cost_includes_capped_resume_and_terminal_reward() -> None:
  samples = frames("baseline", [1.0, -200.0])
  samples += frames("slow", [1.0, 1.0, 1.0], 0.5)
  report = score_braking(samples, step_dt=0.02, gamma=0.5, costs=[1.1])
  baseline, slow = report["arms"]["baseline"], report["arms"]["slow"]
  assert baseline["discounted_original"] == pytest.approx(0.02 - 2.0)
  assert baseline["discounted_braking_seconds"] == 0.0
  assert slow["discounted_braking_seconds"] == pytest.approx(0.0175)
  assert slow["returns"]["1.1"]["discounted"] == pytest.approx(0.035 - 1.1 * 0.0175)


def test_full_hold_cost_exceeds_positive_reward_and_respects_phase_gate() -> None:
  samples = frames("baseline", [1.0, 1.0]) + frames("hold", [1.0, 1.0], 0.0)
  samples[-2]["holding"] = 0.0
  report = score_braking(samples, step_dt=0.02, gamma=1.0, costs=[1.1])
  held = report["arms"]["hold"]
  assert held["braking_seconds"] == 0.02
  assert held["returns"]["1.1"]["undiscounted"] == pytest.approx(0.018)
  samples[-2]["holding"] = 1.0
  report = score_braking(samples, step_dt=0.02, gamma=1.0, costs=[1.1])
  assert report["arms"]["hold"]["returns"]["1.1"]["undiscounted"] < 0.0


def test_discounting_can_prefer_late_failure_to_successful_braking() -> None:
  samples = frames("baseline", [1.0] * 1000 + [-200.0])
  samples += frames("slow", [1.0] * 2000, 0.5)
  report = score_braking(samples, step_dt=0.02, gamma=0.997, costs=[1.1])
  assert (
    report["arms"]["baseline"]["returns"]["1.1"]["discounted"]
    > report["arms"]["slow"]["returns"]["1.1"]["discounted"]
  )


@pytest.mark.parametrize(
  "fault", ["gap", "reward", "worker", "incomplete", "untriggered"]
)
def test_bad_or_unmeasured_traces_are_rejected(fault: str) -> None:
  samples = frames("baseline", [1.0, 1.0]) + frames("hold", [1.0, 1.0], 0.0)
  if fault == "gap":
    samples[-1]["time_s"] += 0.02
  elif fault == "reward":
    samples[-1]["reward"] += 1.0
  elif fault == "worker":
    samples[-1]["worker_failed"] = 1.0
  elif fault == "incomplete":
    samples[-1]["terminated"] = False
  else:
    for frame in samples:
      frame["trigger_time_s"] = None
  with pytest.raises(ValueError):
    score_braking(samples, step_dt=0.02, gamma=0.997, costs=[1.1])
