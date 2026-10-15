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


def test_native_event_cost_is_once_per_start_without_double_charging() -> None:
  samples = frames("baseline", [1.0, 1.0])
  held = frames("hold", [1.0, 1.0], 0.0)
  for index, frame in enumerate(held):
    frame["braking_events"] = 2.0 if index else 0.0
    frame["reward_rates"].update(
      braking=-0.03, braking_start=-0.22 * frame["braking_events"] / 0.02
    )
    frame["reward"] = sum(frame["reward_rates"].values()) * 0.02
  report = score_braking(
    samples + held, step_dt=0.02, gamma=1.0, costs=[0.03], event_costs=[0.22]
  )
  result = report["arms"]["hold"]
  assert result["braking_events"] == 2.0
  assert result["event_count_source"] == "native"
  assert result["discounted_original"] == pytest.approx(0.04)
  assert result["event_returns"]["0.03"]["0.22"] == pytest.approx(
    sum(f["reward"] for f in held)
  )


def test_holding_duration_cost_uses_phase_and_removes_captured_charge() -> None:
  baseline = frames("baseline", [1.0, 1.0])
  held = frames("hold", [1.0, 1.0], 0.0)
  held[0]["holding"] = 0.0
  held[1]["reward_rates"]["holding"] = -0.05
  held[1]["reward"] -= 0.001
  result = score_braking(
    baseline + held, step_dt=0.02, gamma=1.0, costs=[0.0], holding_costs=[0.05]
  )["arms"]["hold"]
  assert result["discounted_holding_seconds"] == 0.02
  assert result["discounted_original"] == pytest.approx(0.04)
  assert result["holding_returns"]["0.0"]["0.0"]["0.05"] == pytest.approx(0.039)


def test_rescaled_safety_rewards_match_independent_reward_calculation() -> None:
  samples = frames("baseline", [0.0])
  samples[0]["reward_rates"] = {
    "contact_dcm_stability": 1.0,
    "termination_penalty": -200.0,
  }
  samples[0]["reward"] = -199.0 * 0.02
  samples[0]["collapsed"] = 1
  result = score_braking(
    samples,
    step_dt=0.02,
    gamma=0.997,
    costs=[0.03],
    reward_scales={"contact_dcm_stability": 0.25, "termination_penalty": 2.0},
    decision_time_s=0.0,
  )["arms"]["baseline"]
  assert result["discounted_original"] == pytest.approx((0.25 - 400) * 0.02)
  assert result["discounted_terms"] == pytest.approx(
    {
      "contact_dcm_stability": 0.005,
      "termination_penalty": -8.0,
    }
  )


def test_post_hold_audit_excludes_sunk_start_cost_and_keeps_future_duration_cost() -> (
  None
):
  samples = frames("hold", [1, 1, 1], 0.0)
  for i, sample in enumerate(samples):
    sample["braking_events"] = float(i == 0)
  result = score_braking(
    samples,
    step_dt=0.02,
    gamma=1,
    costs=[0.03],
    event_costs=[0.22],
    holding_costs=[0.1],
    decision_time_s=0.02,
  )["arms"]["hold"]
  assert result["discounted_braking_events"] == 0
  assert result["holding_returns"]["0.03"]["0.22"]["0.1"] == pytest.approx(
    0.04 * (1 - 0.03 - 0.1)
  )
  with pytest.raises(ValueError, match="no measured rewards"):
    score_braking(samples, step_dt=0.02, gamma=1, costs=[0], decision_time_s=0.06)


@pytest.mark.parametrize(
  "scales", [{"braking": 2}, {"test": -1}, {"test": float("nan")}, {"absent": 1}]
)
def test_invalid_or_uncaptured_reward_scales_are_rejected(
  scales: dict[str, float],
) -> None:
  with pytest.raises(ValueError):
    score_braking(
      frames("hold", [1]), step_dt=0.02, gamma=1, costs=[0], reward_scales=scales
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
