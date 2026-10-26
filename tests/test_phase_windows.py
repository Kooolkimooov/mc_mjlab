"""Raw-phase events, bounded windows and operating/fall separation."""

from __future__ import annotations

import pytest
from evaluation.phase_summary import outcome_matrix, zero_contrasts
from evaluation.phase_windows import (
  GROUPS,
  find_events,
  segment_bounds,
  window_summary,
)

from mc_mjlab.tasks.locomanip.mdp.accessors import HOLD_PHASE

DT = 0.02
NAMES = {
  "joints": ["RSP", "WP", "RCP", "RWRY"],
  "residual": ["RSP", "WP"],
  "rewards": ["foot_slip"],
}


def step(index: int, phase: float, fresh: bool = True) -> dict:
  return {
    "time_s": round((index + 1) * DT, 6),
    "phase_left": phase,
    "phase_right": phase,
    "datastore_fresh": fresh,
    "root_position_m": [0.01 * index, 0.0, 0.8],
    "tilt_rad": 0.001 * index,
    "cart_error_m": [0.1, 0.0, 0.0],
    "hand_force_n": [0.0, 0.0, 10.0, 0.0, 0.0, 20.0],
    "foot_force_kn": [1.0, 2.0],
    "joint_position_rad": [0.0] * 4,
    "joint_position_target_rad": [0.1] * 4,
    "executed_residual_rad": [0.05, -0.02],
    **{f"pd_window_peak_nm/{g}": float(index) for g in GROUPS},
    **{f"pd_window_saturated_s/{g}": 0.0 for g in GROUPS},
    "reward/foot_slip": -1.0,
  }


def trace() -> list[dict]:
  phases = [0.0] * 250 + [HOLD_PHASE] * 750 + [5.0] * 500
  steps = [step(i, phase) for i, phase in enumerate(phases)]
  steps[249]["phase_left"] = steps[249]["phase_right"] = HOLD_PHASE
  steps[249]["datastore_fresh"] = False
  return steps


def test_events_require_fresh_bilateral_hold() -> None:
  steps = trace()
  assert find_events(steps) == {"hold_entry": 250, "release": 1000, "terminal": 1499}
  steps[1000]["datastore_fresh"] = False
  assert find_events(steps)["release"] == 1001
  assert find_events([step(i, 0.0) for i in range(10)])["hold_entry"] is None


def test_windows_span_one_second_before_to_two_after() -> None:
  steps = trace()
  bounds = segment_bounds(steps, find_events(steps), failed=True)
  entry = steps[bounds["hold_entry"][0]]["time_s"], steps[bounds["hold_entry"][1]]
  assert entry[0] == pytest.approx(steps[250]["time_s"] - 1.0, abs=DT)
  assert entry[1]["time_s"] == pytest.approx(steps[250]["time_s"] + 2.0, abs=DT)
  assert bounds["operating"][1] + 1 == bounds["fall_transient"][0]
  assert bounds["sustained_hold"][1] == 999
  completed = segment_bounds(steps, find_events(steps), failed=False)
  assert completed["operating"] == (0, 1499) and "fall_transient" not in completed


def test_window_summary_groups_and_rewards() -> None:
  steps = trace()
  summary = window_summary(steps, 250, 349, NAMES, DT)
  assert summary["steps"] == 100
  assert summary["arms"]["max_executed_rad"] == pytest.approx(0.05)
  assert summary["legs"]["max_executed_rad"] is None
  assert summary["wrists"]["mean_tracking_error_rad"] == pytest.approx(0.1)
  assert summary["arms"]["max_pd_peak_nm"] == 349
  assert summary["reward"]["foot_slip"] == pytest.approx(-100 * DT)
  assert summary["max_hand_force_n"] == [10.0, 20.0]
  assert summary["min_foot_load_kn"] == 3.0


def test_outcomes_and_zero_contrasts_pair_by_condition() -> None:
  def episode(arm: str, lifecycle: str, complete: bool, end: float) -> dict:
    return {
      "arm": arm,
      "condition_id": f"seed=1/mass=700/repeat=0/lifecycle={lifecycle}",
      "mass_kg": 700.0,
      "lifecycle": lifecycle,
      "complete": complete,
      "timeout": False,
      "combined_failure": not complete,
      "terminations": [] if complete else ["collapsed"],
      "event_time_s": {"hold_entry": 11.0, "release": None, "terminal": end},
      "windows": {},
    }

  phases = [
    episode("zero", "fresh", False, 46.0),
    episode("zero", "recycled", False, 46.0),
    episode("arms", "recycled", True, 49.0),
    episode("arms", "fresh", True, 50.0),
  ]
  assert outcome_matrix(phases) == {"zero": {"700": "XX"}, "arms": {"700": "CC"}}
  contrast = zero_contrasts(phases)["arms"]
  assert contrast["completion"] == {"mean": 1.0, "n": 2}
  assert contrast["end_time_s"]["mean"] == 3.5
