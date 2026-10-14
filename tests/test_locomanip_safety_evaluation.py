"""Causal probe scheduling and non-compensating safety acceptance contracts."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from evaluation.locomanip_inputs import (
  assert_payload_masses,
  condition_id,
  draw_encoder_bias,
)
from evaluation.locomanip_promotion import summarize_safety
from evaluation.locomanip_retries import retry_cohort
from evaluation.locomanip_schedules import ForceSchedule, reactive_conditions
from screen_locomanip_safety import select_candidate


def step(schedule: ForceSchedule, time: float, lag: float, hold: bool = True) -> bool:
  return bool(
    schedule.update(
      torch.tensor([time]), torch.tensor([lag]), torch.tensor([hold]), 0.02
    )[0]
  )


def test_pulse_uses_first_crossing_and_stops_after_one_second() -> None:
  schedule = ForceSchedule(["pulse"], "cpu")
  assert not step(schedule, 1.0, 0.1, False)
  assert not step(schedule, 1.02, 0.05)
  assert step(schedule, 1.04, 0.051)
  assert step(schedule, 2.02, 0.0)
  assert not step(schedule, 2.04, 0.1)
  assert not step(schedule, 3.0, 0.2)
  assert schedule.trigger_time[0] == pytest.approx(1.04)


def test_pulse_counts_integer_steps_near_episode_end() -> None:
  schedule = ForceSchedule(["pulse"], "cpu")
  assert step(schedule, 49.0, 0.051)
  for tick in range(1, 50):
    assert step(schedule, 49.0 + tick * 0.02, 0.0)
  assert not step(schedule, 50.0, 0.0)
  assert schedule.duration[0] == pytest.approx(1.0)


def test_reactive_hysteresis_hold_and_partial_reset() -> None:
  schedule = ForceSchedule(["reactive", "reactive"], "cpu")

  def update(lags: list[float], hold: list[bool]) -> list[bool]:
    return schedule.update(
      torch.tensor([2.0, 2.0]), torch.tensor(lags), torch.tensor(hold), 0.02
    ).tolist()

  assert update([0.051, 0.051], [True, True]) == [True, True]
  assert update([0.04, 0.03], [True, True]) == [True, True]
  assert update([0.029, 0.2], [True, False]) == [False, False]
  assert update([0.04, 0.051], [True, True]) == [False, True]
  schedule.reset(torch.tensor([0]))
  assert schedule.trigger_time.tolist() == [-1.0, 2.0]
  assert schedule.duration[0] == 0
  assert schedule.duration[1] > 0


def test_constant_is_hold_gated_and_untriggered_remains_unmeasured() -> None:
  constant = ForceSchedule(["constant"], "cpu")
  assert not step(constant, 0.0, 0, False)
  assert step(constant, 1.0, 0)
  assert not step(constant, 2.0, 0, False)
  pulse = ForceSchedule(["pulse"], "cpu")
  assert not step(pulse, 0.0, 0.04)
  assert not step(pulse, 0.1, float("nan"))
  assert pulse.trigger_time[0] == -1
  assert pulse.duration[0] == 0
  rows = reactive_conditions()
  assert len(rows) == 3 * 3 * 39
  assert sum(row["level"] == 0 for row in rows) == 27


def test_condition_inputs_do_not_depend_on_order_or_global_rng() -> None:
  identifier = condition_id(43, 525, 2)
  a = draw_encoder_bias(identifier, 12, True)
  torch.manual_seed(99)
  torch.rand(1000)
  torch.testing.assert_close(draw_encoder_bias(identifier, 12, True), a)
  assert not torch.equal(draw_encoder_bias(condition_id(44, 525, 2), 12, True), a)
  assert not draw_encoder_bias(identifier, 12, False).any()
  assert torch.all(a.abs() <= 0.01)
  assert_payload_masses(torch.tensor([525.0]), [{"mass_kg": 525.0}])
  with pytest.raises(RuntimeError, match="cart payload differs"):
    assert_payload_masses(torch.tensor([524.0]), [{"mass_kg": 525.0}])


def cohorts() -> list[dict[str, Any]]:
  return [
    {
      "condition_id": condition_id(42, mass, repeat),
      "mass_kg": mass,
      "arm": arm,
      "observed": True,
      "worker_failure": False,
      "combined_failure": mass == 550 and arm == "baseline",
      "complete": mass == 525 or arm == "policy",
      "terminations": {"collapsed": mass == 550 and arm == "baseline"},
    }
    for mass in (525, 550)
    for repeat in range(8)
    for arm in ("baseline", "policy")
  ]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
  return summarize_safety(rows, [525, 550], [525, 550], [42], 8)


def test_pooled_rescue_cannot_buy_a_failure_at_safe_mass() -> None:
  rows = cohorts()
  assert summarize(rows)["promote"]
  row = next(row for row in rows if row["mass_kg"] == 525 and row["arm"] == "policy")
  row.update(combined_failure=True, complete=False, terminations={"fell_over": True})
  result = summarize(rows)
  assert result["pooled_improvement"]
  assert not result["per_mass_protection"]
  assert not result["promote"]
  assert result["per_mass"][0]["baseline"]["combined_failure_ci95"][1] > 0


@pytest.mark.parametrize("change", ["missing", "worker", "incomplete", "unobserved"])
def test_missing_or_incomplete_rows_never_become_safe_successes(change: str) -> None:
  rows = cohorts()
  row = rows[1]
  if change == "missing":
    rows.pop(1)
  elif change == "worker":
    row.update(worker_failure=True, complete=False)
  elif change == "unobserved":
    row.update(observed=False)
  else:
    row.update(complete=False)
  assert not summarize(rows)["promote"]


def test_incomplete_policy_row_at_failing_mass_invalidates_pooled_cohort() -> None:
  rows = cohorts()
  row = next(row for row in rows if row["mass_kg"] == 550 and row["arm"] == "policy")
  row["complete"] = False
  result = summarize(rows)
  assert not result["complete_matched_data"]
  assert result["pooled"] is None
  assert result["per_mass"][1]["policy"]["incomplete"] == 1
  assert result["per_mass"][1]["policy"]["usable"] == 7
  assert not result["promote"]


def test_failure_union_counts_each_episode_once_and_keeps_causes() -> None:
  rows = cohorts()
  for row in rows:
    if row["combined_failure"]:
      row["terminations"] = {"fell_over": True, "controller_failed": True}
  mass = summarize(rows)["per_mass"][1]["baseline"]
  assert mass["combined_failures"] == 8
  assert mass["causes"]["fell_over"] == mass["causes"]["controller_failed"] == 8


def test_duplicate_pairs_are_rejected() -> None:
  rows = cohorts()
  with pytest.raises(ValueError, match="duplicate"):
    summarize([*rows, rows[0]])


def test_checkpoint_selection_refuses_the_safe_mass_regression() -> None:
  rows = cohorts()
  good = summarize(rows)
  bad_rows = cohorts()
  next(
    row for row in bad_rows if row["mass_kg"] == 525 and row["arm"] == "policy"
  ).update(combined_failure=True, complete=False)
  bad = summarize(bad_rows)
  bad["pooled"]["relative_reduction"] = 1.0
  assert select_candidate({200: bad, 400: good}) == 400


def test_checkpoint_selection_refuses_only_negative_reductions() -> None:
  bad = summarize(cohorts())
  bad["pooled"]["relative_reduction"] = -0.1
  assert select_candidate({200: bad}) is None


def test_failure_regression_outside_transition_grid_blocks_promotion() -> None:
  rows = cohorts()
  for repeat in range(8):
    for arm in ("baseline", "policy"):
      failed = arm == "policy" and repeat < 4 or arm == "baseline" and repeat == 0
      rows.append(
        {
          "condition_id": condition_id(42, 100, repeat),
          "mass_kg": 100,
          "arm": arm,
          "observed": True,
          "worker_failure": False,
          "combined_failure": failed,
          "complete": not failed,
          "terminations": {"fell_over": failed},
        }
      )
  report = summarize_safety(rows, [100, 525, 550], [525, 550], [42], 8)
  assert report["pooled_improvement"]
  assert report["per_mass_protection"]
  assert not report["no_mass_regression"]
  assert not report["promote"]


@pytest.mark.parametrize("fault", ["worker", "unobserved", "incomplete", "missing"])
def test_retry_cohort_replays_only_the_affected_pair(fault: str) -> None:
  rows = cohorts()[:4]
  calls = []

  def run_once(pending: list[dict[str, Any]], attempt: int) -> list[dict[str, Any]]:
    calls.append([row["condition_id"] for row in pending])
    result = [row.copy() for row in pending]
    if attempt == 0:
      if fault == "missing":
        result.pop(1)
      elif fault == "worker":
        result[1].update(worker_failure=True, complete=False)
      elif fault == "unobserved":
        result[1]["observed"] = False
      else:
        result[1]["complete"] = False
    return result

  result, history, unresolved = retry_cohort(rows, run_once)
  assert len(calls) == 2
  assert calls[1] == calls[0][:2]
  assert history[0]["unresolved"] == [rows[0]["condition_id"]]
  assert not unresolved
  assert all(not row["worker_failure"] for row in result)


def test_retry_cohort_reports_persistent_infrastructure_failure() -> None:
  rows = cohorts()[:2]
  result, history, unresolved = retry_cohort(
    rows,
    lambda pending, attempt: [
      {**row, "worker_failure": True, "complete": False} for row in pending
    ],
    max_retries=1,
  )
  assert len(history) == 2
  assert unresolved == [rows[0]["condition_id"]]
  assert not summarize(result)["promote"]
