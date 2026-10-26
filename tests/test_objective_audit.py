"""Alive replacement, cost separation and matched ranking defects."""

from __future__ import annotations

import pytest
from evaluation.objective_audit import (
  OBJECTIVES,
  episode_returns,
  ranking_defects,
  reward_matrix,
  verdict,
)

TERMS = ["contact_dcm_stability", "termination_penalty", "residual_magnitude"]
DT = 0.02


def steps(count: int, dcm: float, cost: float, terminated: bool) -> list[dict]:
  rows = [
    {
      "reward/contact_dcm_stability": dcm,
      "reward/termination_penalty": 0.0,
      "reward/residual_magnitude": cost,
    }
    for _ in range(count)
  ]
  if terminated:
    rows[-1]["reward/termination_penalty"] = -200.0
  return rows


def record(failure: bool) -> dict:
  return {"combined_failure": failure, "complete": not failure, "timeout": False}


def test_alive_swap_zeroes_only_the_terminal_step_and_costs_drop() -> None:
  values, names = reward_matrix(steps(5, 0.4, -0.5, True), TERMS, True, True)
  assert names[0] == "is_alive"
  assert values[:, 0].tolist() == [1.0, 1.0, 1.0, 1.0, 0.0]
  assert values[:, 2].tolist() == [0.0] * 5
  assert values[-1, 1] == -200.0


def test_failure_outranking_success_is_a_defect_the_candidate_must_remove() -> None:
  failure = episode_returns(
    steps(100, 1.0, 0.0, True), record(True), TERMS, DT, 0.99, 10
  )
  success = episode_returns(
    steps(110, 0.1, 0.0, False), record(False), TERMS, DT, 0.99, 10
  )
  assert set(failure) == {name for name, _, _ in OBJECTIVES}
  assert failure["current"]["undiscounted"] == pytest.approx(100 * DT - 200 * DT)
  assert success["alive"]["per_second"] == pytest.approx(1.0)
  episodes = [
    {"condition_id": "c", "category": "failure", "objectives": failure},
    {"condition_id": "c", "category": "safe_success", "objectives": success},
  ]
  defects = ranking_defects(episodes)
  assert defects["current"]["failure/undiscounted"] == {"pairs": 1, "defects": 0}
  assert defects["alive"]["failure/undiscounted"] == {"pairs": 1, "defects": 0}
  assert verdict(defects)["decision"].startswith("reject: no consistent")


def test_alive_objective_is_admitted_when_it_removes_a_dcm_defect() -> None:
  failure = episode_returns(
    steps(400, 1.0, 0.0, True), record(True), TERMS, DT, 0.99, 0
  )
  success = episode_returns(
    steps(410, 0.1, 0.0, False), record(False), TERMS, DT, 0.99, 0
  )
  episodes = [
    {"condition_id": "c", "category": "failure", "objectives": failure},
    {"condition_id": "c", "category": "safe_success", "objectives": success},
  ]
  defects = ranking_defects(episodes)
  assert defects["current"]["failure/undiscounted"]["defects"] == 1
  assert defects["alive"]["failure/undiscounted"]["defects"] == 0
  assert verdict(defects)["decision"].startswith("admit")
