"""Matched-condition and outcome contracts for independent policy environments."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from evaluation.locomanip_inputs import assert_payload_masses, draw_encoder_bias
from evaluation.study_cohort import StudyCohort
from evaluation.study_repeatability import compare_sources
from evaluation.study_statistics import measured, summarize
from evaluation.study_trace import StudyTrace


def records(cohort: StudyCohort) -> list[dict]:
  return [
    {
      **row,
      "arm": arm,
      "observed": True,
      "worker_failure": False,
      "complete": False,
      "combined_failure": False,
      "timeout": True,
      "robot_displacement_m": 1.0,
      "actual_mass_kg": row["mass_kg"],
    }
    for row in cohort.conditions()
    for arm in ("reference", "candidate")
  ]


def test_timeouts_are_measured_and_not_promoted() -> None:
  cohort = StudyCohort(masses=(600,), seeds=(1001,), draws=2)
  rows = records(cohort)
  assert all(measured(row) for row in rows)
  report = summarize(rows, cohort, samples=100)
  assert report["complete_matched_data"]
  assert report["full_range_standardized"]["completion_gain_ci95"] == [0, 0]
  assert not report["promote"]
  assert report["walking_preserved"] is None


@pytest.mark.parametrize(
  "field,value",
  [
    ("worker_failure", True),
    ("observed", False),
    ("robot_displacement_m", float("nan")),
    ("actual_mass_kg", float("inf")),
    ("warmup_failed", True),
  ],
)
def test_missing_measurements_cannot_pass(field: str, value: object) -> None:
  cohort = StudyCohort(masses=(600,), seeds=(1001,), draws=2)
  rows = records(cohort)
  rows[0][field] = value
  report = summarize(rows, cohort, samples=100)
  assert not report["complete_matched_data"]
  assert report["verdict"] == "not_measured"
  assert report["full_range_standardized"] is None


def test_order_independence_and_shared_lifecycle_draws() -> None:
  cohort = StudyCohort(masses=(600, 700), seeds=(1001, 1002), draws=3)
  rows = records(cohort)
  for i, row in enumerate(rows):
    row["complete"] = i % 3 == 0
    row["timeout"] = not row["complete"]
  assert summarize(rows, cohort, samples=100) == summarize(
    list(reversed(rows)), cohort, samples=100
  )
  conditions = cohort.conditions()
  for fresh, recycled in zip(conditions[::2], conditions[1::2], strict=True):
    assert fresh["condition_id"] != recycled["condition_id"]
    assert fresh["cluster_id"] == recycled["cluster_id"]
    assert torch.equal(
      draw_encoder_bias(fresh["input_id"], 30, True),
      draw_encoder_bias(recycled["input_id"], 30, True),
    )


def test_nominal_repeats_do_not_claim_independent_uncertainty() -> None:
  cohort = StudyCohort(masses=(600,), seeds=(1001, 1002), draws=8, nominal=True)
  report = summarize(records(cohort), cohort, samples=100)
  assert report["per_mass"][0]["clusters"] == 1
  assert report["full_range_standardized"]["completion_gain_ci95"] is None


def test_corrupt_identity_and_duplicate_arms_are_rejected() -> None:
  cohort = StudyCohort(masses=(600,), seeds=(1001,), draws=2)
  rows = records(cohort)
  with pytest.raises(ValueError, match="duplicate"):
    summarize([*rows, rows[0]], cohort)
  changed = deepcopy(rows)
  changed[0]["mass_kg"] = 700
  with pytest.raises(ValueError, match="metadata"):
    summarize(changed, cohort)


def test_first_step_controller_failure_is_a_measured_failure() -> None:
  cohort = StudyCohort(masses=(600,), seeds=(1001,), draws=1)
  rows = records(cohort)
  rows[0].update(controller_start_failure=True, combined_failure=True, timeout=False)
  assert measured(rows[0])
  report = summarize(rows, cohort, samples=100)
  assert report["complete_matched_data"]
  assert report["per_mass"][0]["reference"]["combined_failure"] == 1


def test_terminal_capture_keeps_first_step_qp_and_physical_failures(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  robot = SimpleNamespace(
    data=SimpleNamespace(
      root_link_pos_w=torch.zeros(2, 3), encoder_bias=torch.zeros(2, 1)
    )
  )
  env = SimpleNamespace(
    num_envs=2,
    device="cpu",
    scene={"robot": robot},
    step_dt=0.02,
    reward_manager=SimpleNamespace(_episode_sums={}),
  )
  monkeypatch.setattr("evaluation.study_trace.metric_snapshot", lambda env, ids: {})
  monkeypatch.setattr(
    "evaluation.study_trace.accessors.object_mass_kg",
    lambda env: torch.full((2,), 600.0),
  )
  trace = StudyTrace(env, [{}, {}], None)
  frame = {
    "terminations": [
      {"controller_failed": True},
      {"controller_failed": True, "fell_over": True},
    ],
    "time_s": [0.02, 0.02],
    "phase_left": [0, 0],
    "phase_right": [0, 0],
  }
  trace.finish(torch.tensor([0, 1]), frame)
  assert all(
    row is not None and row["combined_failure"] and not row["complete"]
    for row in trace.results
  )


def test_integer_json_masses_keep_float_identities() -> None:
  cohort = StudyCohort(masses=(400, 700), seeds=(1001,), draws=1)
  rows = cohort.conditions()
  assert all(type(row["mass_kg"]) is float for row in rows)
  assert rows[0]["input_id"] == "seed=1001/mass=400/repeat=0"
  assert_payload_masses(torch.tensor([400.0, 700.0]), rows[::2])


def test_repeatability_flags_outcome_changes_and_spreads() -> None:
  cohort = StudyCohort(masses=(600,), seeds=(1001,), draws=1)
  first = {r["condition_id"]: {**r, "time_s": 10.0} for r in records(cohort)[::2]}
  second = deepcopy(first)
  changed = next(iter(second))
  second[changed].update(timeout=False, combined_failure=True, time_s=9.5)
  second[changed]["robot_displacement_m"] += 0.25
  report = compare_sources({"a": first, "b": second})
  assert report["outcome_disagreements"] == [
    {"condition_id": changed, "field": "combined_failure"},
    {"condition_id": changed, "field": "timeout"},
  ]
  assert report["max_displacement_spread_m"] == 0.25
  assert report["max_time_spread_s"] == 0.5
