"""Paired completion estimates that retain timeouts and unmeasured conditions."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, TypeGuard

import numpy as np

from evaluation.study_cohort import TRANSITION_GRID, StudyCohort


def measured(row: dict | None) -> TypeGuard[dict]:
  """A measured timeout is usable; infrastructure or nonfinite data is not."""
  return bool(
    row is not None
    and row.get("observed") is True
    and row.get("worker_failure") is False
    and row.get("warmup_failed", False) is False
    and all(
      isinstance(row.get(key), bool)
      for key in ("complete", "combined_failure", "timeout")
    )
    and isinstance(row.get("robot_displacement_m"), (int, float))
    and math.isfinite(row["robot_displacement_m"])
    and isinstance(row.get("actual_mass_kg"), (int, float))
    and math.isfinite(row["actual_mass_kg"])
    and math.isclose(row["actual_mass_kg"], row["mass_kg"], abs_tol=0.05)
  )


def summarize(rows: list[dict], cohort: StudyCohort, samples: int = 10000) -> dict:
  """Report outcome evidence without claiming the campaign's unmeasured gates."""
  if samples < 1:
    raise ValueError("positive bootstrap sample count required")
  expected = {r["condition_id"]: r for r in cohort.conditions()}
  indexed = {}
  for row in rows:
    key = (row["condition_id"], row["arm"])
    if (
      key in indexed
      or key[0] not in expected
      or key[1] not in ("reference", "candidate")
    ):
      raise ValueError(f"duplicate or unexpected condition arm: {key}")
    if any(row.get(k) != v for k, v in expected[key[0]].items()):
      raise ValueError(f"condition metadata changed: {key}")
    indexed[key] = row
  pairs = []
  missing = []
  for identifier in expected:
    b, p = (indexed.get((identifier, arm)) for arm in ("reference", "candidate"))
    if measured(b) and measured(p):
      pairs.append((b, p))
    else:
      missing.append(identifier)
  per_mass = []
  for lifecycle in cohort.lifecycles:
    for mass in cohort.masses:
      selected = [
        (b, p) for b, p in pairs if b["mass_kg"] == mass and b["lifecycle"] == lifecycle
      ]
      per_mass.append(
        {
          "mass_kg": mass,
          "lifecycle": lifecycle,
          **_cell(selected, samples, cohort.nominal),
        }
      )
  complete = not missing
  failure_gate = complete and all(c["no_failure_increase"] for c in per_mass)
  timeout_gate = complete and all(c["no_incomplete_increase"] for c in per_mass)
  jointly_completed = [(b, p) for b, p in pairs if b["complete"] and p["complete"]]
  displacement = None
  if jointly_completed:
    denominator = sum(b["robot_displacement_m"] for b, _ in jointly_completed)
    if denominator > 0:
      displacement = (
        sum(p["robot_displacement_m"] for _, p in jointly_completed) / denominator
      )
  return {
    "promote": False,
    "verdict": "not_measured" if missing else "campaign_gates_pending",
    "complete_matched_data": complete,
    "unmeasured_conditions": missing,
    "worker_failures": sum(bool(row.get("worker_failure")) for row in rows),
    "per_mass": per_mass,
    "full_range_standardized": _standardized(
      pairs, cohort.masses, cohort.lifecycles, samples, cohort.nominal
    )
    if complete
    else None,
    "equal_transition_weight": _standardized(
      pairs,
      tuple(m for m in cohort.masses if m in TRANSITION_GRID),
      cohort.lifecycles,
      samples,
      cohort.nominal,
    )
    if complete
    else None,
    "no_failure_increase": failure_gate,
    "no_incomplete_increase": timeout_gate,
    "jointly_completed_displacement_ratio": displacement,
    "walking_preserved": displacement >= 0.95 if displacement is not None else None,
    "pending_gates": [
      "both baseline references",
      "physical limits and fault containment",
      "held-out qualification",
      "training seeds 43 and 44",
    ],
    "uncertainty": "Paired cluster bootstrap within mass; lifecycle repeats share draws. Nominal repeats measure repeatability only; no population CI.",
    "completion_boundary": None,
  }


def _cell(pairs: list[tuple[dict, dict]], samples: int, nominal: bool) -> dict:
  if not pairs:
    return {
      "pairs": 0,
      "status": "not_measured",
      "no_failure_increase": None,
      "no_incomplete_increase": None,
    }
  counts = {}
  for index, arm in enumerate(("reference", "candidate")):
    rows = [pair[index] for pair in pairs]
    counts[arm] = {
      key: sum(row[key] for row in rows)
      for key in ("complete", "combined_failure", "timeout")
    }
    counts[arm]["incomplete"] = sum(not row["complete"] for row in rows)
    counts[arm]["causes"] = {
      name: sum(row.get("terminations", {}).get(name, False) for row in rows)
      for name in ("fell_over", "collapsed", "controller_failed")
    }
  clusters = _clusters(pairs)
  mean, draws = _estimate(clusters, samples, np.random.default_rng(42))
  return {
    "pairs": len(pairs),
    "clusters": len(clusters),
    "status": "measured",
    **counts,
    "completion_gain": mean,
    "completion_gain_ci95": None
    if nominal or len(clusters) < 2
    else np.quantile(draws, [0.025, 0.975]).tolist(),
    "no_failure_increase": counts["candidate"]["combined_failure"]
    <= counts["reference"]["combined_failure"],
    "no_incomplete_increase": counts["candidate"]["incomplete"]
    <= counts["reference"]["incomplete"],
  }


def _clusters(pairs: list[tuple[dict, dict]]) -> list[float]:
  grouped: dict[str, list[float]] = defaultdict(list)
  for b, p in pairs:
    grouped[b["cluster_id"]].append(float(p["complete"]) - float(b["complete"]))
  return [float(np.mean(values)) for _, values in sorted(grouped.items())]


def _estimate(
  values: list[float], samples: int, rng: np.random.Generator
) -> tuple[float, np.ndarray]:
  array = np.asarray(values)
  indices = rng.integers(0, len(values), size=(samples, len(values)))
  return float(array.mean()), array[indices].mean(axis=1)


def _standardized(
  pairs: list[tuple[dict, dict]],
  masses: tuple,
  lifecycles: tuple,
  samples: int,
  nominal: bool,
) -> dict[str, Any] | None:
  if not masses:
    return None
  rng = np.random.default_rng(42)
  means, draws = [], []
  for mass in masses:
    selected = [
      (b, p) for b, p in pairs if b["mass_kg"] == mass and b["lifecycle"] in lifecycles
    ]
    if not selected:
      return None
    mean, draw = _estimate(_clusters(selected), samples, rng)
    means.append(mean)
    draws.append(draw)
  return {
    "completion_gain": float(np.mean(means)),
    "completion_gain_ci95": None
    if nominal
    else np.quantile(np.mean(draws, axis=0), [0.025, 0.975]).tolist(),
    "mass_count": len(masses),
  }
