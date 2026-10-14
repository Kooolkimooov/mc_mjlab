"""Paired safety estimates with protection for every baseline-safe mass."""

from __future__ import annotations

from typing import Any, TypeGuard

import numpy as np

from evaluation.comparison import wilson
from evaluation.locomanip_inputs import condition_id
from evaluation.locomanip_trace import FAILURES


def summarize_safety(
  rows: list[dict[str, Any]],
  masses: list[float],
  transition_masses: list[float],
  seeds: list[int],
  repetitions: int,
  bootstrap_samples: int = 10000,
) -> dict[str, Any]:
  """Require complete matched cohorts, pooled improvement and zero safe-mass regressions."""
  if not masses or not transition_masses or not set(transition_masses) <= set(masses):
    raise ValueError("transition grid must be a nonempty subset of the evaluation grid")
  if repetitions < 2 or not seeds or len(set(seeds)) != len(seeds):
    raise ValueError("use at least two repetitions and unique explicit seeds")
  if len(set(masses)) != len(masses) or len(set(transition_masses)) != len(
    transition_masses
  ):
    raise ValueError("mass grids must contain unique entries")
  indexed = {}
  for row in rows:
    key = (row["condition_id"], row["arm"])
    if key in indexed:
      raise ValueError(f"duplicate condition arm: {key}")
    indexed[key] = row
  per_mass = []
  pairs_by_mass = []
  protected_ok = True
  complete_data = True
  for mass in masses:
    expected = [
      condition_id(seed, mass, repeat)
      for seed in seeds
      for repeat in range(repetitions)
    ]
    baseline = [indexed.get((identifier, "baseline")) for identifier in expected]
    policy = [indexed.get((identifier, "policy")) for identifier in expected]
    valid_pairs = [
      (b, p) for b, p in zip(baseline, policy, strict=True) if _usable(b) and _usable(p)
    ]
    complete = len(valid_pairs) == len(expected)
    complete_data &= complete
    safe = all(_safe(row) for row in baseline)
    protection = not safe or all(_safe(row) for row in policy)
    protected_ok &= protection
    entry = {
      "mass_kg": mass,
      "expected_pairs": len(expected),
      "usable_pairs": len(valid_pairs),
      "baseline": _counts(baseline),
      "policy": _counts(policy),
      "baseline_safe": safe,
      "protection_passed": protection,
    }
    per_mass.append(entry)
    if mass in transition_masses:
      pairs_by_mass.append(
        np.asarray(
          [[b["combined_failure"], p["combined_failure"]] for b, p in valid_pairs],
          dtype=float,
        ).reshape(-1, 2)
      )
  pooled = _paired_interval(pairs_by_mass, bootstrap_samples) if complete_data else None
  improvement_ok = (
    pooled is not None
    and pooled["relative_reduction"] is not None
    and pooled["relative_reduction"] >= 0.2
    and pooled["absolute_reduction_ci95"][0] > 0
  )
  return {
    "promote": bool(complete_data and protected_ok and improvement_ok),
    "complete_matched_data": complete_data,
    "per_mass_protection": protected_ok,
    "pooled_improvement": bool(improvement_ok),
    "pooled": pooled,
    "per_mass": per_mass,
    "transition_grid_kg": transition_masses,
    "evaluation_grid_kg": masses,
    "seeds": seeds,
    "pairs_per_mass_per_seed": repetitions,
    "uncertainty": "95% paired environment bootstrap, stratified by mass; equal mass weights. Zero observed failures remains a finite-sample result.",
  }


def _usable(row: dict[str, Any] | None) -> TypeGuard[dict[str, Any]]:
  return (
    row is not None
    and row.get("observed", False)
    and not row.get("worker_failure", True)
    and (row.get("complete", False) or row.get("combined_failure", False))
  )


def _safe(row: dict[str, Any] | None) -> bool:
  return bool(
    _usable(row)
    and row is not None
    and row.get("complete", False)
    and not row.get("combined_failure", True)
  )


def _counts(rows: list[dict[str, Any] | None]) -> dict[str, Any]:
  observed = [
    row
    for row in rows
    if row is not None
    and row.get("observed", False)
    and not row.get("worker_failure", True)
  ]
  usable = [row for row in rows if _usable(row)]
  failures = sum(row["combined_failure"] for row in usable)
  return {
    "requested": len(rows),
    "observed": len(observed),
    "usable": len(usable),
    "missing_or_infrastructure": len(rows) - len(observed),
    "combined_failures": failures,
    "combined_failure_ci95": wilson(failures, len(usable)) if usable else None,
    "incomplete": sum(
      not row["complete"] and not row["combined_failure"] for row in observed
    ),
    "complete": sum(_safe(row) for row in rows),
    "causes": {
      name: sum(row["terminations"].get(name, False) for row in usable)
      for name in FAILURES
    },
    "worker_failures": sum(
      row is not None and row.get("worker_failure", False) for row in rows
    ),
  }


def _paired_interval(pairs: list[np.ndarray], samples: int) -> dict[str, Any]:
  rng = np.random.default_rng(42)
  means = np.asarray([values.mean(axis=0) for values in pairs]).mean(axis=0)
  draws = np.zeros((samples, 2))
  for values in pairs:
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    draws += values[indices].mean(axis=1) / len(pairs)
  delta = draws[:, 0] - draws[:, 1]
  return {
    "baseline_failure_rate": float(means[0]),
    "policy_failure_rate": float(means[1]),
    "relative_reduction": float((means[0] - means[1]) / means[0])
    if means[0] > 0
    else None,
    "absolute_reduction": float(means[0] - means[1]),
    "absolute_reduction_ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
    "bootstrap_samples": samples,
  }
