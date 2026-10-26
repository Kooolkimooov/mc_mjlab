"""Replay matched study traces under the current and the alive-only objective."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

DCM_TERM = "contact_dcm_stability"
TERMINATION_TERM = "termination_penalty"
#: The one admissible replacement; every other term keeps its weight.
ALIVE_WEIGHT = 1.0
BOOTSTRAP_WINDOW_S = 1.0
#: Exploration costs, separable from task terms; the candidate keeps them.
COST_TERMS = ("residual_magnitude", "residual_rate")
#: (name, DCM swapped for alive, exploration costs dropped); "alive" is the candidate.
OBJECTIVES = (
  ("current", False, False),
  ("alive", True, False),
  ("current_without_costs", False, True),
  ("alive_without_costs", True, True),
)
ALTERNATIVES = ("failure", "low_motion")
MEASURES = (
  ("undiscounted", None, None),
  ("start/discounted", "start", "discounted"),
  ("start/bootstrapped", "start", "bootstrapped"),
  ("hold_entry/discounted", "hold_entry", "discounted"),
  ("hold_entry/bootstrapped", "hold_entry", "bootstrapped"),
)


def category(record: dict) -> str:
  """Safe success, physical/controller failure, or survival without completion."""
  if record["combined_failure"]:
    return "failure"
  if record["complete"]:
    return "safe_success"
  return "low_motion" if record["timeout"] else "other"


def reward_matrix(
  steps: list[dict], terms: list[str], alive: bool, drop_costs: bool
) -> tuple[np.ndarray, list[str]]:
  """Weighted per-step rewards, optionally alive for DCM and without costs."""
  values = np.array([[step[f"reward/{t}"] for t in terms] for step in steps], float)
  names = list(terms)
  if alive:
    terminated = values[:, terms.index(TERMINATION_TERM)] != 0.0
    values[:, terms.index(DCM_TERM)] = ALIVE_WEIGHT * (~terminated)
    names[terms.index(DCM_TERM)] = "is_alive"
  if drop_costs:
    values[:, [terms.index(t) for t in COST_TERMS if t in terms]] = 0.0
  return values, names


def episode_returns(
  steps: list[dict],
  record: dict,
  terms: list[str],
  dt: float,
  gamma: float,
  hold_entry: int | None,
) -> dict[str, Any]:
  """Totals per term and discounted returns from start and Hold entry."""
  result: dict[str, Any] = {}
  for objective, alive, drop_costs in OBJECTIVES:
    values, names = reward_matrix(steps, terms, alive, drop_costs)
    per_step = values.sum(axis=1) * dt
    truncated = not record["combined_failure"]
    tail = max(1, round(BOOTSTRAP_WINDOW_S / dt))
    bootstrap = float(per_step[-tail:].mean() / (1.0 - gamma)) if truncated else 0.0
    starts = {"start": 0, "hold_entry": hold_entry}
    returns = {}
    for label, start in starts.items():
      if start is None:
        returns[label] = None
        continue
      rewards = per_step[start:]
      discount = gamma ** np.arange(len(rewards))
      plain = float((discount * rewards).sum())
      returns[label] = {
        "discounted": plain,
        "bootstrapped": plain + gamma ** len(rewards) * bootstrap,
      }
    result[objective] = {
      "undiscounted": float(per_step.sum()),
      "per_second": float(per_step.sum() / (len(per_step) * dt)),
      "terms": {name: float(values[:, i].sum() * dt) for i, name in enumerate(names)},
      "returns": returns,
    }
  return result


def ranking_defects(episodes: list[dict]) -> dict[str, Any]:
  """Matched pairs where a failure or low-motion run scores at least a success."""
  by_condition: dict[str, list[dict]] = defaultdict(list)
  for episode in episodes:
    by_condition[episode["condition_id"]].append(episode)
  report: dict[str, Any] = {}
  for objective, _, _ in OBJECTIVES:
    counts: dict[str, dict[str, int]] = {
      f"{alt}/{label}": {"pairs": 0, "defects": 0}
      for alt in ALTERNATIVES
      for label, _, _ in MEASURES
    }
    for group in by_condition.values():
      successes = [e for e in group if e["category"] == "safe_success"]
      for alternative in (e for e in group if e["category"] in ALTERNATIVES):
        for success in successes:
          for label, start, kind in MEASURES:
            a = _score(alternative, objective, start, kind)
            s = _score(success, objective, start, kind)
            if a is None or s is None:
              continue
            cell = counts[f"{alternative['category']}/{label}"]
            cell["pairs"] += 1
            cell["defects"] += int(a >= s)
    report[objective] = counts
  return report


def verdict(defects: dict[str, Any]) -> dict[str, Any]:
  """Admit only a consistent current defect that the alive objective removes."""
  consistent = [
    key
    for key, cell in defects["current"].items()
    if cell["pairs"] and 2 * cell["defects"] > cell["pairs"]
  ]
  remaining = [key for key, cell in defects["alive"].items() if cell["defects"]]
  if not consistent:
    decision = "reject: no consistent ranking defect under the current objective"
  elif remaining:
    decision = "reject: the alive objective leaves ranking defects"
  else:
    decision = "admit: alive objective ranks every safe success first"
  return {
    "decision": decision,
    "consistent_defects": consistent,
    "alive_defects": remaining,
  }


def _score(
  episode: dict, objective: str, start: str | None, kind: str | None
) -> float | None:
  scores = episode["objectives"][objective]
  if start is None:
    return scores["undiscounted"]
  returns = scores["returns"][start]
  return None if returns is None else returns[kind]
