"""Arm-by-mass outcomes and window contrasts against the matched zero policy."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from evaluation.phase_windows import GROUPS

WINDOW_METRICS = ("max_tilt_rad", "max_cart_error_m", "min_height_m", "displacement_m")
GROUP_METRICS = ("max_executed_rad", "mean_tracking_error_rad", "saturated_joint_s")


def outcome_label(episode: dict) -> str:
  """C complete, T timeout, F tilt fall, X collapse, E other controller failure."""
  if episode["complete"]:
    return "C"
  if "fell_over" in episode["terminations"]:
    return "F"
  if "collapsed" in episode["terminations"]:
    return "X"
  if episode["combined_failure"]:
    return "E"
  return "T" if episode["timeout"] else "?"


def outcome_matrix(phases: list[dict]) -> dict[str, dict[str, str]]:
  """Per arm, one label string per mass in fresh-then-recycled order."""
  cells: dict[str, dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
  for episode in phases:
    mass = f"{episode['mass_kg']:g}"
    cells[episode["arm"]][mass][episode["lifecycle"]] = outcome_label(episode)
  return {
    arm: {
      mass: by_life.get("fresh", "-") + by_life.get("recycled", "-")
      for mass, by_life in sorted(masses.items(), key=lambda item: float(item[0]))
    }
    for arm, masses in cells.items()
  }


def zero_contrasts(phases: list[dict], zero: str = "zero") -> dict[str, Any]:
  """Mean paired differences from the zero arm on identical conditions."""
  reference = {e["condition_id"]: e for e in phases if e["arm"] == zero}
  deltas: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
  for episode in phases:
    base = reference.get(episode["condition_id"])
    if episode["arm"] == zero or base is None:
      continue
    arm = deltas[episode["arm"]]
    arm["completion"].append(float(episode["complete"]) - float(base["complete"]))
    arm["failure"].append(
      float(episode["combined_failure"]) - float(base["combined_failure"])
    )
    arm["end_time_s"].append(
      episode["event_time_s"]["terminal"] - base["event_time_s"]["terminal"]
    )
    for window in ("sustained_hold", "operating", "fall_transient"):
      ours, theirs = episode["windows"].get(window), base["windows"].get(window)
      if ours is None or theirs is None:
        continue
      for metric in WINDOW_METRICS:
        arm[f"{window}/{metric}"].append(ours[metric] - theirs[metric])
      for group in GROUPS:
        for metric in GROUP_METRICS:
          a, b = ours[group][metric], theirs[group][metric]
          if a is not None and b is not None:
            arm[f"{window}/{group}/{metric}"].append(a - b)
  return {
    arm: {key: {"mean": float(np.mean(v)), "n": len(v)} for key, v in values.items()}
    for arm, values in deltas.items()
  }


def event_timing(phases: list[dict]) -> dict[str, dict[str, Any]]:
  """Median event times per arm and how often each event was reached."""
  times: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
  counts: dict[str, int] = defaultdict(int)
  for episode in phases:
    counts[episode["arm"]] += 1
    for name, value in episode["event_time_s"].items():
      if value is not None:
        times[episode["arm"]][name].append(value)
  return {
    arm: {
      name: {"median_s": float(np.median(v)), "reached": len(v), "of": counts[arm]}
      for name, v in events.items()
    }
    for arm, events in times.items()
  }
