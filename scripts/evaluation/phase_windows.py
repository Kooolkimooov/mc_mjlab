"""Event-anchored windows over study traces, keyed by the controller's raw phases."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.locomanip_authority import JOINT_GROUPS
from mc_mjlab.tasks.locomanip.mdp.accessors import HOLD_PHASE

BEFORE_S, AFTER_S = 1.0, 2.0
GROUPS = ("arms", "waist", "legs", "wrists")


def load_traces(paths: list[Path]) -> dict[tuple[str, str], list[dict]]:
  """Per-step lines grouped by (arm, condition) and ordered by episode time."""
  episodes: dict[tuple[str, str], list[dict]] = defaultdict(list)
  for path in paths:
    with path.open() as stream:
      for line in stream:
        step = json.loads(line)
        episodes[(step["arm"], step["condition_id"])].append(step)
  for steps in episodes.values():
    steps.sort(key=lambda step: step["time_s"])
  return dict(episodes)


def find_events(steps: list[dict]) -> dict[str, int | None]:
  """First fresh bilateral Hold, the first fresh step leaving it, and the last step."""
  holding = [
    step["phase_left"] == HOLD_PHASE
    and step["phase_right"] == HOLD_PHASE
    and bool(step["datastore_fresh"])
    for step in steps
  ]
  entry = next((i for i, held in enumerate(holding) if held), None)
  release = None
  if entry is not None:
    release = next(
      (
        i
        for i in range(entry + 1, len(steps))
        if bool(steps[i]["datastore_fresh"]) and not holding[i]
      ),
      None,
    )
  return {"hold_entry": entry, "release": release, "terminal": len(steps) - 1}


def segment_bounds(
  steps: list[dict], events: dict[str, int | None], failed: bool
) -> dict[str, tuple[int, int]]:
  """Inclusive index ranges; operating and fall-transient split one second out."""
  times = np.array([step["time_s"] for step in steps])
  terminal = len(steps) - 1

  def at(time: float) -> int:
    return int(np.clip(np.searchsorted(times, time), 0, terminal))

  bounds: dict[str, tuple[int, int]] = {}
  entry, release = events["hold_entry"], events["release"]
  if entry is not None:
    bounds["pre_hold"] = (0, max(entry - 1, 0))
    bounds["hold_entry"] = (at(times[entry] - BEFORE_S), at(times[entry] + AFTER_S))
    end = release - 1 if release is not None else terminal
    start = at(times[entry] + AFTER_S)
    if start <= end:
      bounds["sustained_hold"] = (start, end)
  else:
    bounds["pre_hold"] = (0, terminal)
  if release is not None:
    bounds["release"] = (at(times[release] - BEFORE_S), at(times[release] + AFTER_S))
  onset = at(times[terminal] - BEFORE_S)
  bounds["terminal"] = (onset, terminal)
  if failed and onset > 0:
    bounds["operating"] = (0, onset - 1)
    bounds["fall_transient"] = (onset, terminal)
  else:
    bounds["operating"] = (0, terminal)
  return bounds


def window_summary(
  steps: list[dict], lo: int, hi: int, names: dict[str, list[str]], dt: float
) -> dict[str, Any]:
  """Motion, contact, delivery, tracking, PD demand and reward over one window."""
  window = steps[lo : hi + 1]

  def column(key: str) -> np.ndarray:
    return np.array([step[key] for step in window], dtype=float)

  root = column("root_position_m")
  hands = column("hand_force_n").reshape(len(window), -1, 3)
  joint_error = np.abs(
    column("joint_position_target_rad") - column("joint_position_rad")
  )
  summary: dict[str, Any] = {
    "start_s": window[0]["time_s"],
    "end_s": window[-1]["time_s"],
    "steps": len(window),
    "fresh_fraction": float(column("datastore_fresh").mean()),
    "raw_phases": sorted({(s["phase_left"], s["phase_right"]) for s in window}),
    "min_height_m": float(root[:, 2].min()),
    "max_tilt_rad": float(column("tilt_rad").max()),
    "displacement_m": float(root[-1, 0] - root[0, 0]),
    "max_cart_error_m": float(np.linalg.norm(column("cart_error_m"), axis=1).max()),
    "max_hand_force_n": np.linalg.norm(hands, axis=2).max(axis=0).tolist(),
    "min_foot_load_kn": float(column("foot_force_kn").sum(axis=1).min()),
  }
  executed = np.abs(column("executed_residual_rad"))
  for group in GROUPS:
    residual = _columns(names["residual"], group)
    joints = _columns(names["joints"], group)
    summary[group] = {
      "max_executed_rad": float(executed[:, residual].max()) if residual else None,
      "mean_tracking_error_rad": float(joint_error[:, joints].mean()),
      "max_pd_peak_nm": float(column(f"pd_window_peak_nm/{group}").max()),
      "saturated_joint_s": float(column(f"pd_window_saturated_s/{group}").sum()),
    }
  summary["reward"] = {
    name: float(column(f"reward/{name}").sum() * dt) for name in names["rewards"]
  }
  return summary


def analyze_episode(
  steps: list[dict], record: dict, names: dict[str, list[str]], dt: float
) -> dict[str, Any]:
  """Events, every bounded window and the measured outcome of one episode."""
  events = find_events(steps)
  bounds = segment_bounds(steps, events, bool(record["combined_failure"]))
  return {
    key: record[key]
    for key in ("arm", "condition_id", "mass_kg", "lifecycle", "complete", "timeout")
  } | {
    "combined_failure": record["combined_failure"],
    "terminations": [name for name, hit in record["terminations"].items() if hit],
    "event_time_s": {
      name: None if index is None else steps[index]["time_s"]
      for name, index in events.items()
    },
    "windows": {
      name: window_summary(steps, lo, hi, names, dt)
      for name, (lo, hi) in bounds.items()
    },
  }


def _columns(names: list[str], group: str) -> list[int]:
  return [i for i, name in enumerate(names) if re.fullmatch(JOINT_GROUPS[group], name)]
