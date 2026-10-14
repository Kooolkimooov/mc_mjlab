"""Separate input delivery, physical response and safety benefit in force probes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def measure_responses(rows: list[dict[str, Any]], trace_path: Path) -> None:
  """Compare matched Hold samples around failure against repeated-zero variability."""
  frames: dict[int, dict[float, dict[str, Any]]] = {
    index: {} for index in range(len(rows))
  }
  with trace_path.open() as stream:
    for line in stream:
      frame = json.loads(line)
      frames[frame["row"]][round(frame["time_s"], 4)] = frame
  for repeat in sorted({row["repeat"] for row in rows}):
    ids = [i for i, row in enumerate(rows) if row["repeat"] == repeat]
    zeros = [i for i in ids if rows[i]["level"] == 0]
    reference_id = zeros[0]
    baseline = rows[reference_id]
    start = max(0.0, (baseline["failure_time_s"] or baseline["time_s"]) - 5.0)
    responses = {
      index: _response(frames[index], frames[reference_id], start) for index in ids
    }
    target_floor = max(
      responses[index]["target_response_rms_rad"] or 0 for index in zeros
    )
    physical_floor = max(
      responses[index]["cart_response_rms_m"] or 0 for index in zeros
    )
    for index in ids:
      row, response = rows[index], responses[index]
      row.update(response)
      row["response_window_start_s"] = start
      row["zero_target_floor_rad"] = target_floor
      row["zero_cart_floor_m"] = physical_floor
      measured = (
        row["probe_status"] == "measured"
        and row["input_delivery_verified"]
        and response["matched_steps"] > 0
      )
      row["measurable_target_response"] = (
        measured and response["target_response_rms_rad"] > target_floor + 1e-6
      )
      row["measurable_physical_response"] = (
        measured and response["cart_response_rms_m"] > physical_floor + 1e-5
      )
      row["observed_safety_benefit"] = bool(
        measured and baseline["combined_failure"] and row["complete"]
      )


def _response(
  frames: dict[float, dict[str, Any]],
  reference: dict[float, dict[str, Any]],
  start: float,
) -> dict[str, Any]:
  target, physical = [], []
  for time, frame in frames.items():
    other = reference.get(time)
    if (
      time < start
      or other is None
      or frame["phases"] != [4.0, 4.0]
      or other["phases"] != [4.0, 4.0]
    ):
      continue
    target.append(
      np.mean((np.asarray(frame["controller_q"]) - other["controller_q"]) ** 2)
    )
    physical.append(
      np.mean((np.asarray(frame["cart_position_m"]) - other["cart_position_m"]) ** 2)
    )
  return {
    "matched_steps": len(target),
    "target_response_rms_rad": float(np.sqrt(np.mean(target))) if target else None,
    "cart_response_rms_m": float(np.sqrt(np.mean(physical))) if physical else None,
  }
