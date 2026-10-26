"""Joint-group episode summaries of instrumented PD demand."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import torch

from evaluation.locomanip_authority import JOINT_GROUPS
from mc_mjlab.robots.pd_telemetry import FIELDS, InstrumentedPdActuator, PdDemand

GROUPS = ("arms", "waist", "legs", "wrists")
_AT_PEAK = ("p_nm", "d_nm", "feedforward_nm", "time_s", "limit_nm", "kp", "kd")


class PdDemandSummary:
  """Fold each policy window into per-joint episode extrema, read before any reset."""

  def __init__(
    self,
    env: Any,
    rows: list[dict],
    rate_ids: frozenset[str] = frozenset(),
    rate_prefix: Path | None = None,
  ) -> None:
    actuator = next(
      (
        a for a in env.scene["robot"].actuators if isinstance(a, InstrumentedPdActuator)
      ),
      None,
    )
    if actuator is None or actuator.telemetry is None:
      raise ValueError("environment was built without instrument_pd")
    self.telemetry: PdDemand = actuator.telemetry
    self.names = actuator.target_names
    self.columns = {group: _columns(self.names, group) for group in GROUPS}
    self.rows = rows
    zeros = torch.zeros_like(self.telemetry.peak)
    self.episode = {key: zeros.clone() for key in _EPISODE_KEYS}
    self.rate_prefix = rate_prefix
    selected = [row["condition_id"] in rate_ids for row in rows]
    if any(selected) and rate_prefix is None:
      raise ValueError("full-rate PD traces need an output prefix")
    self.rate: dict[int, list[np.ndarray]] = {}
    self.rate_index = {
      row: offset for offset, row in enumerate(np.flatnonzero(selected).tolist())
    }
    if self.rate_index:
      self.telemetry.rate_rows = torch.tensor(selected, device=zeros.device)

  def begin(self, ids: torch.Tensor) -> None:
    """Clear selected rows' episode extrema when their target episode starts."""
    for value in self.episode.values():
      value[ids] = 0
    for i in ids.tolist():
      self.rate.pop(i, None)

  def update(self, active: torch.Tensor) -> dict[str, list]:
    """Fold the just-finished window for active rows; return its group peaks."""
    window = self.telemetry.snapshot()
    episode = self.episode
    rows = active.unsqueeze(1)
    newer = rows & (window["peak_abs_nm"] > episode["peak_abs_nm"])
    episode["peak_abs_nm"] = torch.where(
      newer, window["peak_abs_nm"], episode["peak_abs_nm"]
    )
    for name in _AT_PEAK:
      key = f"at_total_peak/{name}"
      episode[key] = torch.where(newer, window[key], episode[key])
    for key in (
      "independent_p_peak_nm",
      "independent_d_peak_nm",
      "longest_continuous_s",
    ):
      episode[key] = torch.where(
        rows, torch.maximum(episode[key], window[key]), episode[key]
      )
    episode["saturated_s"] += window["saturated_s"] * rows
    self._keep_rate(active)
    frame = {}
    for group, columns in self.columns.items():
      peak = window["peak_abs_nm"][:, columns].amax(dim=1)
      saturated = window["saturated_s"][:, columns].sum(dim=1)
      frame[f"pd_window_peak_nm/{group}"] = peak.cpu().tolist()
      frame[f"pd_window_saturated_s/{group}"] = saturated.cpu().tolist()
    return frame

  def result(self, i: int) -> dict[str, Any]:
    """Per-group episode peak with its contributions, saturation and rate-trace path."""
    episode = {key: value[i].cpu() for key, value in self.episode.items()}
    summary: dict[str, Any] = {}
    for group, columns in self.columns.items():
      peak = episode["peak_abs_nm"][columns]
      joint = columns[int(peak.argmax())]
      limit = episode["at_total_peak/limit_nm"][columns]
      summary[group] = {
        "peak_abs_nm": float(peak.max()),
        "peak_joint": self.names[joint],
        **{
          f"at_peak_{name}": float(episode[f"at_total_peak/{name}"][joint])
          for name in _AT_PEAK
        },
        "peak_over_limit": float((peak / limit).max()),
        "independent_p_peak_nm": float(episode["independent_p_peak_nm"][columns].max()),
        "independent_d_peak_nm": float(episode["independent_d_peak_nm"][columns].max()),
        "saturated_joint_s": float(episode["saturated_s"][columns].sum()),
        "longest_continuous_s": float(episode["longest_continuous_s"][columns].max()),
      }
    summary["rate_trace"] = self._write_rate(i)
    return summary

  def _keep_rate(self, active: torch.Tensor) -> None:
    trace = self.telemetry.drain_rate()
    if trace is None:
      return
    for row, offset in self.rate_index.items():
      if bool(active[row]):
        self.rate.setdefault(row, []).append(trace[:, :, offset].cpu().numpy())

  def _write_rate(self, i: int) -> str | None:
    # values: (substeps, fields, joints) for one target episode.
    samples = self.rate.pop(i, None)
    if not samples or self.rate_prefix is None:
      return None
    stacked = np.concatenate(samples)
    path = Path(f"{self.rate_prefix}-pd-rate-row{i}.npz")
    np.savez_compressed(
      path,
      values=stacked,
      fields=np.array(FIELDS),
      joints=np.array(self.names),
      condition_id=np.array(self.rows[i]["condition_id"]),
    )
    return str(path)


_EPISODE_KEYS = (
  "peak_abs_nm",
  "independent_p_peak_nm",
  "independent_d_peak_nm",
  "saturated_s",
  "longest_continuous_s",
  *(f"at_total_peak/{name}" for name in _AT_PEAK),
)


def _columns(names: list[str], group: str) -> list[int]:
  columns = [
    i for i, name in enumerate(names) if re.fullmatch(JOINT_GROUPS[group], name)
  ]
  if not columns:
    raise ValueError(f"joint group {group!r} matches no instrumented joint")
  return columns
