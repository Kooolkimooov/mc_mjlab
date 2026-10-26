"""One log format for simulator traces and future robot recordings."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Units every channel is stored in; a recording must convert before saving.
UNITS = {
  "time": "s",
  "joint_position": "rad",
  "joint_velocity": "rad/s",
  "joint_effort": "N m",
  "imu_angular_velocity": "rad/s",
  "imu_linear_acceleration": "m/s^2",
  "contact_force": "N",
}
REQUIRED = ("time_s", "target_position_rad", "encoder_position_rad")
OPTIONAL = (
  "encoder_velocity_rad_s",
  "link_position_rad",
  "effort_nm",
  "imu_angular_velocity_rad_s",
  "imu_linear_acceleration_m_s2",
  "contact_force_n",
)


@dataclass
class IdentificationLog:
  """Time-major channels with explicit joint order, frames and provenance."""

  joint_names: tuple[str, ...]
  source: str
  frames: dict[str, str]
  channels: dict[str, np.ndarray]
  metadata: dict[str, str] = field(default_factory=dict)

  def validate(self) -> None:
    """Reject missing, misshaped, non-finite or non-monotonic data."""
    missing = [name for name in REQUIRED if name not in self.channels]
    unknown = set(self.channels) - set(REQUIRED) - set(OPTIONAL)
    if missing or unknown:
      raise ValueError(f"missing channels {missing}, unknown {sorted(unknown)}")
    time = self.channels["time_s"]
    if time.ndim != 1 or len(time) < 2 or not np.all(np.diff(time) > 0):
      raise ValueError("time_s must be strictly increasing")
    for name, values in self.channels.items():
      if len(values) != len(time) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite with one row per timestamp")
      joint_channel = name.startswith(("target_", "encoder_", "link_", "effort"))
      if joint_channel and values.shape[1:] != (len(self.joint_names),):
        raise ValueError(f"{name} columns must follow joint_names")
    if "imu_angular_velocity_rad_s" in self.channels and "imu" not in self.frames:
      raise ValueError("IMU channels need their frame declared")

  def save(self, path: Path) -> None:
    """Write arrays to ``.npz`` and everything else to a JSON header inside it."""
    self.validate()
    header = asdict(self) | {"channels": sorted(self.channels), "units": UNITS}
    arrays: dict[str, Any] = {"header": np.array(json.dumps(header)), **self.channels}
    np.savez_compressed(path, allow_pickle=False, **arrays)

  @classmethod
  def load(cls, path: Path) -> IdentificationLog:
    """Read and validate a saved log."""
    with np.load(path) as data:
      header = json.loads(str(data["header"]))
      channels = {name: data[name] for name in header["channels"]}
    log = cls(
      joint_names=tuple(header["joint_names"]),
      source=header["source"],
      frames=header["frames"],
      channels=channels,
      metadata=header["metadata"],
    )
    log.validate()
    return log

  def segments(self, length_s: float) -> list[slice]:
    """Consecutive whole windows for fit/held-out splits."""
    time = self.channels["time_s"]
    dt = float(np.median(np.diff(time)))
    size = int(round(length_s / dt))
    return [slice(i, i + size) for i in range(0, len(time) - size + 1, size)]
