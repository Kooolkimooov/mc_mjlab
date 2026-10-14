"""Read selected force and controller signals from one mc_rtc native log."""

from __future__ import annotations

import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def _read_size(stream: Any) -> int:
  """Read the native flat-log size field."""
  return struct.unpack("Q", stream.read(8))[0]


def _read_string(stream: Any) -> str:
  """Read one length-prefixed flat-log string."""
  return stream.read(_read_size(stream)).decode("ascii")


def native_channels(path: Path) -> dict[str, Any]:
  """Convert a native log and retain only the diagnostic arrays needed here."""
  with tempfile.TemporaryDirectory(prefix="mc_mjlab_native_") as directory:
    flat = Path(directory) / "trace.flat"
    subprocess.run(["mc_bin_to_flat", str(path), str(flat)], check=True)
    selected = {
      "t",
      "FloatingBase_position_z",
      "ManipManager_manipPhase_Left",
      "ManipManager_manipPhase_Right",
    }
    for side in ("Left", "Right"):
      for kind in ("measuredWrench", "targetWrench"):
        selected.update(f"HandTask_{side}_{kind}_f{axis}" for axis in "xyz")
    data: dict[str, Any] = {}
    with flat.open("rb") as stream:
      for _ in range(_read_size(stream)):
        numeric = struct.unpack("?", stream.read(1))[0]
        name = _read_string(stream)
        length = _read_size(stream)
        if numeric:
          if name in selected:
            data[name] = np.frombuffer(stream.read(length * 8), dtype=np.float64).copy()
          else:
            stream.seek(length * 8, 1)
        else:
          values = [_read_string(stream) for _ in range(length)]
          if name in selected:
            data[name] = values
  return data


def summarize_native(path: Path) -> dict[str, Any]:
  """Report actual and commanded hand forces over the last five Hold seconds."""
  data = native_channels(path)
  if not data or "t" not in data or len(data["t"]) == 0:
    return {"log": str(path), "samples": 0}
  time = data["t"]
  hold = np.asarray(data["ManipManager_manipPhase_Left"]) == "Hold"
  hold &= np.asarray(data["ManipManager_manipPhase_Right"]) == "Hold"
  last_hold = time[hold][-1] if hold.any() else time[-1]
  window = hold & (time >= last_hold - 5.0)
  result = {
    "log": str(path),
    "samples": len(time),
    "end_s": float(time[-1]),
    "min_root_z_m": float(np.min(data["FloatingBase_position_z"])),
    "hold_samples_last_5s": int(window.sum()),
    "last_hold_s": float(last_hold),
    "hands": {},
  }
  for side in ("Left", "Right"):
    measured = np.stack(
      [data[f"HandTask_{side}_measuredWrench_f{axis}"] for axis in "xyz"], axis=1
    )
    commanded = np.stack(
      [data[f"HandTask_{side}_targetWrench_f{axis}"] for axis in "xyz"], axis=1
    )
    if window.any():
      result["hands"][side.lower()] = {
        "measured_mean_n": measured[window].mean(axis=0).tolist(),
        "commanded_mean_n": commanded[window].mean(axis=0).tolist(),
        "tracking_error_rms_n": float(
          np.sqrt(np.mean((measured[window] - commanded[window]) ** 2))
        ),
      }
  return result
