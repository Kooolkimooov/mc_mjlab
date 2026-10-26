"""Fit per-joint delay and lag on a PD rate trace and predict its held-out half."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from evaluation.study_io import write_json

from mc_mjlab.identification.fits import fit_delay_lag, held_out_error
from mc_mjlab.identification.log import IdentificationLog


def main() -> None:
  """Convert the trace to the shared log, then fit and cross-predict each joint."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("trace", type=Path, help="--pd-rate-trace .npz file")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--joints", default="", help="comma-separated subset")
  parser.add_argument("--max-delay-s", type=float, default=0.03)
  args = parser.parse_args()

  log = rate_trace_log(args.trace)
  log.save(args.output.with_suffix(".log.npz"))
  time = log.channels["time_s"]
  dt = float(np.median(np.diff(time)))
  half = len(time) // 2
  names = [n for n in args.joints.split(",") if n] or list(log.joint_names)
  joints = {}
  for name in names:
    column = log.joint_names.index(name)
    command = log.channels["target_position_rad"][:, column]
    response = log.channels["encoder_position_rad"][:, column]
    if np.ptp(command[:half]) == 0:
      joints[name] = {"status": "not_measured", "reason": "no command excitation"}
      continue
    fit = fit_delay_lag(command[:half], response[:half], dt, args.max_delay_s)
    joints[name] = {
      "delay_s": fit.delay_s,
      "tau_s": fit.tau_s,
      "fit_rmse_rad": fit.rmse,
      "fit_r2": fit.r2,
      "held_out_rmse_rad": held_out_error(fit, command[half:], response[half:], dt),
    }
    print(f"{name}: {json.dumps(joints[name])}", flush=True)
  write_json(
    args.output,
    {"trace": str(args.trace), "source": log.source, "dt_s": dt, "joints": joints},
  )


def rate_trace_log(path: Path) -> IdentificationLog:
  """The study's full-rate PD samples as a simulator identification log."""
  with np.load(path) as data:
    fields = [str(f) for f in data["fields"]]
    values = data["values"]
    joints = tuple(str(j) for j in data["joints"])
    condition = str(data["condition_id"])

  def channel(name: str) -> np.ndarray:
    return values[:, fields.index(name)].astype(float)

  # float32 stamps lose the 1 ms step late in an episode; read it near zero.
  dt = float(np.round(np.median(np.diff(channel("time_s")[:1000, 0])), 7))
  return IdentificationLog(
    joint_names=joints,
    source="simulator",
    frames={"joint": "simulated joint; no motor/link split"},
    channels={
      "time_s": np.arange(len(values)) * dt,
      "target_position_rad": channel("position_target_rad"),
      "encoder_position_rad": channel("position_rad"),
      "encoder_velocity_rad_s": channel("velocity_rad_s"),
      "effort_nm": channel("clipped_nm"),
    },
    metadata={"condition_id": condition, "trace": str(path)},
  )


if __name__ == "__main__":
  main()
