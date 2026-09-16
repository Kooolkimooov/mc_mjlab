"""Lookup of compiled MuJoCo wrench sensor addresses."""

from __future__ import annotations

import mujoco


def wrench_sensor(
  mj_model: mujoco.MjModel, suffix: str, sensor_type: int
) -> tuple[int, int]:
  """``(sensordata offset, site id)`` of the model sensor named ``*suffix``."""
  for i in range(mj_model.nsensor):
    sensor = mj_model.sensor(i)
    if sensor.name.endswith(suffix) and int(sensor.type[0]) == sensor_type:
      return int(sensor.adr[0]), int(sensor.objid[0])
  raise ValueError(f"missing MuJoCo sensor '*{suffix}' of type {sensor_type}")
