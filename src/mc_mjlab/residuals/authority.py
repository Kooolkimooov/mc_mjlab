"""Per-joint residual authority derived from the robot's own hardware limits."""

from __future__ import annotations

import re
from pathlib import Path

from mc_mjlab.robots import robot_module as mc_rtc

#: Share of a joint's torque capacity a residual may command. docs/residual-authority.md
TORQUE_FRACTION = 0.2


def reference_stiffness(robot_name: str, pd_gains_path: Path) -> dict[str, float]:
  """Read the position gains in ``refJointOrder`` order."""
  rows = [
    line.split()
    for line in Path(pd_gains_path).read_text().splitlines()
    if line.strip()
  ]
  return {
    joint: float(row[0])
    for joint, row in zip(mc_rtc.get_ref_joint_order(robot_name), rows, strict=True)
  }


def hardware_residual_scales(
  robot_name: str,
  control: str,
  residual_joints: tuple[str, ...],
  pd_gains_path: Path,
  actuated_joints: tuple[str, ...],
  fallback: float,
  cap: float | None = None,
) -> dict[str, float]:
  """Exact per-actuator scales from effort limits and, for position, PD stiffness."""
  limits = mc_rtc.get_effort_limits(robot_name)
  stiffness = reference_stiffness(robot_name, pd_gains_path)
  authority: dict[str, float] = {}

  for joint in residual_joints:
    if joint not in limits:
      raise KeyError(f"no effort limit for residual joint {joint}")
    if control == "position":
      if joint not in stiffness or stiffness[joint] <= 0.0:
        raise KeyError(f"no positive PD stiffness for residual joint {joint}")
      scale = TORQUE_FRACTION * limits[joint] / stiffness[joint]
    else:
      scale = TORQUE_FRACTION * limits[joint]
    authority[joint] = scale if cap is None else min(cap, scale)

  # Must partition the entity's actuators exactly: a missing one silently gets
  # scale 1.0 and no clip, an extra one fails to resolve at all.
  # docs/residual-authority.md#residual_scales
  return {re.escape(joint): authority.get(joint, fallback) for joint in actuated_joints}
