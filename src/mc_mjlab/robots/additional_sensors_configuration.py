"""RL-only sensors added to the mc_rtc robot MJCFs."""

from __future__ import annotations

import mujoco


def add_locomotion_sensors(
  spec: mujoco.MjSpec,
  *,
  root_body: str,
  left_foot_site: str = "lf_force",
  right_foot_site: str = "rf_force",
) -> None:
  """Add sole velocimeters and a root subtree-angular-momentum sensor."""
  existing = {sensor.name for sensor in spec.sensors}
  for name, site in (
    ("left_foot_lin_vel", left_foot_site),
    ("right_foot_lin_vel", right_foot_site),
  ):
    if name not in existing:
      spec.add_sensor(
        name=name,
        type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        objname=site,
      )
  if "root_angmom" not in existing:
    spec.add_sensor(
      name="root_angmom",
      type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
      objtype=mujoco.mjtObj.mjOBJ_BODY,
      objname=root_body,
    )
