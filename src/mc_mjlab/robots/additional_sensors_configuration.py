"""RL-only sensors added to the mc_rtc robot MJCFs.

The mc_mujoco XMLs ship the force/IMU sensors the stabilizer needs, but not the
sole velocimeters and root angular-momentum sensor the RL rewards read. They are
added here, uniformly, keyed on frames every robot already has: the ``lf_force``
/ ``rf_force`` sole sites the foot force sensors sit on, and the floating-base
body.
"""

from __future__ import annotations

import mujoco


def add_locomotion_sensors(
  spec: mujoco.MjSpec,
  *,
  root_body: str,
  left_foot_site: str = "lf_force",
  right_foot_site: str = "rf_force",
) -> None:
  """Add sole velocimeters and a root subtree-angular-momentum sensor.

  - ``left_foot_lin_vel`` / ``right_foot_lin_vel``: sole linear velocity, for
    impact-velocity rewards. The force-sensor sites already sit at the soles, so
    this needs no new sites -- and those frames are the ones mc_rtc's wrenches
    are expressed in.
  - ``root_angmom``: subtree angular momentum about the floating-base body.

  Idempotent: any sensor name already present is left as-is.
  """
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
