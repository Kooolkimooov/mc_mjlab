"""PD actuator config shared by the mc_rtc robot entities."""

from __future__ import annotations

import math
from collections.abc import Iterable

import mujoco
from mjlab.actuator import IdealPdActuatorCfg

# The PD model every robot uses: gains derived from a target natural frequency
# and damping ratio times the joint's reflected rotor inertia.
NATURAL_FREQ = 3.0 * 2.0 * math.pi  # rad/s
DAMPING_RATIO = 1.5

# Unclamped, like mc_mujoco's PD torque: with the real gains, nominal limits
# would saturate constantly. docs/robots.md#pd_actuator_configuration
EFFORT_LIMIT = float("inf")


def get_armature_from_spec(
  spec: mujoco.MjSpec, joints: Iterable[str]
) -> dict[str, float]:
  """Per-joint reflected rotor inertia, as authored in the MJCF."""
  keep = set(joints)
  return {j.name: float(j.armature) for j in spec.joints if j.name in keep}


def get_pd_actuator_cfgs(
  joints: Iterable[str],
  armature: float | dict[str, float],
  *,
  natural_freq: float = NATURAL_FREQ,
  damping_ratio: float = DAMPING_RATIO,
  effort_limit: float = EFFORT_LIMIT,
) -> tuple[IdealPdActuatorCfg, ...]:
  """One ``IdealPdActuatorCfg`` per joint, gains from the natural-frequency model."""

  def armature_of(name: str) -> float:
    return armature[name] if isinstance(armature, dict) else armature

  return tuple(
    IdealPdActuatorCfg(
      target_names_expr=(name,),
      stiffness=armature_of(name) * natural_freq**2,
      damping=2 * damping_ratio * armature_of(name) * natural_freq,
      effort_limit=effort_limit,
    )
    for name in joints
  )
