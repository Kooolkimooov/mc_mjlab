"""Per-robot data read from the mc_rtc ``RobotModule``, lazily and once.

The controller's robot module owns the ground truth for refJointOrder, the
half-sitting stance, the default floating-base attitude, and (via ``bounds``)
the nominal torque limits. Reading them here keeps the mjlab side from carrying
hand-copied transcriptions that drift from the robot the controller actually
drives.

Everything is lazy and cached on purpose: importing this module -- and, through
it, the registry and the per-robot constants -- must not require a sourced
mc_rtc workspace, and must not pay the module's construction cost until
something actually needs controller-derived data.

Binding quirk worth knowing: ``stance()`` keys come back as ``bytes`` while
``bounds()`` keys come back as ``str``; ``_decode_joint_key`` normalises both.
"""

from __future__ import annotations

import functools
from collections.abc import Iterable


@functools.lru_cache(maxsize=None)
def get_robot_module(name: str):
  """The mc_rtc ``RobotModule`` for ``name`` (e.g. ``"RHPS1_MuJoCo"``)."""
  import mc_rbdyn

  return mc_rbdyn.get_robot_module(name)


def _decode_joint_key(key: object) -> str:
  return key.decode() if isinstance(key, (bytes, bytearray)) else str(key)


def get_ref_joint_order(name: str) -> tuple[str, ...]:
  """The module's refJointOrder -- the default joint set every robot drives."""
  return tuple(get_robot_module(name).ref_joint_order())


def get_actuated_joints(
  name: str, *, non_actuated: Iterable[str] = ()
) -> tuple[str, ...]:
  """refJointOrder minus the deactivated joints: the joints to give actuators."""
  excluded = set(non_actuated)
  return tuple(j for j in get_ref_joint_order(name) if j not in excluded)


def get_residual_joints(
  name: str,
  *,
  non_actuated: Iterable[str] = (),
  non_residual: Iterable[str] = (),
) -> tuple[str, ...]:
  """The actuated joints minus those excluded from the RL residual."""
  excluded = set(non_actuated) | set(non_residual)
  return tuple(j for j in get_ref_joint_order(name) if j not in excluded)


def get_default_root_position(name: str) -> tuple[float, float, float]:
  """Floating-base ``(x, y, z)`` from the module's default attitude."""
  x, y, z = (float(v) for v in get_robot_module(name).default_attitude()[-3:])
  return (x, y, z)


def get_default_joint_positions(
  name: str, joints: Iterable[str] | None = None, *, drop_zeros: bool = True
) -> dict[str, float]:
  """The half-sitting stance as ``{joint: angle}`` over 1-DoF joints."""
  keep = None if joints is None else set(joints)
  out: dict[str, float] = {}
  for key, value in get_robot_module(name).stance().items():
    if len(value) != 1:  # passive linkage / multi-DoF / camera-frame joints
      continue
    joint = _decode_joint_key(key)
    if keep is not None and joint not in keep:
      continue
    angle = float(value[0])
    if drop_zeros and angle == 0.0:
      continue
    out[joint] = angle
  return out


def get_effort_limits(name: str) -> dict[str, float]:
  """Per-joint nominal torque limits (upper tau bound), 1-DoF joints only."""
  return {
    _decode_joint_key(key): float(value[0])
    for key, value in get_robot_module(name).bounds()[5].items()
    if len(value) == 1
  }
