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

import dataclasses
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


##
# Anatomical joint subsets, read off the module's kinematic tree.
##

# mc_rtc's standard sensor names. The stabilizer is written against these, so a
# humanoid module that runs a walking controller has them by construction --
# which is what makes limbs derivable here without a per-robot table of body
# names. `get_limb_bodies` raises if one is missing rather than guessing.
FLOATING_BASE_SENSORS = ("FloatingBase", "Accelerometer")
FOOT_FORCE_SENSOR = {"left": "LeftFootForceSensor", "right": "RightFootForceSensor"}
HAND_FORCE_SENSOR = {"left": "LeftHandForceSensor", "right": "RightHandForceSensor"}

_SIDES = ("left", "right")


@dataclasses.dataclass(frozen=True)
class _Tree:
  """The module's ``MultiBody``, decoded once into plain Python.

  Built from ``joints()``/``bodies()`` rather than ``jointIndexByName``: that
  binding throws a C++ ``std::out_of_range`` on a missing name which terminates
  the process uncatchably (the same trap ``ControllerHost.joint_index`` avoids).
  """

  joints: tuple[str, ...]
  dofs: tuple[int, ...]
  bodies: tuple[str, ...]
  predecessor: tuple[int, ...]  # parent body index, per joint
  successor: tuple[int, ...]  # child body index, per joint
  body_index: dict[str, int]
  joint_of_body: dict[int, int]  # child body index -> the joint that drives it


@functools.lru_cache(maxsize=None)
def _get_tree(name: str) -> _Tree:
  mb = get_robot_module(name).mb
  joints = tuple(_decode_joint_key(j.name()) for j in mb.joints())
  dofs = tuple(int(j.dof()) for j in mb.joints())
  bodies = tuple(_decode_joint_key(b.name()) for b in mb.bodies())
  predecessor = tuple(mb.predecessor(i) for i in range(mb.nrJoints()))
  successor = tuple(mb.successor(i) for i in range(mb.nrJoints()))
  return _Tree(
    joints=joints,
    dofs=dofs,
    bodies=bodies,
    predecessor=predecessor,
    successor=successor,
    body_index={b: i for i, b in enumerate(bodies)},
    joint_of_body={s: i for i, s in enumerate(successor)},
  )


def _in_ref_order(name: str, joints: Iterable[str]) -> tuple[str, ...]:
  """``joints`` restricted to refJointOrder and returned in that order."""
  selected = set(joints)
  return tuple(j for j in get_ref_joint_order(name) if j in selected)


def _sensor_bodies(name: str, kind: str) -> dict[str, str]:
  module = get_robot_module(name)
  return {
    _decode_joint_key(s.name()): _decode_joint_key(s.parentBody())
    for s in (module.forceSensors() if kind == "force" else module.bodySensors())
  }


def get_root_body(name: str) -> str:
  """The floating-base body, from the module's own body sensor."""
  sensors = _sensor_bodies(name, "body")
  for sensor in FLOATING_BASE_SENSORS:
    if sensor in sensors:
      return sensors[sensor]
  raise ValueError(
    f"{name} has no {' / '.join(FLOATING_BASE_SENSORS)} body sensor "
    f"(has: {', '.join(sorted(sensors)) or 'none'}); the root body cannot be "
    "derived, pass it explicitly."
  )


def get_limb_body(name: str, limb: str, side: str) -> str:
  """The body carrying the ``limb`` ("foot"/"hand") force sensor on ``side``."""
  table = FOOT_FORCE_SENSOR if limb == "foot" else HAND_FORCE_SENSOR
  sensor = table[side]
  sensors = _sensor_bodies(name, "force")
  if sensor not in sensors:
    raise ValueError(
      f"{name} has no {sensor} (has: {', '.join(sorted(sensors)) or 'none'}); "
      f"the {side} {limb} cannot be located."
    )
  return sensors[sensor]


def get_chain_joints(name: str, to_body: str) -> tuple[str, ...]:
  """The joints on the path from the floating base down to ``to_body``."""
  tree = _get_tree(name)
  if to_body not in tree.body_index:
    raise ValueError(f"{name} has no body '{to_body}'")
  root = tree.body_index[get_root_body(name)]
  found: list[str] = []
  body = tree.body_index[to_body]
  while body != root and body in tree.joint_of_body:
    joint = tree.joint_of_body[body]
    found.append(tree.joints[joint])
    body = tree.predecessor[joint]
  return _in_ref_order(name, found)


def get_subtree_joints(name: str, from_body: str) -> tuple[str, ...]:
  """Every joint strictly distal to ``from_body`` -- its whole subtree."""
  tree = _get_tree(name)
  if from_body not in tree.body_index:
    raise ValueError(f"{name} has no body '{from_body}'")
  frontier = [tree.body_index[from_body]]
  reached = set(frontier)
  found: list[str] = []
  while frontier:
    body = frontier.pop()
    for joint, parent in enumerate(tree.predecessor):
      if parent == body and tree.successor[joint] not in reached:
        found.append(tree.joints[joint])
        reached.add(tree.successor[joint])
        frontier.append(tree.successor[joint])
  return _in_ref_order(name, found)


def get_leg_joints(name: str, side: str | None = None) -> tuple[str, ...]:
  """The leg chains, base to foot. Both legs unless ``side`` picks one."""
  sides = _SIDES if side is None else (side,)
  return _in_ref_order(
    name,
    [j for s in sides for j in get_chain_joints(name, get_limb_body(name, "foot", s))],
  )


def get_arm_joints(name: str, side: str | None = None) -> tuple[str, ...]:
  """The arm chains, base to wrist -- fingers excluded (see ``get_hand_joints``)."""
  sides = _SIDES if side is None else (side,)
  return _in_ref_order(
    name,
    [j for s in sides for j in get_chain_joints(name, get_limb_body(name, "hand", s))],
  )


def get_hand_joints(name: str, side: str | None = None) -> tuple[str, ...]:
  """Everything past the wrist force sensor: hand yaw, fingers, thumbs."""
  sides = _SIDES if side is None else (side,)
  return _in_ref_order(
    name,
    [
      j for s in sides for j in get_subtree_joints(name, get_limb_body(name, "hand", s))
    ],
  )


def get_lower_body_joints(name: str) -> tuple[str, ...]:
  """The legs -- the joints the stabilizer actually balances on."""
  return get_leg_joints(name)


def get_upper_body_joints(name: str) -> tuple[str, ...]:
  """refJointOrder minus the legs: waist, chest, head, arms and hands."""
  legs = set(get_leg_joints(name))
  return tuple(j for j in get_ref_joint_order(name) if j not in legs)


def get_fixed_joints(name: str) -> tuple[str, ...]:
  """refJointOrder entries the module models as fixed (0 DoF)."""
  tree = _get_tree(name)
  fixed = {j for j, dof in zip(tree.joints, tree.dofs, strict=True) if dof == 0}
  return _in_ref_order(name, fixed)


def get_mobile_joints(name: str) -> tuple[str, ...]:
  """refJointOrder entries the module actually drives (1 DoF)."""
  fixed = set(get_fixed_joints(name))
  return tuple(j for j in get_ref_joint_order(name) if j not in fixed)


def get_effort_limits(name: str) -> dict[str, float]:
  """Per-joint nominal torque limits (upper tau bound), 1-DoF joints only."""
  return {
    _decode_joint_key(key): float(value[0])
    for key, value in get_robot_module(name).bounds()[5].items()
    if len(value) == 1
  }
