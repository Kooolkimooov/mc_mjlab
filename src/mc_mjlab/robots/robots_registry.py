"""Which robot the coupling runs, and how its cfg must be prepared.

``etc/mc_rtc.yaml``'s ``MainRobot`` is the single source of truth: it picks the
mc_rtc robot module and, through ``ROBOTS``, the matching mjlab entity. Both
the demo and the RL task go through here so the two sides cannot drift.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from mjlab.entity import EntityCfg

from mc_mjlab.robots.collision_configuration import enable_all_collision_geoms
from mc_mjlab.robots.HRP5P import hrp5p_constants
from mc_mjlab.robots.JVRC1 import jvrc1_constants
from mc_mjlab.robots.RHPS1 import rhps1_constants
from mc_mjlab.utils.mc_rtc_config import get_main_robot_name


@dataclass(frozen=True)
class RobotSpec:
  """Per-robot wiring for the mc_rtc coupling."""

  cfg_fn: Callable[[], EntityCfg]
  get_residual_joints: Callable[[], tuple[str, ...]]
  """The joints that receive the RL residual, resolved on call.

  A callable, not a tuple, so nothing that only imports the registry forces a
  robot module to load: RHPS1 derives its set from the mc_rtc module, and that
  must not happen until a robot is actually used.
  """
  pd_gains_path: Path
  names_collision_geoms: bool = False
  """Whether ``cfg_fn`` names its collision geoms, so its ``CollisionCfg``
  presets actually match.

  Cannot be inferred from a non-empty ``EntityCfg.collisions``: a robot can
  ship a preset whose regexes match none of its unnamed geoms, and enabling
  that preset instead of the blanket fallback drops the robot on the floor.
  """


ROBOTS: dict[str, RobotSpec] = {
  "HRP5P": RobotSpec(
    cfg_fn=hrp5p_constants.get_robot_cfg,
    get_residual_joints=hrp5p_constants.get_residual_joints,
    pd_gains_path=hrp5p_constants.HRP5P_PD_GAINS_PATH,
    names_collision_geoms=True,
  ),
  "JVRC1": RobotSpec(
    cfg_fn=jvrc1_constants.get_robot_cfg,
    get_residual_joints=jvrc1_constants.get_residual_joints,
    pd_gains_path=jvrc1_constants.JVRC1_PD_GAINS_PATH,
    names_collision_geoms=True,
  ),
  "RHPS1_MuJoCo": RobotSpec(
    cfg_fn=rhps1_constants.get_robot_cfg,
    get_residual_joints=rhps1_constants.get_residual_joints,
    pd_gains_path=rhps1_constants.RHPS1_PD_GAINS_PATH,
    names_collision_geoms=True,
  ),
}


def get_main_robot_spec(mc_rtc_yaml: Path) -> tuple[str, RobotSpec]:
  """``(name, spec)`` for the config's MainRobot, or a clear error."""
  name = get_main_robot_name(mc_rtc_yaml)
  if name not in ROBOTS:
    raise ValueError(
      f"MainRobot '{name}' in {mc_rtc_yaml} has no RobotSpec "
      f"(known: {', '.join(sorted(ROBOTS))})."
    )
  return name, ROBOTS[name]


def prepare_cfg_for_mc_rtc(
  robot_cfg: EntityCfg, *, names_collision_geoms: bool = False
) -> EntityCfg:
  """Prepare a robot cfg for the mc_rtc coupling.

  Always deletes the XML's own motors (mjlab adds its own; keeping both
  doubles ``nu`` with dead actuators).

  Collisions take one of two paths, per ``RobotSpec.names_collision_geoms``.
  A robot that names its collision geoms -- all three do today -- keeps its
  ``CollisionCfg`` presets and mjlab applies them normally. A robot whose geoms
  are still unnamed has no preset that can match, so its presets are dropped
  and the collision geoms re-enabled wholesale by group instead -- without that
  fallback it stands on nothing and sinks through the floor.

  The default is the fallback, so a robot that has not declared itself keeps
  the behaviour it had before presets existed.
  """
  base_spec_fn = robot_cfg.spec_fn

  def spec_fn():
    spec = base_spec_fn()
    if not names_collision_geoms:
      enable_all_collision_geoms(spec)  # unnamed geoms: presets can't match
    for act in list(spec.actuators):
      spec.delete(act)
    return spec

  robot_cfg.spec_fn = spec_fn
  if not names_collision_geoms:
    robot_cfg.collisions = ()  # the preset's regexes match nothing here
  return robot_cfg
