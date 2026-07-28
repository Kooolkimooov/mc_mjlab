"""JVRC1 constants and helpers."""

import functools
from pathlib import Path

import mujoco
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mc_mjlab.robots import mc_rtc_robot_configuration as mc_rtc
from mc_mjlab.robots.additional_sensors_configuration import add_locomotion_sensors
from mc_mjlab.robots.collision_configuration import (
  get_collision_presets,
  group_and_disable_collision_geoms,
  name_foot_collision_geoms,
  name_remaining_collision_geoms,
)
from mc_mjlab.robots.mc_mujoco_assets import (
  MC_MUJOCO_SHARE_DIR,
  ensure_asset_symlink,
)
from mc_mjlab.robots.pd_actuator_configuration import (
  get_armature_from_spec,
  get_pd_actuator_cfgs,
)

##
# MJCF and assets.
##

JVRC1_MC_RTC_MODULE_NAME = "JVRC1"
JVRC1_MC_RTC_ASSETS_DIR = MC_MUJOCO_SHARE_DIR / "JVRC1"
JVRC1_XML: Path = Path(__file__).parent / "xmls" / "JVRC1.xml"
JVRC1_MESH_DIR: Path = Path(__file__).parent / "meshes"
JVRC1_PD_GAINS_DIR: Path = Path(__file__).parent / "pdgains"
JVRC1_PD_GAINS_PATH: Path = JVRC1_PD_GAINS_DIR / "PDgains_sim.dat"


def ensure_assets() -> None:
  ensure_asset_symlink(JVRC1_MESH_DIR, JVRC1_MC_RTC_ASSETS_DIR / "meshes")
  ensure_asset_symlink(JVRC1_XML, JVRC1_MC_RTC_ASSETS_DIR / "xml" / "jvrc1.xml")
  ensure_asset_symlink(JVRC1_PD_GAINS_DIR, JVRC1_MC_RTC_ASSETS_DIR / "pdgains")


# Root body, for the subtree angular-momentum sensor.
JVRC1_ROOT_BODY = "PELVIS_S"

# Each foot is one collision mesh on the ankle-pitch link, unnamed like every
# other collision geom. Name it semantically so the presets can address the
# feet apart from the body.
JVRC1_FOOT_BODIES: dict[str, str] = {
  "L_ANKLE_P_S": "left_foot_collision",
  "R_ANKLE_P_S": "right_foot_collision",
}


# JVRC1's hands are an underactuated grasp: five finger joints per side follow
# ``<side>_UTHUMB`` through active mjEQ_JOINT equalities with fixed gear ratios
# (+/-1, +/-3). That leaves two coherent configurations, and the spec and the
# actuator set have to pick the same one:
#
#   True  (default) -- keep the couplings and leave the five slaves per hand
#     unactuated. One commanded grasp DoF per hand, the fingers following it;
#     this is the joint set mc_mujoco motorizes.
#   False -- delete the couplings and actuate all 44, giving mc_rtc the
#     per-finger control its refJointOrder and its 44 gain rows imply. Pick
#     this for a task that actually manipulates something.
#
# Actuating a slave while its equality is live puts the PD and the constraint
# solver on one DoF, pulling against each other. The whole finger stance is
# zero and zero satisfies every ratio, so that conflict stays invisible until
# something commands a finger away from zero -- which is why it can hide.
JVRC1_COUPLED_FINGERS = True


def _coupled_joints(spec: mujoco.MjSpec) -> tuple[str, ...]:
  """The slaved side of every active ``mjEQ_JOINT`` equality in ``spec``"""
  return tuple(
    eq.name1
    for eq in spec.equalities
    if eq.type == mujoco.mjtEq.mjEQ_JOINT and eq.active and eq.name1
  )


def _delete_couplings(spec: mujoco.MjSpec) -> None:
  """Drop the joint couplings, making the slaved fingers independent DoFs."""
  for eq in list(spec.equalities):
    if eq.type == mujoco.mjtEq.mjEQ_JOINT:
      spec.delete(eq)


def get_spec(coupled_fingers: bool = JVRC1_COUPLED_FINGERS) -> mujoco.MjSpec:
  """Load the JVRC1 MJCF."""
  ensure_assets()

  spec = mujoco.MjSpec.from_file(str(JVRC1_XML))
  name_foot_collision_geoms(spec, JVRC1_FOOT_BODIES)
  name_remaining_collision_geoms(spec, "jvrc1")
  add_locomotion_sensors(spec, root_body=JVRC1_ROOT_BODY)
  group_and_disable_collision_geoms(spec)
  if not coupled_fingers:
    _delete_couplings(spec)
  return spec


@functools.lru_cache(maxsize=None)
def _coupled_finger_names() -> frozenset[str]:
  """The slaved finger joints, resolved once"""
  return frozenset(_coupled_joints(get_spec(coupled_fingers=True)))


def get_non_actuated_joints(
  coupled_fingers: bool = JVRC1_COUPLED_FINGERS,
) -> frozenset[str]:
  """The joints to leave without an actuator, for the chosen hand mode"""
  if not coupled_fingers:
    return frozenset()
  return _coupled_finger_names()


##
# Joint tables.
##


def get_residual_joints(
  coupled_fingers: bool = JVRC1_COUPLED_FINGERS,
) -> tuple[str, ...]:
  """The joints a residual may act on: refJointOrder minus the carve-outs"""
  return mc_rtc.get_residual_joints(
    JVRC1_MC_RTC_MODULE_NAME,
    non_actuated=get_non_actuated_joints(coupled_fingers),
    non_residual=mc_rtc.get_fixed_joints(JVRC1_MC_RTC_MODULE_NAME),
  )


##
# Collision presets. See collision.get_collision_presets for the contact model.
##

JVRC1_FOOT_COLLISION_EXPR = r"^(left|right)_foot_collision$"

(
  JVRC1_FEET_ONLY_COLLISION,
  JVRC1_FULL_COLLISION,
  JVRC1_FULL_COLLISION_WITHOUT_SELF,
) = get_collision_presets("jvrc1", JVRC1_FOOT_COLLISION_EXPR)

JVRC1_COLLISION = JVRC1_FULL_COLLISION


def get_robot_cfg(coupled_fingers: bool = JVRC1_COUPLED_FINGERS) -> EntityCfg:
  """Return a fresh JVRC1 EntityCfg."""

  spec = get_spec(coupled_fingers)

  joints = mc_rtc.get_actuated_joints(
    JVRC1_MC_RTC_MODULE_NAME, non_actuated=get_non_actuated_joints(coupled_fingers)
  )

  simulated = {j.name for j in spec.joints}

  init_state = EntityCfg.InitialStateCfg(
    pos=mc_rtc.get_default_root_position(JVRC1_MC_RTC_MODULE_NAME),
    joint_pos=mc_rtc.get_default_joint_positions(JVRC1_MC_RTC_MODULE_NAME, simulated),
    joint_vel={".*": 0.0},
  )

  articulation = EntityArticulationInfoCfg(
    actuators=get_pd_actuator_cfgs(joints, get_armature_from_spec(spec, joints)),
    soft_joint_pos_limit_factor=0.99,
  )

  return EntityCfg(
    init_state=init_state,
    collisions=(JVRC1_COLLISION,),
    spec_fn=functools.partial(get_spec, coupled_fingers),
    articulation=articulation,
  )
