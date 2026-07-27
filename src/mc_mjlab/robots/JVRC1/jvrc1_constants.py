"""JVRC1 constants and helpers."""

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


def get_spec() -> mujoco.MjSpec:
  """Load the JVRC1 MJCF."""
  ensure_assets()

  spec = mujoco.MjSpec.from_file(str(JVRC1_XML))
  name_foot_collision_geoms(spec, JVRC1_FOOT_BODIES)
  name_remaining_collision_geoms(spec, "jvrc1")
  add_locomotion_sensors(spec, root_body=JVRC1_ROOT_BODY)
  group_and_disable_collision_geoms(spec)
  return spec


##
# Joint tables.
##

# Default: every joint in the mc_rtc refJointOrder (all 44, fingers included) is
# actuated and receives the RL residual. Carve joints out with:
#   - JVRC1_NON_ACTUATED_JOINTS: left fully passive (no actuator, no residual).
#   - JVRC1_NON_RESIDUAL_JOINTS: actuated (tracks the controller) but no residual
#     -- e.g. add the finger joints here to keep them out of the policy's action.
JVRC1_NON_ACTUATED_JOINTS: frozenset[str] = frozenset()
JVRC1_NON_RESIDUAL_JOINTS: frozenset[str] = frozenset()


def get_residual_joints() -> tuple[str, ...]:
  return mc_rtc.get_residual_joints(
    JVRC1_MC_RTC_MODULE_NAME,
    non_actuated=JVRC1_NON_ACTUATED_JOINTS,
    non_residual=JVRC1_NON_RESIDUAL_JOINTS,
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


def get_robot_cfg() -> EntityCfg:
  """Return a fresh JVRC1 EntityCfg."""

  spec = get_spec()

  joints = mc_rtc.get_actuated_joints(
    JVRC1_MC_RTC_MODULE_NAME, non_actuated=JVRC1_NON_ACTUATED_JOINTS
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
    spec_fn=get_spec,
    articulation=articulation,
  )
