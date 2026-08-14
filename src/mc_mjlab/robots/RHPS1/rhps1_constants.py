"""RHPS1 constants and helpers."""

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

# MJCF and assets.

RHPS1_MC_RTC_MODULE_NAME = "RHPS1_MuJoCo"
RHPS1_MC_RTC_ASSETS_DIR = MC_MUJOCO_SHARE_DIR / "RHPS1"
RHPS1_XML: Path = Path(__file__).parent / "xmls" / "RHPS1main.xml"
RHPS1_MESH_DIR: Path = Path(__file__).parent / "meshes"
RHPS1_PD_GAINS_DIR: Path = Path(__file__).parent / "pdgains"
RHPS1_PD_GAINS_PATH: Path = RHPS1_PD_GAINS_DIR / "RHPS1main" / "PDgains_sim.dat"

# Root body, for the subtree angular-momentum sensor.
ROOT_BODY = "BODY"


def ensure_assets() -> None:
  ensure_asset_symlink(RHPS1_MESH_DIR, RHPS1_MC_RTC_ASSETS_DIR / "meshes")
  ensure_asset_symlink(RHPS1_XML, RHPS1_MC_RTC_ASSETS_DIR / "xml" / "RHPS1main.xml")
  ensure_asset_symlink(RHPS1_PD_GAINS_DIR, RHPS1_MC_RTC_ASSETS_DIR / "pdgains")


# The sole is one box geom per foot (the ankle link's collision *mesh* is
# commented out in the XML), unnamed like every other collision geom. Name it
# semantically so the presets can address the feet apart from the body.
RHPS1_FOOT_BODIES: dict[str, str] = {
  "L_ANKLE_P_LINK": "left_foot_collision",
  "R_ANKLE_P_LINK": "right_foot_collision",
}


def get_spec() -> mujoco.MjSpec:
  """Load the RHPS1 MJCF."""
  ensure_assets()

  spec = mujoco.MjSpec.from_file(str(RHPS1_XML))

  name_foot_collision_geoms(spec, RHPS1_FOOT_BODIES)
  name_remaining_collision_geoms(spec, "rhps1")
  add_locomotion_sensors(spec, root_body=ROOT_BODY)
  group_and_disable_collision_geoms(spec)
  return spec


# Joint tables.


# All refJointOrder joints are actuated; `get_actuated_joints`'s
# `non_actuated` is how to make one passive. docs/robots.md#actuated-joint-sets


def get_residual_joints() -> tuple[str, ...]:
  """The joints a residual may act on: refJointOrder minus the carve-outs."""
  return mc_rtc.get_residual_joints(
    RHPS1_MC_RTC_MODULE_NAME,
    non_residual=mc_rtc.get_fixed_joints(RHPS1_MC_RTC_MODULE_NAME),
  )


# Collision presets. See collision.get_collision_presets for the contact model.

RHPS1_FOOT_COLLISION_EXPR = r"^(left|right)_foot_collision$"

(
  RHPS1_FEET_ONLY_COLLISION,
  RHPS1_FULL_COLLISION,
  RHPS1_FULL_COLLISION_WITHOUT_SELF,
) = get_collision_presets("rhps1", RHPS1_FOOT_COLLISION_EXPR)

RHPS1_COLLISION = RHPS1_FULL_COLLISION

# configuration build.


def get_robot_cfg() -> EntityCfg:
  """Return a fresh RHPS1 EntityCfg"""

  spec = get_spec()

  joints = mc_rtc.get_actuated_joints(RHPS1_MC_RTC_MODULE_NAME)

  simulated = {j.name for j in spec.joints}

  init_state = EntityCfg.InitialStateCfg(
    # Starting above the module's default attitude injects a drop transient.
    pos=mc_rtc.get_default_root_position(RHPS1_MC_RTC_MODULE_NAME),
    joint_pos=mc_rtc.get_default_joint_positions(RHPS1_MC_RTC_MODULE_NAME, simulated),
    joint_vel={".*": 0.0},
  )

  articulation = EntityArticulationInfoCfg(
    actuators=get_pd_actuator_cfgs(joints, get_armature_from_spec(spec, joints)),
    soft_joint_pos_limit_factor=0.99,
  )

  return EntityCfg(
    init_state=init_state,
    collisions=(RHPS1_COLLISION,),
    spec_fn=get_spec,
    articulation=articulation,
  )
