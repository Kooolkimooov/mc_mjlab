"""HRP5P constants and helpers."""

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

HRP5P_MC_RTC_MODULE_NAME = "HRP5P"
HRP5P_MC_RTC_ASSETS_DIR = MC_MUJOCO_SHARE_DIR / "HRP5P"
HRP5P_XML: Path = Path(__file__).parent / "xmls" / "HRP5P.xml"
HRP5P_MESH_DIR: Path = Path(__file__).parent / "meshes"
HRP5P_PD_GAINS_DIR: Path = Path(__file__).parent / "pdgains"
HRP5P_PD_GAINS_PATH: Path = HRP5P_PD_GAINS_DIR / "PDgains_sim.dat"


def ensure_assets() -> None:
  ensure_asset_symlink(HRP5P_MESH_DIR, HRP5P_MC_RTC_ASSETS_DIR / "meshes")
  ensure_asset_symlink(HRP5P_XML, HRP5P_MC_RTC_ASSETS_DIR / "xml" / "HRP5Pmain.xml")
  ensure_asset_symlink(HRP5P_PD_GAINS_DIR, HRP5P_MC_RTC_ASSETS_DIR / "pdgains")


# Root body, for the subtree angular-momentum sensor.
HRP5P_ROOT_BODY = "Body"

# Each foot is one primitive sole box on the last leg link, unnamed like every
# other collision geom. Name it semantically so the presets can address the
# feet apart from the body.
HRP5P_FOOT_BODIES: dict[str, str] = {
  "Lleg_Link5": "left_foot_collision",
  "Rleg_Link5": "right_foot_collision",
}


def get_spec() -> mujoco.MjSpec:
  """Load the HRP5P MJCF."""
  ensure_assets()

  spec = mujoco.MjSpec.from_file(str(HRP5P_XML))
  spec.compiler.balanceinertia = True  # URDF-derived inertias can be invalid
  name_foot_collision_geoms(spec, HRP5P_FOOT_BODIES)
  name_remaining_collision_geoms(spec, "hrp5p")
  add_locomotion_sensors(spec, root_body=HRP5P_ROOT_BODY)
  group_and_disable_collision_geoms(spec)
  return spec


##
# Joint tables.
##

# Default: every joint in the mc_rtc refJointOrder (all 53, fingers included) is
# actuated and receives the RL residual. Gains come from PDgains_sim.dat at
# action-term init, so the fingers get their real gains without a special case.
# Carve joints out with:
#   - HRP5P_NON_ACTUATED_JOINTS: left fully passive (no actuator, no residual).
#   - HRP5P_NON_RESIDUAL_JOINTS: actuated (tracks the controller) but no residual
#     -- e.g. add the finger joints here to keep them out of the policy's action.
HRP5P_NON_ACTUATED_JOINTS: frozenset[str] = frozenset()
HRP5P_NON_RESIDUAL_JOINTS: frozenset[str] = frozenset()


def get_residual_joints() -> tuple[str, ...]:
  return mc_rtc.get_residual_joints(
    HRP5P_MC_RTC_MODULE_NAME,
    non_actuated=HRP5P_NON_ACTUATED_JOINTS,
    non_residual=HRP5P_NON_RESIDUAL_JOINTS,
  )


##
# Collision presets. See collision.get_collision_presets for the contact model.
##

HRP5P_FOOT_COLLISION_EXPR = r"^(left|right)_foot_collision$"

(
  HRP5P_FEET_ONLY_COLLISION,
  HRP5P_FULL_COLLISION,
  HRP5P_FULL_COLLISION_WITHOUT_SELF,
) = get_collision_presets("hrp5p", HRP5P_FOOT_COLLISION_EXPR)

HRP5P_COLLISION = HRP5P_FULL_COLLISION


def get_robot_cfg() -> EntityCfg:
  """Return a fresh HRP5P EntityCfg."""
  spec = get_spec()

  joints = mc_rtc.get_actuated_joints(
    HRP5P_MC_RTC_MODULE_NAME, non_actuated=HRP5P_NON_ACTUATED_JOINTS
  )

  simulated = {j.name for j in spec.joints}

  init_state = EntityCfg.InitialStateCfg(
    pos=mc_rtc.get_default_root_position(HRP5P_MC_RTC_MODULE_NAME),
    joint_pos=mc_rtc.get_default_joint_positions(HRP5P_MC_RTC_MODULE_NAME, simulated),
    joint_vel={".*": 0.0},
  )

  articulation = EntityArticulationInfoCfg(
    actuators=get_pd_actuator_cfgs(joints, get_armature_from_spec(spec, joints)),
    soft_joint_pos_limit_factor=0.99,
  )

  return EntityCfg(
    init_state=init_state,
    collisions=(HRP5P_COLLISION,),
    spec_fn=get_spec,
    articulation=articulation,
  )
