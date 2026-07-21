"""RHPS1 constants and helpers."""

import math
from pathlib import Path

import mujoco
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mc_mjlab.robots.workspace_assets import MC_MUJOCO_SHARE, ensure_workspace_link

##
# MJCF and assets.
##

_WORKSPACE_RHPS1 = MC_MUJOCO_SHARE / "RHPS1"
RHPS1_XML: Path = Path(__file__).parent / "xmls" / "RHPS1main.xml"
MESH_DIR: Path = Path(__file__).parent / "meshes"
PD_GAINS_DIR: Path = Path(__file__).parent / "pdgains"
# The real robot PD gains mc_mujoco applies (one `kp kd` row per
# refJointOrder joint); consumed by McRtcResidualJointPositionActionCfg.
# The share keeps one subdirectory per hand-attachment variant; RHPS1main is
# the bare robot this package models.
PD_GAINS_PATH: Path = PD_GAINS_DIR / "RHPS1main" / "PDgains_sim.dat"


def _ensure_assets() -> None:
  ensure_workspace_link(MESH_DIR, _WORKSPACE_RHPS1 / "meshes")
  ensure_workspace_link(RHPS1_XML, _WORKSPACE_RHPS1 / "xml" / "RHPS1main.xml")
  ensure_workspace_link(PD_GAINS_DIR, _WORKSPACE_RHPS1 / "pdgains")


def get_spec() -> mujoco.MjSpec:
  """Load the RHPS1 MJCF."""
  _ensure_assets()

  spec = mujoco.MjSpec.from_file(str(RHPS1_XML))

  # Grouping convention: visual geoms 2, collision geoms 3, sites 4. The MJCF
  # defaults (`class="visual"/"collision"`) carry the semantic split.
  for geom in spec.geoms:
    if geom.conaffinity == 0 and geom.contype == 0:
      geom.group = 2
    else:
      geom.group = 3
  for site in spec.sites:
    site.group = 4

  # Collisions off by default; consumers re-enable selected sets (the geoms
  # are unnamed, so the name-based presets cannot).
  for geom in spec.geoms:
    geom.contype = 0
    geom.conaffinity = 0
  return spec


##
# Joint tables (from RHPS1main.xml / the RHPS1_MuJoCo robot module).
##

# The joints mc_mujoco motorizes, i.e. every rotary joint. The 8 slide joints
# (`[RL]C-[CA][IO]-linear-joint`) are the knee/ankle drive linkages: they stay
# passive and follow through the MJCF's equality constraints.
RHPS1_MOTORIZED_JOINTS: tuple[str, ...] = (
  "R_SHOULDER_P",
  "R_SHOULDER_R",
  "R_SHOULDER_Y",
  "R_ELBOW_P",
  "R_ELBOW_Y",
  "R_WRIST_R",
  "R_WRIST_Y",
  "L_SHOULDER_P",
  "L_SHOULDER_R",
  "L_SHOULDER_Y",
  "L_ELBOW_P",
  "L_ELBOW_Y",
  "L_WRIST_R",
  "L_WRIST_Y",
  "CHEST_Y",
  "CHEST_P",
  "HEAD_Y",
  "HEAD_P",
  "L_CROTCH_Y",
  "L_CROTCH_R",
  "L_CROTCH_P",
  "L_KNEE_P",
  "L_ANKLE_R",
  "L_ANKLE_P",
  "R_CROTCH_Y",
  "R_CROTCH_R",
  "R_CROTCH_P",
  "R_KNEE_P",
  "R_ANKLE_R",
  "R_ANKLE_P",
)

# All motorized joints receive the RL residual (RHPS1main has no fingers).
RHPS1_RESIDUAL_JOINTS: tuple[str, ...] = RHPS1_MOTORIZED_JOINTS

##
# Actuator config.
##

NATURAL_FREQ = 3.0 * 2.0 * math.pi  # rad/s
DAMPING_RATIO = 1.5

# The MJCF sets a uniform armature of 1 through its joint default class.
ARMATURE = 1.0

# Unclamped, like mc_mujoco's PD torque (its motors set forcelimited=false);
# with the real gains nominal limits would saturate constantly and change the
# stabilizer's behavior. The armature-derived gains are defaults; the demo
# overrides them with PDgains_sim.dat via pd_gains_path.
EFFORT_LIMIT = float("inf")

RHPS1_ACTUATORS: tuple[IdealPdActuatorCfg, ...] = tuple(
  IdealPdActuatorCfg(
    target_names_expr=(name,),
    stiffness=ARMATURE * NATURAL_FREQ**2,
    damping=2 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ,
    effort_limit=EFFORT_LIMIT,
    armature=ARMATURE,
  )
  for name in RHPS1_MOTORIZED_JOINTS
)

##
# Initial state: half-sitting stance from the RHPS1_MuJoCo robot module.
##

RHPS1_INIT_STATE = EntityCfg.InitialStateCfg(
  # z = the module's default attitude; starting higher injects a drop
  # transient at reset.
  pos=(0.0, 0.0, 0.8377),
  joint_pos={
    "L_CROTCH_Y": -0.01053,
    "L_CROTCH_R": 0.02879,
    "L_CROTCH_P": -0.27119,
    "L_KNEE_P": 0.62202,
    "L_ANKLE_R": -0.03065,
    "L_ANKLE_P": -0.35068,
    "R_CROTCH_Y": 0.01053,
    "R_CROTCH_R": -0.02879,
    "R_CROTCH_P": -0.27119,
    "R_KNEE_P": 0.62202,
    "R_ANKLE_R": 0.03065,
    "R_ANKLE_P": -0.35068,
    "L_SHOULDER_P": 0.2618,
    "L_SHOULDER_R": 0.17453,
    "L_SHOULDER_Y": -0.08727,
    "L_ELBOW_P": -0.5236,
    "R_SHOULDER_P": 0.2618,
    "R_SHOULDER_R": -0.17453,
    "R_SHOULDER_Y": 0.08727,
    "R_ELBOW_P": -0.5236,
  },
  joint_vel={".*": 0.0},
)

##
# Final config.
##

RHPS1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=RHPS1_ACTUATORS,
  soft_joint_pos_limit_factor=0.99,
)


def get_rhps1_robot_cfg() -> EntityCfg:
  """Return a fresh RHPS1 EntityCfg."""
  return EntityCfg(
    init_state=RHPS1_INIT_STATE,
    collisions=(),  # geoms are unnamed; consumers enable them by group
    spec_fn=get_spec,
    articulation=RHPS1_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_rhps1_robot_cfg())

  viewer.launch(robot.spec.compile())
