"""JVRC1 constants and helpers."""

import math
from pathlib import Path

import mujoco
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from mc_mjlab.robots.workspace_assets import MC_MUJOCO_SHARE, ensure_workspace_link

##
# MJCF and assets.
##

_WORKSPACE_JVRC1 = MC_MUJOCO_SHARE / "JVRC1"
JVRC1_XML: Path = Path(__file__).parent / "xmls" / "JVRC1.xml"
MESH_DIR: Path = Path(__file__).parent / "meshes"
PD_GAINS_DIR: Path = Path(__file__).parent / "pdgains"
# The real robot PD gains mc_mujoco applies (one `kp kd` row per
# refJointOrder joint); consumed by McRtcResidualJointPositionActionCfg.
PD_GAINS_PATH: Path = PD_GAINS_DIR / "PDgains_sim.dat"


def _ensure_assets() -> None:
  ensure_workspace_link(MESH_DIR, _WORKSPACE_JVRC1 / "meshes")
  ensure_workspace_link(JVRC1_XML, _WORKSPACE_JVRC1 / "xml" / "jvrc1.xml")
  ensure_workspace_link(PD_GAINS_DIR, _WORKSPACE_JVRC1 / "pdgains")


def get_spec() -> mujoco.MjSpec:
  """Load the JVRC1 MJCF."""
  _ensure_assets()

  spec = mujoco.MjSpec.from_file(str(JVRC1_XML))

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
# Joint tables (from jvrc1.xml / the mc_rtc robot module).
##

# Rotor armature per joint, as in the MJCF.
JVRC1_ARMATURE: dict[str, float] = {
  "R_HIP_P": 0.1925,
  "R_HIP_R": 0.1813,
  "R_HIP_Y": 0.1237,
  "R_KNEE": 0.1305,
  "R_ANKLE_R": 0.0653,
  "R_ANKLE_P": 0.1337,
  "L_HIP_P": 0.1925,
  "L_HIP_R": 0.1813,
  "L_HIP_Y": 0.1237,
  "L_KNEE": 0.1305,
  "L_ANKLE_R": 0.0653,
  "L_ANKLE_P": 0.1337,
  "WAIST_Y": 0.1221,
  "WAIST_P": 0.1813,
  "WAIST_R": 0.1054,
  "NECK_Y": 0.0567,
  "NECK_R": 0.0596,
  "NECK_P": 0.0596,
  "R_SHOULDER_P": 0.1210,
  "R_SHOULDER_R": 0.1210,
  "R_SHOULDER_Y": 0.1231,
  "R_ELBOW_P": 0.1054,
  "R_ELBOW_Y": 0.1240,
  "R_WRIST_R": 0.0876,
  "R_WRIST_Y": 0.1240,
  "R_UTHUMB": 0.0130,
  "R_LTHUMB": 0.0320,
  "R_UINDEX": 0.0073,
  "R_LINDEX": 0.0039,
  "R_ULITTLE": 0.0073,
  "R_LLITTLE": 0.0039,
  "L_SHOULDER_P": 0.1210,
  "L_SHOULDER_R": 0.1210,
  "L_SHOULDER_Y": 0.1231,
  "L_ELBOW_P": 0.1054,
  "L_ELBOW_Y": 0.1240,
  "L_WRIST_R": 0.0876,
  "L_WRIST_Y": 0.1240,
  "L_UTHUMB": 0.0130,
  "L_LTHUMB": 0.0320,
  "L_UINDEX": 0.0073,
  "L_LINDEX": 0.0039,
  "L_ULITTLE": 0.0073,
  "L_LLITTLE": 0.0039,
}

# The joints mc_mujoco motorizes (all but the lower-finger joints); the rest
# stay passive, matching its dynamics.
JVRC1_MOTORIZED_JOINTS: tuple[str, ...] = (
  "R_HIP_P",
  "R_HIP_R",
  "R_HIP_Y",
  "R_KNEE",
  "R_ANKLE_R",
  "R_ANKLE_P",
  "L_HIP_P",
  "L_HIP_R",
  "L_HIP_Y",
  "L_KNEE",
  "L_ANKLE_R",
  "L_ANKLE_P",
  "WAIST_Y",
  "WAIST_P",
  "WAIST_R",
  "NECK_Y",
  "NECK_R",
  "NECK_P",
  "R_SHOULDER_P",
  "R_SHOULDER_R",
  "R_SHOULDER_Y",
  "R_ELBOW_P",
  "R_ELBOW_Y",
  "R_WRIST_R",
  "R_WRIST_Y",
  "R_UTHUMB",
  "L_SHOULDER_P",
  "L_SHOULDER_R",
  "L_SHOULDER_Y",
  "L_ELBOW_P",
  "L_ELBOW_Y",
  "L_WRIST_R",
  "L_WRIST_Y",
  "L_UTHUMB",
)

# Motorized joints that receive the RL residual (no fingers).
JVRC1_RESIDUAL_JOINTS: tuple[str, ...] = tuple(
  n for n in JVRC1_MOTORIZED_JOINTS if not n.endswith("THUMB")
)

##
# Actuator config.
##

NATURAL_FREQ = 3.0 * 2.0 * math.pi  # rad/s
DAMPING_RATIO = 1.5

# Unclamped, like mc_mujoco's PD torque (its motors set forcelimited=false);
# with the real gains nominal limits would saturate constantly and change the
# stabilizer's behavior. The armature-derived gains are defaults; the demo
# overrides them with PDgains_sim.dat via pd_gains_path.
EFFORT_LIMIT = float("inf")

JVRC1_ACTUATORS: tuple[IdealPdActuatorCfg, ...] = tuple(
  IdealPdActuatorCfg(
    target_names_expr=(name,),
    stiffness=JVRC1_ARMATURE[name] * NATURAL_FREQ**2,
    damping=2 * DAMPING_RATIO * JVRC1_ARMATURE[name] * NATURAL_FREQ,
    effort_limit=EFFORT_LIMIT,
    armature=JVRC1_ARMATURE[name],
  )
  for name in JVRC1_MOTORIZED_JOINTS
)

##
# Initial state: half-sitting stance from the mc_rtc robot module.
##

JVRC1_INIT_STATE = EntityCfg.InitialStateCfg(
  # z = the module's default attitude; starting higher injects a drop
  # transient at reset.
  pos=(0.0, 0.0, 0.8275),
  joint_pos={
    "R_HIP_P": -0.38,
    "R_HIP_R": -0.01,
    "R_KNEE": 0.72,
    "R_ANKLE_R": -0.01,
    "R_ANKLE_P": -0.33,
    "L_HIP_P": -0.38,
    "L_HIP_R": 0.02,
    "L_KNEE": 0.72,
    "L_ANKLE_R": -0.02,
    "L_ANKLE_P": -0.33,
    "WAIST_P": 0.13,
    "R_SHOULDER_P": -0.052,
    "R_SHOULDER_R": -0.17,
    "R_ELBOW_P": -0.52,
    "L_SHOULDER_P": -0.052,
    "L_SHOULDER_R": 0.17,
    "L_ELBOW_P": -0.52,
  },
  joint_vel={".*": 0.0},
)

##
# Final config.
##

JVRC1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=JVRC1_ACTUATORS,
  soft_joint_pos_limit_factor=0.99,
)


def get_jvrc1_robot_cfg() -> EntityCfg:
  """Return a fresh JVRC1 EntityCfg."""
  return EntityCfg(
    init_state=JVRC1_INIT_STATE,
    collisions=(),  # geoms are unnamed; consumers enable them by group
    spec_fn=get_spec,
    articulation=JVRC1_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_jvrc1_robot_cfg())

  viewer.launch(robot.spec.compile())
