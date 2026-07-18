"""HRP5P constants and helpers."""

import math
from pathlib import Path

import mujoco
import torch
from mjlab.actuator import IdealPdActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from mc_mjlab.robots.workspace_assets import MC_MUJOCO_SHARE, ensure_workspace_link

##
# MJCF and assets.
##


HRP5_XML: Path = Path(__file__).parent / "xmls" / "HRP5P.xml"

# The XML symlink keeps the local name `HRP5P.xml` even though the workspace
# file is `HRP5Pmain.xml`: get_spec()'s HRP5Pmain-specific geom-naming branch
# keys on the file name, and must stay off (the demo's collision handling in
# test_mc_rtc.py is built around it not running).
_WORKSPACE_HRP5P = MC_MUJOCO_SHARE / "HRP5P"
MESH_DIR: Path = Path(__file__).parent / "meshes"
PD_GAINS_DIR: Path = Path(__file__).parent / "pdgains"
# The real robot PD gains mc_mujoco applies (one `kp kd` row per
# refJointOrder joint); consumed by McRtcResidualJointPositionActionCfg.
PD_GAINS_PATH: Path = PD_GAINS_DIR / "PDgains_sim.dat"


def _ensure_assets() -> None:
  ensure_workspace_link(MESH_DIR, _WORKSPACE_HRP5P / "meshes")
  ensure_workspace_link(HRP5_XML, _WORKSPACE_HRP5P / "xml" / "HRP5Pmain.xml")
  ensure_workspace_link(PD_GAINS_DIR, _WORKSPACE_HRP5P / "pdgains")


def get_assets(meshdir: str) -> dict[str, bytes]:
  """No inlined assets: the MJCF resolves meshes on disk via its meshdir."""
  del meshdir
  return {}


def _name_hrp5_main_collision_geoms(spec: mujoco.MjSpec) -> None:
  """Name unnamed collision geoms after their meshes, so the regex-based
  collision presets can select them."""
  existing_names = {geom.name for geom in spec.geoms if geom.name}
  for geom in spec.geoms:
    if geom.name:
      continue
    if geom.contype == 0 and geom.conaffinity == 0:
      continue
    meshname = getattr(geom, "meshname", "")
    if not meshname:
      continue
    base_name = meshname[:-5] if meshname.endswith("_mesh") else meshname
    candidate = f"hrp5p_collision_{base_name}"
    if candidate in existing_names:
      continue
    geom.name = candidate
    existing_names.add(candidate)


def get_spec() -> mujoco.MjSpec:
  """Load the HRP5 MJCF/URDF and inline assets."""
  _ensure_assets()

  spec = mujoco.MjSpec.from_file(str(HRP5_XML))
  if HRP5_XML.name == "HRP5Pmain.xml":
    _name_hrp5_main_collision_geoms(spec)
  spec.compiler.balanceinertia = True  # URDF-derived inertias can be invalid
  spec.assets = get_assets(spec.meshdir)

  # Grouping convention: visual geoms 2, collision geoms 3, sites 4. The MJCF
  # defaults (`class="visual"/"collision"`) carry the semantic split.
  for geom in spec.geoms:
    if geom.conaffinity == 0 and geom.contype == 0:
      geom.group = 2
    else:
      geom.group = 3
  for site in spec.sites:
    site.group = 4

  # Collisions off by default; the collision presets re-enable selected sets.
  for geom in spec.geoms:
    geom.contype = 0
    geom.conaffinity = 0
  return spec


##
# Actuator config.
##
NATURAL_FREQ = 3.0 * 2.0 * math.pi  # rad/s
DAMPING_RATIO = 1.5

# Unclamped, like mc_mujoco's PD torque: with the real gains (kp up to 36000)
# nominal limits saturate constantly and change the stabilizer's behavior.
# HRP5P_NOMINAL_EFFORT_LIMITS keeps the per-joint scale for consumers.
EFFORT_LIMIT = float("inf")

HRP5P_NOMINAL_EFFORT_LIMITS: dict[str, float] = {
  "RCY": 200.0,
  "RCR": 500.0,
  "RCP": 500.0,
  "RKP": 1200.0,
  "RAP": 250.0,
  "RAR": 200.0,
  "LCY": 200.0,
  "LCR": 500.0,
  "LCP": 500.0,
  "LKP": 1200.0,
  "LAP": 250.0,
  "LAR": 200.0,
  "WP": 900.0,
  "WR": 900.0,
  "WY": 250.0,
  "HY": 45.0,
  "HP": 45.0,
  "RSC": 200.0,
  "RSP": 400.0,
  "RSR": 350.0,
  "RSY": 200.0,
  "REP": 400.0,
  "RWRY": 300.0,
  "RWRR": 300.0,
  "RWRP": 120.0,
  "LSC": 200.0,
  "LSP": 400.0,
  "LSR": 350.0,
  "LSY": 200.0,
  "LEP": 400.0,
  "LWRY": 300.0,
  "LWRR": 300.0,
  "LWRP": 120.0,
}
ARMATURE = torch.tensor(
  [
    0.110084846291,
    0.317032732584,
    0.317032732584,
    1.174334694856,
    0.161699903568,
    0.112299753167,
    0.110084846291,
    0.317032732584,
    0.317032732584,
    1.174334694856,
    0.161699903568,
    0.112299753167,
    0.782889796570,
    0.782889796570,
    0.158516366292,
    0.023698573895,
    0.024173043016,
    0.164700000000,
    0.391444898285,
    0.277300653884,
    0.110084846291,
    0.325195312499,
    0.164700000000,
    0.164700000000,
    0.164700000000,
    0.164700000000,
    0.391444898285,
    0.277300653884,
    0.110084846291,
    0.325195312499,
    0.164700000000,
    0.164700000000,
    0.164700000000,
  ],
  dtype=torch.float32,
)
STIFFNESS = [a * NATURAL_FREQ**2 for a in ARMATURE]
DAMPING = [2 * DAMPING_RATIO * a * NATURAL_FREQ for a in ARMATURE]
HRP5P_ACTUATOR_R_CROTCH_Y = IdealPdActuatorCfg(
  target_names_expr=(r"RCY",),
  stiffness=STIFFNESS[0].item(),
  damping=DAMPING[0].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[0].item(),
)

HRP5P_ACTUATOR_R_CROTCH_R = IdealPdActuatorCfg(
  target_names_expr=(r"RCR",),
  stiffness=STIFFNESS[1].item(),
  damping=DAMPING[1].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[1].item(),
)

HRP5P_ACTUATOR_R_CROTCH_P = IdealPdActuatorCfg(
  target_names_expr=(r"RCP",),
  stiffness=STIFFNESS[2].item(),
  damping=DAMPING[2].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[2].item(),
)

HRP5P_ACTUATOR_R_KNEE = IdealPdActuatorCfg(
  target_names_expr=(r"RKP",),
  stiffness=STIFFNESS[3].item(),
  damping=DAMPING[3].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[3].item(),
)

HRP5P_ACTUATOR_R_ANKLE_P = IdealPdActuatorCfg(
  target_names_expr=(r"RAP",),
  stiffness=STIFFNESS[4].item(),
  damping=DAMPING[4].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[4].item(),
)

HRP5P_ACTUATOR_R_ANKLE_R = IdealPdActuatorCfg(
  target_names_expr=(r"RAR",),
  stiffness=STIFFNESS[5].item(),
  damping=DAMPING[5].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[5].item(),
)


HRP5P_ACTUATOR_L_CROTCH_Y = IdealPdActuatorCfg(
  target_names_expr=(r"LCY",),
  stiffness=STIFFNESS[6].item(),
  damping=DAMPING[6].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[6].item(),
)

HRP5P_ACTUATOR_L_CROTCH_R = IdealPdActuatorCfg(
  target_names_expr=(r"LCR",),
  stiffness=STIFFNESS[7].item(),
  damping=DAMPING[7].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[7].item(),
)

HRP5P_ACTUATOR_L_CROTCH_P = IdealPdActuatorCfg(
  target_names_expr=(r"LCP",),
  stiffness=STIFFNESS[8].item(),
  damping=DAMPING[8].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[8].item(),
)

HRP5P_ACTUATOR_L_KNEE = IdealPdActuatorCfg(
  target_names_expr=(r"LKP",),
  stiffness=STIFFNESS[9].item(),
  damping=DAMPING[9].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[9].item(),
)

HRP5P_ACTUATOR_L_ANKLE_P = IdealPdActuatorCfg(
  target_names_expr=(r"LAP",),
  stiffness=STIFFNESS[10].item(),
  damping=DAMPING[10].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[10].item(),
)

HRP5P_ACTUATOR_L_ANKLE_R = IdealPdActuatorCfg(
  target_names_expr=(r"LAR",),
  stiffness=STIFFNESS[11].item(),
  damping=DAMPING[11].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[11].item(),
)


HRP5P_ACTUATOR_CHEST_P = IdealPdActuatorCfg(
  target_names_expr=(r"WP",),
  stiffness=STIFFNESS[12].item(),
  damping=DAMPING[12].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[12].item(),
)

HRP5P_ACTUATOR_CHEST_R = IdealPdActuatorCfg(
  target_names_expr=(r"WR",),
  stiffness=STIFFNESS[13].item(),
  damping=DAMPING[13].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[13].item(),
)
HRP5P_ACTUATOR_CHEST_Y = IdealPdActuatorCfg(
  target_names_expr=(r"WY",),
  stiffness=STIFFNESS[14].item(),
  damping=DAMPING[14].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[14].item(),
)
HRP5P_ACTUATOR_HEAD_Y = IdealPdActuatorCfg(
  target_names_expr=(r"HY",),
  stiffness=STIFFNESS[15].item(),
  damping=DAMPING[15].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[15].item(),
)

HRP5P_ACTUATOR_HEAD_P = IdealPdActuatorCfg(
  target_names_expr=(r"HP",),
  stiffness=STIFFNESS[16].item(),
  damping=DAMPING[16].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[16].item(),
)

HRP5P_ACTUATOR_R_SCAPULA = IdealPdActuatorCfg(
  target_names_expr=(r"RSC",),
  stiffness=STIFFNESS[17].item(),
  damping=DAMPING[17].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[17].item(),
)

HRP5P_ACTUATOR_R_SHOULDER_P = IdealPdActuatorCfg(
  target_names_expr=(r"RSP",),
  stiffness=STIFFNESS[18].item(),
  damping=DAMPING[18].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[18].item(),
)

HRP5P_ACTUATOR_R_SHOULDER_R = IdealPdActuatorCfg(
  target_names_expr=(r"RSR",),
  stiffness=STIFFNESS[19].item(),
  damping=DAMPING[19].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[19].item(),
)


HRP5P_ACTUATOR_R_SHOULDER_Y = IdealPdActuatorCfg(
  target_names_expr=(r"RSY",),
  stiffness=STIFFNESS[20].item(),
  damping=DAMPING[20].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[20].item(),
)

HRP5P_ACTUATOR_R_ELBOW_P = IdealPdActuatorCfg(
  target_names_expr=(r"REP",),
  stiffness=STIFFNESS[21].item(),
  damping=DAMPING[21].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[21].item(),
)

HRP5P_ACTUATOR_R_WRIST_Y = IdealPdActuatorCfg(
  target_names_expr=(r"RWRY",),
  stiffness=STIFFNESS[22].item(),
  damping=DAMPING[22].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[22].item(),
)

HRP5P_ACTUATOR_R_WRIST_R = IdealPdActuatorCfg(
  target_names_expr=(r"RWRR",),
  stiffness=STIFFNESS[23].item(),
  damping=DAMPING[23].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[23].item(),
)

HRP5P_ACTUATOR_R_WRIST_P = IdealPdActuatorCfg(
  target_names_expr=(r"RWRP",),
  stiffness=STIFFNESS[24].item(),
  damping=DAMPING[24].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[24].item(),
)

HRP5P_ACTUATOR_L_SCAPULA = IdealPdActuatorCfg(
  target_names_expr=(r"LSC",),
  stiffness=STIFFNESS[25].item(),
  damping=DAMPING[25].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[25].item(),
)

HRP5P_ACTUATOR_L_SHOULDER_P = IdealPdActuatorCfg(
  target_names_expr=(r"LSP",),
  stiffness=STIFFNESS[26].item(),
  damping=DAMPING[26].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[26].item(),
)

HRP5P_ACTUATOR_L_SHOULDER_R = IdealPdActuatorCfg(
  target_names_expr=(r"LSR",),
  stiffness=STIFFNESS[27].item(),
  damping=DAMPING[27].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[27].item(),
)
HRP5P_ACTUATOR_L_SHOULDER_Y = IdealPdActuatorCfg(
  target_names_expr=(r"LSY",),
  stiffness=STIFFNESS[28].item(),
  damping=DAMPING[28].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[28].item(),
)

HRP5P_ACTUATOR_L_ELBOW_P = IdealPdActuatorCfg(
  target_names_expr=(r"LEP",),
  stiffness=STIFFNESS[29].item(),
  damping=DAMPING[29].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[29].item(),
)
HRP5P_ACTUATOR_L_WRIST_Y = IdealPdActuatorCfg(
  target_names_expr=(r"LWRY",),
  stiffness=STIFFNESS[30].item(),
  damping=DAMPING[30].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[30].item(),
)

HRP5P_ACTUATOR_L_WRIST_R = IdealPdActuatorCfg(
  target_names_expr=(r"LWRR",),
  stiffness=STIFFNESS[31].item(),
  damping=DAMPING[31].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[31].item(),
)
HRP5P_ACTUATOR_L_WRIST_P = IdealPdActuatorCfg(
  target_names_expr=(r"LWRP",),
  stiffness=STIFFNESS[32].item(),
  damping=DAMPING[32].item(),
  effort_limit=EFFORT_LIMIT,
  armature=ARMATURE[32].item(),
)

# Hand yaw + finger joints: actuated like mc_mujoco (passive fingers flop and
# deviate from its dynamics); gains from their PDgains_sim.dat rows.
HRP5P_ACTUATOR_HANDS = IdealPdActuatorCfg(
  target_names_expr=(r"[RL](HDY|IMP|IPIP|IDIP|MMP|MPIP|MDIP|TMP|TPIP|TDIP)",),
  stiffness=693.0,
  damping=6.93,
  effort_limit=EFFORT_LIMIT,
)


HRP5P_ACTUATORS: tuple[IdealPdActuatorCfg, ...] = (
  HRP5P_ACTUATOR_R_CROTCH_Y,
  HRP5P_ACTUATOR_R_CROTCH_R,
  HRP5P_ACTUATOR_R_CROTCH_P,
  HRP5P_ACTUATOR_R_KNEE,
  HRP5P_ACTUATOR_R_ANKLE_P,
  HRP5P_ACTUATOR_R_ANKLE_R,
  HRP5P_ACTUATOR_L_CROTCH_Y,
  HRP5P_ACTUATOR_L_CROTCH_R,
  HRP5P_ACTUATOR_L_CROTCH_P,
  HRP5P_ACTUATOR_L_KNEE,
  HRP5P_ACTUATOR_L_ANKLE_P,
  HRP5P_ACTUATOR_L_ANKLE_R,
  HRP5P_ACTUATOR_CHEST_P,
  HRP5P_ACTUATOR_CHEST_R,
  HRP5P_ACTUATOR_CHEST_Y,
  HRP5P_ACTUATOR_HEAD_Y,
  HRP5P_ACTUATOR_HEAD_P,
  HRP5P_ACTUATOR_R_SCAPULA,
  HRP5P_ACTUATOR_R_SHOULDER_P,
  HRP5P_ACTUATOR_R_SHOULDER_R,
  HRP5P_ACTUATOR_R_SHOULDER_Y,
  HRP5P_ACTUATOR_R_ELBOW_P,
  HRP5P_ACTUATOR_R_WRIST_Y,
  HRP5P_ACTUATOR_R_WRIST_R,
  HRP5P_ACTUATOR_R_WRIST_P,
  HRP5P_ACTUATOR_L_SCAPULA,
  HRP5P_ACTUATOR_L_SHOULDER_P,
  HRP5P_ACTUATOR_L_SHOULDER_R,
  HRP5P_ACTUATOR_L_SHOULDER_Y,
  HRP5P_ACTUATOR_L_WRIST_Y,
  HRP5P_ACTUATOR_L_ELBOW_P,
  HRP5P_ACTUATOR_L_WRIST_R,
  HRP5P_ACTUATOR_L_WRIST_P,
  HRP5P_ACTUATOR_HANDS,
)

##
# Reference joint order from mc_HRP5P (useful when wiring observations/actions).
# The full 53-slot refJointOrder, verified against
# mc_rbdyn.get_robot_module("HRP5P").ref_joint_order(); the mjlab model
# simulates a subset (no fingers), which the action term expands to these
# slots at runtime.
HRP5P_REF_JOINT_ORDER = [
  "RCY",
  "RCR",
  "RCP",
  "RKP",
  "RAP",
  "RAR",
  "LCY",
  "LCR",
  "LCP",
  "LKP",
  "LAP",
  "LAR",
  "WP",
  "WR",
  "WY",
  "HY",
  "HP",
  "RSC",
  "RSP",
  "RSR",
  "RSY",
  "REP",
  "RWRY",
  "RWRR",
  "RWRP",
  "RHDY",
  "RTMP",
  "RTPIP",
  "RTDIP",
  "RIMP",
  "RIPIP",
  "RIDIP",
  "RMMP",
  "RMPIP",
  "RMDIP",
  "LSC",
  "LSP",
  "LSR",
  "LSY",
  "LEP",
  "LWRY",
  "LWRR",
  "LWRP",
  "LHDY",
  "LTMP",
  "LTPIP",
  "LTDIP",
  "LIMP",
  "LIPIP",
  "LIDIP",
  "LMMP",
  "LMPIP",
  "LMDIP",
]

# Initial state from mc_HRP5P _stance (deg -> rad).
HRP5P_INIT_STATE = EntityCfg.InitialStateCfg(
  # z = the module's default attitude; starting higher injects a drop
  # transient at reset (mc_mujoco starts at the controller's posW).
  pos=(0.0, 0.0, 0.79),
  joint_pos={
    "RCY": 0.0,
    "RCR": 0.0,
    "RCP": math.radians(-26.87),
    "RKP": math.radians(50.0),
    "RAP": math.radians(-23.13),
    "RAR": 0.0,
    "LCY": 0.0,
    "LCR": 0.0,
    "LCP": math.radians(-26.87),
    "LKP": math.radians(50.0),
    "LAP": math.radians(-23.13),
    "LAR": 0.0,
    "WP": 0.0,
    "WR": 0.0,
    "WY": 0.0,
    "HY": 0.0,
    "HP": 0.0,
    "RSC": 0.0,
    "RSP": math.radians(60.0),
    "RSR": math.radians(-20.0),
    "RSY": math.radians(-5.0),
    "REP": math.radians(-105.0),
    "RWRY": 0.0,
    "RWRR": math.radians(-20.0),
    "RWRP": 0.0,
    "RHDY": 0.0,
    "RIMP": math.radians(-60.5),
    "RIPIP": 0.0,
    "RIDIP": 0.0,
    "RMMP": math.radians(-60.5),
    "RMPIP": 0.0,
    "RMDIP": 0.0,
    "RTMP": math.radians(60.5),
    "RTPIP": 0.0,
    "RTDIP": 0.0,
    "LSC": 0.0,
    "LSP": math.radians(60.0),
    "LSR": math.radians(20.0),
    "LSY": math.radians(5.0),
    "LEP": math.radians(-105.0),
    "LWRY": 0.0,
    "LWRR": math.radians(20.0),
    "LWRP": 0.0,
    "LHDY": 0.0,
    "LIMP": math.radians(60.5),
    "LIPIP": 0.0,
    "LIDIP": 0.0,
    "LMMP": math.radians(60.5),
    "LMPIP": 0.0,
    "LMDIP": 0.0,
    "LTMP": math.radians(-60.5),
    "LTPIP": 0.0,
    "LTDIP": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_HRP5P_FOOT_COLLISION_EXPR = (
  r"^(left|right)_foot_(left|right)_(toes|heel)_geom_collision$"
)
_HRP5P_BODY_COLLISION_EXPR = r"^hrp5p_collision_.*$"
_HRP5P_ALL_COLLISION_EXPR = (
  r"^((left|right)_foot_(left|right)_(toes|heel)_geom_collision|hrp5p_collision_.*)$"
)

# Feet-ground contacts only.
HRP5P_FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_HRP5P_FOOT_COLLISION_EXPR,),
  condim=3,
  priority=1,
  friction=(0.5,),
  disable_other_geoms=False,
)

# Enable all named collision geoms, including self-collisions.
HRP5P_FULL_COLLISION = CollisionCfg(
  geom_names_expr=(_HRP5P_ALL_COLLISION_EXPR,),
  condim={_HRP5P_FOOT_COLLISION_EXPR: 3, r"^hrp5p_collision_.*$": 1},
  priority={_HRP5P_FOOT_COLLISION_EXPR: 1},
  friction={_HRP5P_FOOT_COLLISION_EXPR: (0.5,)},
  disable_other_geoms=False,
)

# Enable world/body collisions while avoiding robot self-collisions.
HRP5P_FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(_HRP5P_ALL_COLLISION_EXPR,),
  contype=0,
  conaffinity=1,
  condim={_HRP5P_FOOT_COLLISION_EXPR: 3, r"^hrp5p_collision_.*$": 1},
  priority={_HRP5P_FOOT_COLLISION_EXPR: 1},
  friction={_HRP5P_FOOT_COLLISION_EXPR: (0.5,)},
  disable_other_geoms=False,
)

# Default collision mode.
HRP5P_COLLISION = HRP5P_FULL_COLLISION

##
# Final config.
##

HRP5P_ARTICULATION = EntityArticulationInfoCfg(
  actuators=HRP5P_ACTUATORS,
  soft_joint_pos_limit_factor=0.99,
)


def get_hrp5p_robot_cfg() -> EntityCfg:
  """Return a fresh HRP5P EntityCfg."""
  return EntityCfg(
    init_state=HRP5P_INIT_STATE,
    collisions=(HRP5P_COLLISION,),
    spec_fn=get_spec,
    articulation=HRP5P_ARTICULATION,
  )


# Residual action scale = nominal torque limit / stiffness. Uses the nominal
# limits table (the actuators themselves are unclamped for mc_mujoco parity);
# joints without a nominal entry (fingers) get no scale.
HRP5P_ACTION_SCALE: dict[str, float] = {}
for a in HRP5P_ARTICULATION.actuators:
  assert isinstance(a, IdealPdActuatorCfg)
  s = a.stiffness
  for n in a.target_names_expr:
    e = HRP5P_NOMINAL_EFFORT_LIMITS.get(n)
    if e is not None:
      HRP5P_ACTION_SCALE[n] = e / s

# Dampen the action scale to keep joints near the reference pose.
upper_scale = 0.2
lower_scale = 0.2
for name in (
  "HY",
  "HP",
  "RSC",
  "LSC",
  "LSP",
  "LSR",
  "LSY",
  "LEP",
  "WR",
  "WY",
  "WP",
  "RSP",
  "RSR",
  "RSY",
  "REP",
  "RWRY",
  "RWRR",
  "RWRP",
  "LWRY",
  "LWRR",
  "LWRP",
  "LAR",
  "RAR",
  "LCY",
  "LCR",
  "LAP",
  "RAP",
  "RCY",
  "RCR",
):
  if name in HRP5P_ACTION_SCALE:
    HRP5P_ACTION_SCALE[name] *= upper_scale

for name in (
  "LKP",
  "RKP",
  "RCP",
  "LCP",
):
  if name in HRP5P_ACTION_SCALE:
    HRP5P_ACTION_SCALE[name] *= lower_scale

if __name__ == "__main__":
  import mujoco.viewer as viewer
  from mjlab.entity.entity import Entity

  robot = Entity(get_hrp5p_robot_cfg())

  viewer.launch(robot.spec.compile())
