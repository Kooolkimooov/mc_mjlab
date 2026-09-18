"""The per-robot Locomanip inputs that no mc_rtc module can supply."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from mc_mjlab import MC_RTC_CONFIG_PATH


@dataclass(frozen=True)
class LocomanipProfile:
  """One robot's mc_rtc profile plus the two scene values that go with it."""

  mc_rtc_yaml: Path
  """The profile whose ``MainRobot`` decides the robot, and so the task id."""

  cart_init_x: float
  """Where the cart stands, bounded by the arms' reach. docs/locomanip.md#cart_init_x"""

  hand_bodies: dict[str, str]
  """Side to the MuJoCo body whose subtree grasps -- not an mc_rtc frame name."""


HRP5P = LocomanipProfile(
  mc_rtc_yaml=MC_RTC_CONFIG_PATH / "mc_rtc_hrp5_locomanip_patched.yaml",
  cart_init_x=0.90,
  hand_bodies={"left": "Lhand_Link0_Plan2", "right": "Rhand_Link0_Plan2"},
)

JVRC1 = LocomanipProfile(
  mc_rtc_yaml=MC_RTC_CONFIG_PATH / "mc_rtc_jvrc1_locomanip_patched.yaml",
  cart_init_x=0.75,
  hand_bodies={"left": "L_WRIST_Y_S", "right": "R_WRIST_Y_S"},
)

#: Every robot the task registers, in registration order.
PROFILES: tuple[LocomanipProfile, ...] = (HRP5P, JVRC1)
