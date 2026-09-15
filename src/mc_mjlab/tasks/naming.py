"""Building the registered task ids from the active mc_rtc configuration."""

from __future__ import annotations

from pathlib import Path

from mc_mjlab import MC_RTC_YAML_PATH
from mc_mjlab.bridge.config import get_controller_name, get_main_robot_name

MC_MJLAB_PREFIX: str = "Mc-Mjlab-"


def get_task_name(
  dir_name: str, suffix: str = "", mc_rtc_config_path: Path = MC_RTC_YAML_PATH
) -> str:
  task_name: str = MC_MJLAB_PREFIX
  task_name += dir_name + "-"
  task_name += get_controller_name(mc_rtc_config_path) + "-"
  task_name += get_main_robot_name(mc_rtc_config_path)
  if suffix:
    task_name += "-"
    task_name += suffix
  return task_name.title().replace("_", "-")
