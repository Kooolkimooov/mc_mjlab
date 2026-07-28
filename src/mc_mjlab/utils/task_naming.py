from mc_mjlab import MC_RTC_YAML_PATH
from mc_mjlab.utils.mc_rtc_config import get_controller_name, get_main_robot_name

MC_MJLAB_PREFIX: str = "Mc-Mjlab-"


def get_task_name(dir_name: str, suffix: str = "") -> str:
  task_name: str = MC_MJLAB_PREFIX
  task_name += dir_name + "-"
  task_name += get_controller_name(MC_RTC_YAML_PATH) + "-"
  task_name += get_main_robot_name(MC_RTC_YAML_PATH)
  if suffix:
    task_name += "-"
    task_name += suffix
  return task_name.title().replace("_", "-")
