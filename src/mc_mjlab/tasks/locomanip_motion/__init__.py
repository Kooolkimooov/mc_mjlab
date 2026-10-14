"""Register joint residual policies with explicit controller motion authority."""

from __future__ import annotations

from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from mc_mjlab.bridge.config import get_controller_name, get_main_robot_name
from mc_mjlab.rl.runner import McRtcResidualOnPolicyRunner
from mc_mjlab.tasks.evaluation import register_evaluation
from mc_mjlab.tasks.locomanip.profiles import PROFILES
from mc_mjlab.tasks.locomanip_motion.evaluation import LOCOMANIP_MOTION_EVALUATION
from mc_mjlab.tasks.locomanip_motion.locomanip_motion_env_cfg import (
  locomanip_motion_env_cfg,
)
from mc_mjlab.tasks.locomanip_motion.locomanip_motion_ppo_cfg import (
  locomanip_motion_ppo_cfg,
)
from mc_mjlab.tasks.naming import get_task_name
from mc_mjlab.utils.controller_install import controller_is_installed

TASK_IDS = {
  get_main_robot_name(profile.mc_rtc_yaml): get_task_name(
    Path(__file__).resolve().parent.name, profile.mc_rtc_yaml, "position-residual"
  )
  for profile in PROFILES
}

for profile in PROFILES:
  if not controller_is_installed(get_controller_name(profile.mc_rtc_yaml)):
    continue
  task_id = TASK_IDS[get_main_robot_name(profile.mc_rtc_yaml)]
  register_mjlab_task(
    task_id=task_id,
    env_cfg=locomanip_motion_env_cfg(profile),
    play_env_cfg=locomanip_motion_env_cfg(profile, play=True),
    rl_cfg=locomanip_motion_ppo_cfg(experiment_name=task_id),
    runner_cls=McRtcResidualOnPolicyRunner,
  )
  register_evaluation(task_id, LOCOMANIP_MOTION_EVALUATION)
