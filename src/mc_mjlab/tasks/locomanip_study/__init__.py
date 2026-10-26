"""Explicit study registrations independent of the production locomanip task."""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from mc_mjlab.rl.runner import McRtcResidualOnPolicyRunner
from mc_mjlab.tasks.evaluation import register_evaluation
from mc_mjlab.tasks.locomanip.evaluation import LOCOMANIP_EVALUATION
from mc_mjlab.tasks.locomanip.locomanip_ppo_cfg import locomanip_ppo_cfg
from mc_mjlab.tasks.locomanip.profiles import HRP5P
from mc_mjlab.tasks.locomanip_study.config import study_control_cfg
from mc_mjlab.tasks.naming import get_task_name
from mc_mjlab.utils.controller_install import controller_is_installed

MC_RTC_YAML = HRP5P.mc_rtc_yaml
CONTROL_TASK_ID = get_task_name("locomanip_study", MC_RTC_YAML, "control")

if controller_is_installed("LocomanipController"):
  register_mjlab_task(
    task_id=CONTROL_TASK_ID,
    env_cfg=study_control_cfg(),
    play_env_cfg=study_control_cfg(play=True),
    rl_cfg=locomanip_ppo_cfg(experiment_name=CONTROL_TASK_ID),
    runner_cls=McRtcResidualOnPolicyRunner,
  )
  register_evaluation(CONTROL_TASK_ID, LOCOMANIP_EVALUATION)
