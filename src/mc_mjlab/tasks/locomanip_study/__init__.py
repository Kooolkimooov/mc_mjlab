"""Explicit study registrations independent of the production locomanip task."""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from mc_mjlab.rl.runner import McRtcResidualOnPolicyRunner
from mc_mjlab.tasks.evaluation import register_evaluation
from mc_mjlab.tasks.locomanip.evaluation import LOCOMANIP_EVALUATION
from mc_mjlab.tasks.locomanip.locomanip_ppo_cfg import locomanip_ppo_cfg
from mc_mjlab.tasks.locomanip.profiles import HRP5P
from mc_mjlab.tasks.locomanip_study.config import CONTROL, SPECS, study_cfg
from mc_mjlab.tasks.naming import get_task_name
from mc_mjlab.utils.controller_install import controller_is_installed

MC_RTC_YAML = HRP5P.mc_rtc_yaml
TASK_IDS = {
  spec.suffix: get_task_name("locomanip_study", MC_RTC_YAML, spec.suffix)
  for spec in SPECS
}
CONTROL_TASK_ID = TASK_IDS[CONTROL.suffix]

if controller_is_installed("LocomanipController"):
  for spec in SPECS:
    task_id = TASK_IDS[spec.suffix]
    register_mjlab_task(
      task_id=task_id,
      env_cfg=study_cfg(spec),
      play_env_cfg=study_cfg(spec, play=True),
      rl_cfg=locomanip_ppo_cfg(experiment_name=task_id),
      runner_cls=McRtcResidualOnPolicyRunner,
    )
    register_evaluation(task_id, LOCOMANIP_EVALUATION)
