"""Register the HRP5P safety feedback task under a distinct experiment name."""

from __future__ import annotations

import warnings
from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from mc_mjlab.bridge.config import get_controller_name
from mc_mjlab.tasks.evaluation import register_evaluation
from mc_mjlab.tasks.locomanip.evaluation import LOCOMANIP_EVALUATION
from mc_mjlab.tasks.locomanip.locomanip_ppo_cfg import locomanip_ppo_cfg
from mc_mjlab.tasks.locomanip_safety.locomanip_safety_env_cfg import (
  MAX_ITERATIONS,
  SAFETY_CONFIG,
  locomanip_safety_env_cfg,
)
from mc_mjlab.tasks.locomanip_safety.runner import LocomanipSafetyRunner
from mc_mjlab.tasks.naming import get_task_name
from mc_mjlab.utils.controller_install import controller_is_installed

TASK_ID = get_task_name(
  Path(__file__).resolve().parent.name, SAFETY_CONFIG, "position-feedback"
)

controller_name = get_controller_name(SAFETY_CONFIG)
if controller_is_installed(controller_name):
  register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=locomanip_safety_env_cfg(),
    play_env_cfg=locomanip_safety_env_cfg(play=True),
    rl_cfg=locomanip_ppo_cfg(experiment_name=TASK_ID, max_iterations=MAX_ITERATIONS),
    runner_cls=LocomanipSafetyRunner,
  )
  register_evaluation(TASK_ID, LOCOMANIP_EVALUATION)
else:
  warnings.warn(
    f"{TASK_ID} was not registered because {controller_name} is not installed",
    RuntimeWarning,
    stacklevel=2,
  )
