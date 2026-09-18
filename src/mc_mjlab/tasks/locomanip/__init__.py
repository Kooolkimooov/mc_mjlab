"""Registration for the Locomanip cart demo and its trainable residual task."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from mc_mjlab.bridge.config import get_controller_name, get_main_robot_name
from mc_mjlab.rl.runner import McRtcResidualOnPolicyRunner
from mc_mjlab.tasks.evaluation import register_evaluation
from mc_mjlab.tasks.locomanip.evaluation import LOCOMANIP_EVALUATION
from mc_mjlab.tasks.locomanip.locomanip_env_cfg import locomanip_env_cfg
from mc_mjlab.tasks.locomanip.locomanip_feedback_env_cfg import (
  locomanip_feedback_env_cfg,
)
from mc_mjlab.tasks.locomanip.locomanip_ppo_cfg import locomanip_ppo_cfg
from mc_mjlab.tasks.locomanip.locomanip_residual_env_cfg import (
  locomanip_residual_env_cfg,
)
from mc_mjlab.tasks.locomanip.profiles import PROFILES, LocomanipProfile
from mc_mjlab.tasks.naming import get_task_name
from mc_mjlab.tasks.zero_residual.zero_residual_env_cfg import zero_residual_rl_cfg
from mc_mjlab.utils.controller_install import controller_is_installed

TASK_DIR = Path(__file__).resolve().parent.name

#: Every profile names it, since the cart is one controller's object.
CONTROLLER_NAME = get_controller_name(PROFILES[0].mc_rtc_yaml)

#: MainRobot to its play-only demo id, one entry per registered profile.
DEMO_TASK_IDS: dict[str, str] = {
  get_main_robot_name(p.mc_rtc_yaml): get_task_name(TASK_DIR, p.mc_rtc_yaml, "position")
  for p in PROFILES
}

#: MainRobot to its trainable residual id, one entry per registered profile.
RESIDUAL_TASK_IDS: dict[str, str] = {
  get_main_robot_name(p.mc_rtc_yaml): get_task_name(
    TASK_DIR, p.mc_rtc_yaml, "position-residual"
  )
  for p in PROFILES
}

FEEDBACK_TASK_IDS: dict[str, str] = {
  get_main_robot_name(p.mc_rtc_yaml): get_task_name(
    TASK_DIR, p.mc_rtc_yaml, "position-feedback"
  )
  for p in PROFILES
}


def _refuse_to_train_the_demo() -> None:
  """Exit now if a zero-residual demo id was handed to ``train``."""
  if Path(sys.argv[0]).name != "train":
    return
  # Equality, not containment: a trainable id has its demo id as its prefix.
  # The split handles `--task=<id>` as well as a positional id.
  requested = {arg.rsplit("=", 1)[-1] for arg in sys.argv[1:]}
  for robot, demo_id in DEMO_TASK_IDS.items():
    if demo_id not in requested:
      continue
    raise SystemExit(
      f"\n{demo_id} is the play-only demo: it has no reward to optimise.\n"
      f"Train {RESIDUAL_TASK_IDS[robot]} instead."
    )


def _registrable_profiles() -> tuple[LocomanipProfile, ...]:
  """Every profile, or none at all when the controller shipping the cart is absent."""
  if controller_is_installed(CONTROLLER_NAME):
    return PROFILES
  # Its cart asset is read while the cfgs are built, so raising here would take
  # every other task's registration down with it. docs/coupling.md#controller_is_installed
  warnings.warn(
    f"{CONTROLLER_NAME} is not installed: no locomanip task is registered",
    stacklevel=2,
  )
  return ()


_refuse_to_train_the_demo()

for profile in _registrable_profiles():
  robot = get_main_robot_name(profile.mc_rtc_yaml)
  residual_id = RESIDUAL_TASK_IDS[robot]

  register_mjlab_task(
    task_id=DEMO_TASK_IDS[robot],
    env_cfg=locomanip_env_cfg(profile),
    play_env_cfg=locomanip_env_cfg(profile, play=True),
    rl_cfg=zero_residual_rl_cfg(),
  )

  register_mjlab_task(
    task_id=residual_id,
    env_cfg=locomanip_residual_env_cfg(profile),
    play_env_cfg=locomanip_residual_env_cfg(profile, play=True),
    # A robot's own id is its experiment name, so a resume cannot cross robots.
    rl_cfg=locomanip_ppo_cfg(experiment_name=residual_id),
    runner_cls=McRtcResidualOnPolicyRunner,
  )

  register_evaluation(residual_id, LOCOMANIP_EVALUATION)

  feedback_id = FEEDBACK_TASK_IDS[robot]
  register_mjlab_task(
    task_id=feedback_id,
    env_cfg=locomanip_feedback_env_cfg(profile),
    play_env_cfg=locomanip_feedback_env_cfg(profile, play=True),
    rl_cfg=locomanip_ppo_cfg(experiment_name=feedback_id),
    runner_cls=McRtcResidualOnPolicyRunner,
  )
  register_evaluation(feedback_id, LOCOMANIP_EVALUATION)
