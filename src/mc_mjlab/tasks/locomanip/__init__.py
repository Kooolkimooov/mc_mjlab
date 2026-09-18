"""Registration for the Locomanip cart demo and its trainable residual task."""

from __future__ import annotations

import sys
from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from mc_mjlab.bridge.config import get_main_robot_name
from mc_mjlab.rl.runner import McRtcResidualOnPolicyRunner
from mc_mjlab.tasks.evaluation import register_evaluation
from mc_mjlab.tasks.locomanip.evaluation import LOCOMANIP_EVALUATION
from mc_mjlab.tasks.locomanip.locomanip_env_cfg import locomanip_env_cfg
from mc_mjlab.tasks.locomanip.locomanip_ppo_cfg import locomanip_ppo_cfg
from mc_mjlab.tasks.locomanip.locomanip_residual_env_cfg import (
  locomanip_residual_env_cfg,
)
from mc_mjlab.tasks.locomanip.profiles import PROFILES
from mc_mjlab.tasks.naming import get_task_name
from mc_mjlab.tasks.zero_residual.zero_residual_env_cfg import zero_residual_rl_cfg

TASK_DIR = Path(__file__).resolve().parent.name

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


_refuse_to_train_the_demo()

for profile in PROFILES:
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
