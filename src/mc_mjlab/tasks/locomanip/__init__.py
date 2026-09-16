"""Registration for the Locomanip cart demo and its trainable residual task."""

from __future__ import annotations

import sys
from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from mc_mjlab.rl.runner import McRtcResidualOnPolicyRunner
from mc_mjlab.tasks.evaluation import register_evaluation
from mc_mjlab.tasks.locomanip.evaluation import LOCOMANIP_EVALUATION
from mc_mjlab.tasks.locomanip.locomanip_env_cfg import (
  MC_RTC_YAML,
  locomanip_env_cfg,
)
from mc_mjlab.tasks.locomanip.locomanip_ppo_cfg import locomanip_ppo_cfg
from mc_mjlab.tasks.locomanip.locomanip_residual_env_cfg import (
  locomanip_residual_env_cfg,
)
from mc_mjlab.tasks.naming import get_task_name
from mc_mjlab.tasks.zero_residual.zero_residual_env_cfg import zero_residual_rl_cfg

TASK_DIR = Path(__file__).resolve().parent.name
DEMO_TASK_ID = get_task_name(TASK_DIR, MC_RTC_YAML, "position")
RESIDUAL_TASK_ID = get_task_name(TASK_DIR, MC_RTC_YAML, "position-residual")


def _refuse_to_train_the_demo() -> None:
  """Exit now if the zero-residual demo id was handed to ``train``."""
  if Path(sys.argv[0]).name != "train":
    return
  # Equality, not containment: the trainable id has the demo id as its prefix.
  # The split handles `--task=<id>` as well as a positional id.
  if not any(arg.rsplit("=", 1)[-1] == DEMO_TASK_ID for arg in sys.argv[1:]):
    return
  raise SystemExit(
    f"\n{DEMO_TASK_ID} is the play-only demo: it has no reward to optimise.\n"
    f"Train {RESIDUAL_TASK_ID} instead."
  )


_refuse_to_train_the_demo()

register_mjlab_task(
  task_id=DEMO_TASK_ID,
  env_cfg=locomanip_env_cfg(),
  play_env_cfg=locomanip_env_cfg(play=True),
  rl_cfg=zero_residual_rl_cfg(),
)

register_mjlab_task(
  task_id=RESIDUAL_TASK_ID,
  env_cfg=locomanip_residual_env_cfg(),
  play_env_cfg=locomanip_residual_env_cfg(play=True),
  rl_cfg=locomanip_ppo_cfg(experiment_name=RESIDUAL_TASK_ID),
  runner_cls=McRtcResidualOnPolicyRunner,
)

register_evaluation(RESIDUAL_TASK_ID, LOCOMANIP_EVALUATION)
