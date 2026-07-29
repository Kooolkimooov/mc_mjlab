"""Task ids for the zero-residual demo, one per control mode.

Driven by ``scripts/demos/run_test_mc_rtc.sh``, which runs mjlab's ``play
--agent zero`` against these: the zero action is the "no residual" part, so the
robot tracks raw mc_rtc output and a healthy run holds a steady root height.
"""

import sys
from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from mc_mjlab.tasks.zero_residual.zero_residual_env_cfg import (
  zero_residual_position_env_cfg,
  zero_residual_rl_cfg,
  zero_residual_torque_env_cfg,
)
from mc_mjlab.utils.task_naming import get_task_name

TASK_DIR: str = Path(__file__).resolve().parent.name


def _refuse_to_train() -> None:
  """Exit now if one of these ids was handed to ``train``."""
  if Path(sys.argv[0]).name != "train":
    return
  # Scan every argument rather than argv[1]: `train` with no task id opens an
  # interactive picker, and indexing would raise IndexError -- which mjlab's
  # loader catches, silently dropping this package's registrations.
  if not any("Zero-Residual" in arg for arg in sys.argv[1:]):
    return
  raise SystemExit(
    "\nZero-Residual is a play-only task and cannot be trained.\n"
    "It has no reward term to optimise, and it is built to be driven by a zero\n"
    "action"
  )


_refuse_to_train()

register_mjlab_task(
  task_id=get_task_name(TASK_DIR, "position"),
  env_cfg=zero_residual_position_env_cfg(),
  play_env_cfg=zero_residual_position_env_cfg(play=True),
  rl_cfg=zero_residual_rl_cfg(),
)

register_mjlab_task(
  task_id=get_task_name(TASK_DIR, "torque"),
  env_cfg=zero_residual_torque_env_cfg(),
  play_env_cfg=zero_residual_torque_env_cfg(play=True),
  rl_cfg=zero_residual_rl_cfg(),
)
