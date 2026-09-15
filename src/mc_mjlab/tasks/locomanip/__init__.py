"""Registration for the play-only Locomanip cart demo."""

import sys
from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from mc_mjlab.tasks.locomanip.locomanip_env_cfg import (
  LOCOMANIP_CONFIG,
  locomanip_env_cfg,
)
from mc_mjlab.tasks.zero_residual.zero_residual_env_cfg import zero_residual_rl_cfg
from utils.task_naming import get_task_name

if Path(sys.argv[0]).name == "train" and any(
  "Locomanip" in arg for arg in sys.argv[1:]
):
  raise SystemExit("Locomanip is a play-only task; launch play with --agent zero")

register_mjlab_task(
  task_id=get_task_name("locomanip", "position", LOCOMANIP_CONFIG),
  env_cfg=locomanip_env_cfg(),
  play_env_cfg=locomanip_env_cfg(play=True),
  rl_cfg=zero_residual_rl_cfg(),
)
