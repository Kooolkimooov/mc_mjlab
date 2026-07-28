"""Task ids for the residual balance task.

One id per control mode: the mode selects a different action *class*, not a
field on one, so it cannot be a CLI override the way every other knob is.

The robot and the base controller are baked into the id because a residual is
only meaningful relative to what it rides on: a policy trained against
``LogisticController_ismpc`` on HRP5P means nothing on a different robot, and
nothing under a different base controller either. Both come from
``etc/mc_rtc.yaml``, so editing that file changes the ids -- which is the
point, since it also changes what a checkpoint is valid against.
"""

from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import (
  residual_balance_position_env_cfg,
  residual_balance_ppo_cfg,
  residual_balance_torque_env_cfg,
)
from mc_mjlab.utils.task_naming import get_task_name

TASK_DIR = Path(__file__).resolve().parent.name

register_mjlab_task(
  task_id=get_task_name(TASK_DIR, "position"),
  env_cfg=residual_balance_position_env_cfg(),
  play_env_cfg=residual_balance_position_env_cfg(play=True),
  rl_cfg=residual_balance_ppo_cfg(),
)

register_mjlab_task(
  task_id=get_task_name(TASK_DIR, "torque"),
  env_cfg=residual_balance_torque_env_cfg(),
  play_env_cfg=residual_balance_torque_env_cfg(play=True),
  rl_cfg=residual_balance_ppo_cfg(),
)
