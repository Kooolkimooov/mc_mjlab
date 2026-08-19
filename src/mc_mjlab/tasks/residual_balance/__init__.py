"""Task ids for the residual balance task, one per control mode."""

from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import (
  residual_balance_position_env_cfg,
  residual_balance_torque_env_cfg,
)
from mc_mjlab.tasks.residual_balance.residual_balance_ppo_cfg import (
  residual_balance_ppo_cfg,
)
from mc_mjlab.tasks.residual_balance.residual_balance_runner import (
  ResidualBalanceOnPolicyRunner,
)
from mc_mjlab.utils.task_naming import get_task_name

TASK_DIR = Path(__file__).resolve().parent.name
POSITION_TASK_ID = get_task_name(TASK_DIR, "position")
TORQUE_TASK_ID = get_task_name(TASK_DIR, "torque")

register_mjlab_task(
  task_id=POSITION_TASK_ID,
  env_cfg=residual_balance_position_env_cfg(),
  play_env_cfg=residual_balance_position_env_cfg(play=True),
  rl_cfg=residual_balance_ppo_cfg(experiment_name=POSITION_TASK_ID),
  runner_cls=ResidualBalanceOnPolicyRunner,
)

register_mjlab_task(
  task_id=TORQUE_TASK_ID,
  env_cfg=residual_balance_torque_env_cfg(),
  play_env_cfg=residual_balance_torque_env_cfg(play=True),
  rl_cfg=residual_balance_ppo_cfg(experiment_name=TORQUE_TASK_ID),
  runner_cls=ResidualBalanceOnPolicyRunner,
)
