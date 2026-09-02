"""Register the ResidualMPC reproduction task."""

from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from mc_mjlab.tasks.residual_mpc.residual_mpc_env_cfg import residual_mpc_env_cfg
from mc_mjlab.tasks.residual_mpc.residual_mpc_ppo_cfg import residual_mpc_ppo_cfg
from mc_mjlab.tasks.residual_mpc.residual_mpc_runner import (
  ResidualMpcOnPolicyRunner,
)
from mc_mjlab.utils.task_naming import get_task_name

TASK_DIR = Path(__file__).resolve().parent.name
TASK_ID = get_task_name(TASK_DIR, "joint_torque")

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=residual_mpc_env_cfg(),
  play_env_cfg=residual_mpc_env_cfg(play=True),
  rl_cfg=residual_mpc_ppo_cfg(experiment_name=TASK_ID),
  # The installed controller yaml, PD gains and workspace patches decide this
  # task's behaviour, so a checkpoint is only meaningful beside them.
  runner_cls=ResidualMpcOnPolicyRunner,
)
