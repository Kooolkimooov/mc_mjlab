"""Register the residual-feedback variant of the ResidualMPC task."""

from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from mc_mjlab.tasks.naming import get_task_name
from mc_mjlab.tasks.residual_feedback.residual_feedback_env_cfg import (
  residual_feedback_env_cfg,
)
from mc_mjlab.tasks.residual_mpc.residual_mpc_ppo_cfg import residual_mpc_ppo_cfg

TASK_DIR = Path(__file__).resolve().parent.name
TASK_ID = get_task_name(TASK_DIR, "joint_torque")

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=residual_feedback_env_cfg(),
  play_env_cfg=residual_feedback_env_cfg(play=True),
  rl_cfg=residual_mpc_ppo_cfg(experiment_name=TASK_ID),
)
