"""Opt-in registrations for historical residual-balance ablations."""

from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import (
  residual_balance_position_curriculum_env_cfg,
  residual_balance_position_env_cfg,
  residual_balance_position_velocity_env_cfg,
)
from mc_mjlab.tasks.residual_balance.residual_balance_ppo_cfg import (
  residual_balance_ppo_cfg,
)
from mc_mjlab.tasks.residual_balance.residual_balance_runner import (
  ResidualBalanceOnPolicyRunner,
)
from utils.task_naming import get_task_name

TASK_DIR = Path(__file__).resolve().parent.name


def _register(task_id: str, env_cfg, play_env_cfg, recurrent: bool = False) -> None:
  """Register one historical configuration under its original task id."""
  register_mjlab_task(
    task_id=task_id,
    env_cfg=env_cfg,
    play_env_cfg=play_env_cfg,
    rl_cfg=residual_balance_ppo_cfg(experiment_name=task_id, recurrent=recurrent),
    runner_cls=ResidualBalanceOnPolicyRunner,
  )


def register_archived_tasks() -> None:
  """Restore historical task ids for old checkpoint loading."""
  task_id = get_task_name(TASK_DIR, "position-velocity")
  _register(
    task_id,
    residual_balance_position_velocity_env_cfg(),
    residual_balance_position_velocity_env_cfg(play=True),
  )
  for authority_set in ("sagittal", "hardware"):
    task_id = get_task_name(TASK_DIR, f"position-{authority_set}")
    _register(
      task_id,
      residual_balance_position_env_cfg(authority_set=authority_set),
      residual_balance_position_env_cfg(play=True, authority_set=authority_set),
    )
  for schedule in ("frozen", "gradual"):
    task_id = get_task_name(TASK_DIR, f"position-ankle-curriculum-{schedule}")
    _register(
      task_id,
      residual_balance_position_curriculum_env_cfg(schedule),
      residual_balance_position_curriculum_env_cfg(schedule, play=True),
    )
  for randomization_stage in (1, 2):
    task_id = get_task_name(TASK_DIR, f"position-robust{randomization_stage}")
    _register(
      task_id,
      residual_balance_position_env_cfg(randomization_stage=randomization_stage),
      residual_balance_position_env_cfg(
        play=True, randomization_stage=randomization_stage
      ),
    )
  for controller_history in (10, 5):
    task_id = get_task_name(TASK_DIR, f"position-history{controller_history}")
    _register(
      task_id,
      residual_balance_position_env_cfg(controller_history=controller_history),
      residual_balance_position_env_cfg(
        play=True, controller_history=controller_history
      ),
    )
  task_id = get_task_name(TASK_DIR, "position-gru256")
  _register(
    task_id,
    residual_balance_position_env_cfg(controller_history=1, proprio_history=1),
    residual_balance_position_env_cfg(
      play=True, controller_history=1, proprio_history=1
    ),
    recurrent=True,
  )
