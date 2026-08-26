"""Task ids for the residual balance task, one per control mode."""

from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import (
  residual_balance_position_achievement_curriculum_env_cfg,
  residual_balance_position_curriculum_env_cfg,
  residual_balance_position_env_cfg,
  residual_balance_position_velocity_env_cfg,
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
POSITION_VELOCITY_TASK_ID = get_task_name(TASK_DIR, "position-velocity")

register_mjlab_task(
  task_id=POSITION_TASK_ID,
  env_cfg=residual_balance_position_env_cfg(),
  play_env_cfg=residual_balance_position_env_cfg(play=True),
  rl_cfg=residual_balance_ppo_cfg(experiment_name=POSITION_TASK_ID),
  runner_cls=ResidualBalanceOnPolicyRunner,
)

register_mjlab_task(
  task_id=POSITION_VELOCITY_TASK_ID,
  env_cfg=residual_balance_position_velocity_env_cfg(),
  play_env_cfg=residual_balance_position_velocity_env_cfg(play=True),
  rl_cfg=residual_balance_ppo_cfg(experiment_name=POSITION_VELOCITY_TASK_ID),
  runner_cls=ResidualBalanceOnPolicyRunner,
)

for authority_set in ("ankle", "sagittal", "hardware"):
  task_id = get_task_name(TASK_DIR, f"position-{authority_set}")
  register_mjlab_task(
    task_id=task_id,
    env_cfg=residual_balance_position_env_cfg(authority_set=authority_set),
    play_env_cfg=residual_balance_position_env_cfg(
      play=True, authority_set=authority_set
    ),
    rl_cfg=residual_balance_ppo_cfg(experiment_name=task_id),
    runner_cls=ResidualBalanceOnPolicyRunner,
  )

for schedule in ("frozen", "gradual"):
  task_id = get_task_name(TASK_DIR, f"position-ankle-curriculum-{schedule}")
  register_mjlab_task(
    task_id=task_id,
    env_cfg=residual_balance_position_curriculum_env_cfg(schedule),
    play_env_cfg=residual_balance_position_curriculum_env_cfg(schedule, play=True),
    rl_cfg=residual_balance_ppo_cfg(experiment_name=task_id),
    runner_cls=ResidualBalanceOnPolicyRunner,
  )

ACHIEVEMENT_TASK_ID = get_task_name(TASK_DIR, "position-ankle-curriculum-achievement")
register_mjlab_task(
  task_id=ACHIEVEMENT_TASK_ID,
  env_cfg=residual_balance_position_achievement_curriculum_env_cfg(),
  play_env_cfg=residual_balance_position_achievement_curriculum_env_cfg(play=True),
  rl_cfg=residual_balance_ppo_cfg(experiment_name=ACHIEVEMENT_TASK_ID),
  runner_cls=ResidualBalanceOnPolicyRunner,
)

for randomization_stage in (1, 2):
  task_id = get_task_name(TASK_DIR, f"position-robust{randomization_stage}")
  register_mjlab_task(
    task_id=task_id,
    env_cfg=residual_balance_position_env_cfg(randomization_stage=randomization_stage),
    play_env_cfg=residual_balance_position_env_cfg(
      play=True, randomization_stage=randomization_stage
    ),
    rl_cfg=residual_balance_ppo_cfg(experiment_name=task_id),
    runner_cls=ResidualBalanceOnPolicyRunner,
  )

for controller_history in (10, 5):
  task_id = get_task_name(TASK_DIR, f"position-history{controller_history}")
  register_mjlab_task(
    task_id=task_id,
    env_cfg=residual_balance_position_env_cfg(controller_history=controller_history),
    play_env_cfg=residual_balance_position_env_cfg(
      play=True, controller_history=controller_history
    ),
    rl_cfg=residual_balance_ppo_cfg(experiment_name=task_id),
    runner_cls=ResidualBalanceOnPolicyRunner,
  )

GRU_TASK_ID = get_task_name(TASK_DIR, "position-gru256")
register_mjlab_task(
  task_id=GRU_TASK_ID,
  env_cfg=residual_balance_position_env_cfg(controller_history=1, proprio_history=1),
  play_env_cfg=residual_balance_position_env_cfg(
    play=True, controller_history=1, proprio_history=1
  ),
  rl_cfg=residual_balance_ppo_cfg(experiment_name=GRU_TASK_ID, recurrent=True),
  runner_cls=ResidualBalanceOnPolicyRunner,
)

register_mjlab_task(
  task_id=TORQUE_TASK_ID,
  env_cfg=residual_balance_torque_env_cfg(),
  play_env_cfg=residual_balance_torque_env_cfg(play=True),
  rl_cfg=residual_balance_ppo_cfg(experiment_name=TORQUE_TASK_ID),
  runner_cls=ResidualBalanceOnPolicyRunner,
)
