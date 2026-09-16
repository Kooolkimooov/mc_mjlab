"""Supported residual-balance task registrations."""

from __future__ import annotations

from pathlib import Path

from mjlab.tasks.registry import register_mjlab_task

from mc_mjlab.tasks.naming import get_task_name
from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import (
  MC_RTC_YAML,
  residual_balance_position_achievement_curriculum_env_cfg,
  residual_balance_position_env_cfg,
  residual_balance_position_matched_impulse_env_cfg,
  residual_balance_torque_env_cfg,
)
from mc_mjlab.tasks.residual_balance.residual_balance_ppo_cfg import (
  residual_balance_ppo_cfg,
)
from mc_mjlab.tasks.residual_balance.residual_balance_runner import (
  ResidualBalanceOnPolicyRunner,
)

TASK_DIR = Path(__file__).resolve().parent.name
POSITION_TASK_ID = get_task_name(TASK_DIR, MC_RTC_YAML, "position")
ANKLE_TASK_ID = get_task_name(TASK_DIR, MC_RTC_YAML, "position-ankle")
ACHIEVEMENT_TASK_ID = get_task_name(
  TASK_DIR, MC_RTC_YAML, "position-ankle-curriculum-achievement"
)
MATCHED_TASK_ID = get_task_name(TASK_DIR, MC_RTC_YAML, "position-ankle-matched-impulse")
PITCH_TASK_ID = get_task_name(
  TASK_DIR, MC_RTC_YAML, "position-ankle-pitch-matched-impulse"
)
TORQUE_TASK_ID = get_task_name(TASK_DIR, MC_RTC_YAML, "torque")

register_mjlab_task(
  task_id=POSITION_TASK_ID,
  env_cfg=residual_balance_position_env_cfg(),
  play_env_cfg=residual_balance_position_env_cfg(play=True),
  rl_cfg=residual_balance_ppo_cfg(experiment_name=POSITION_TASK_ID),
  runner_cls=ResidualBalanceOnPolicyRunner,
)

register_mjlab_task(
  task_id=ANKLE_TASK_ID,
  env_cfg=residual_balance_position_env_cfg(authority_set="ankle"),
  play_env_cfg=residual_balance_position_env_cfg(play=True, authority_set="ankle"),
  rl_cfg=residual_balance_ppo_cfg(experiment_name=ANKLE_TASK_ID),
  runner_cls=ResidualBalanceOnPolicyRunner,
)

register_mjlab_task(
  task_id=ACHIEVEMENT_TASK_ID,
  env_cfg=residual_balance_position_achievement_curriculum_env_cfg(),
  play_env_cfg=residual_balance_position_achievement_curriculum_env_cfg(play=True),
  rl_cfg=residual_balance_ppo_cfg(experiment_name=ACHIEVEMENT_TASK_ID),
  runner_cls=ResidualBalanceOnPolicyRunner,
)

register_mjlab_task(
  task_id=MATCHED_TASK_ID,
  env_cfg=residual_balance_position_matched_impulse_env_cfg(),
  play_env_cfg=residual_balance_position_matched_impulse_env_cfg(play=True),
  rl_cfg=residual_balance_ppo_cfg(experiment_name=MATCHED_TASK_ID),
  runner_cls=ResidualBalanceOnPolicyRunner,
)

register_mjlab_task(
  task_id=PITCH_TASK_ID,
  env_cfg=residual_balance_position_matched_impulse_env_cfg(
    authority_set="ankle_pitch"
  ),
  play_env_cfg=residual_balance_position_matched_impulse_env_cfg(
    play=True, authority_set="ankle_pitch"
  ),
  rl_cfg=residual_balance_ppo_cfg(experiment_name=PITCH_TASK_ID),
  runner_cls=ResidualBalanceOnPolicyRunner,
)

register_mjlab_task(
  task_id=TORQUE_TASK_ID,
  env_cfg=residual_balance_torque_env_cfg(),
  play_env_cfg=residual_balance_torque_env_cfg(play=True),
  rl_cfg=residual_balance_ppo_cfg(experiment_name=TORQUE_TASK_ID),
  runner_cls=ResidualBalanceOnPolicyRunner,
)
