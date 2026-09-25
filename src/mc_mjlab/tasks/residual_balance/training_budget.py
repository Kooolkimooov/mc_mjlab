"""The iteration budget a residual-balance run is sized against."""

from __future__ import annotations


def training_budget(num_envs: int, train_cfg: dict) -> dict[str, int]:
  """Iterations, policy steps per env and transitions this run is budgeted for."""
  # Iteration counts are not comparable across rollout lengths; these are.
  # docs/ppo.md#training-budget
  steps = int(train_cfg["num_steps_per_env"])
  iterations = int(train_cfg.get("max_iterations", 0))
  return {
    "num_envs": int(num_envs),
    "num_steps_per_env": steps,
    "max_iterations": iterations,
    "policy_steps_per_env": steps * iterations,
    "total_transitions": steps * iterations * int(num_envs),
  }
