"""PPO diagnostics rsl_rl does not log -- docs/ppo.md#training-diagnostics."""

from __future__ import annotations

from typing import Any

import torch

from mc_mjlab.tasks.mdp import RAW_CLIP


def ppo_diagnostics(alg: Any, saturation_level: float = RAW_CLIP) -> dict[str, float]:
  """Policy movement, value fit and action saturation over the rollout just learned."""
  storage = getattr(alg, "storage", None)
  # `update()` clears only the write cursor, so the rollout is still readable --
  # but it is gone one `act()` later, so this has to run before the next rollout.
  if storage is None or getattr(storage, "distribution_params", None) is None:
    return {}
  actor = alg.actor
  with torch.inference_mode():
    old_params = tuple(p.flatten(0, 1) for p in storage.distribution_params)
    actions = storage.actions.flatten(0, 1)
    old_log_prob = storage.actions_log_prob.flatten(0, 1).reshape(-1)
    values = storage.values.flatten(0, 1).reshape(-1)
    returns = storage.returns.flatten(0, 1).reshape(-1)

    actor(storage.observations.flatten(0, 1), stochastic_output=True)
    kl = actor.get_kl_divergence(old_params, actor.output_distribution_params)
    ratio = torch.exp(actor.get_output_log_prob(actions).reshape(-1) - old_log_prob)

    return {
      # End of the iteration, not the per-minibatch value the adaptive schedule
      # reacts to: this is how far the policy moved in total.
      "approx_kl": kl.mean().item(),
      "clip_fraction": ((ratio - 1.0).abs() > alg.clip_param).float().mean().item(),
      "explained_variance": _explained_variance(values, returns),
      "action_saturation": (actions.abs() >= saturation_level).float().mean().item(),
    }


def _explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
  """``1 - Var(returns - values) / Var(returns)``: 0 is as good as predicting a mean."""
  variance = returns.var()
  if variance <= 0.0:
    return float("nan")
  return (1.0 - (returns - values).var() / variance).item()


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
