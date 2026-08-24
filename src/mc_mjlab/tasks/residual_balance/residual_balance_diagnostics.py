"""PPO diagnostics rsl_rl does not log -- docs/ppo.md#training-diagnostics."""

from __future__ import annotations

from typing import Any

import torch


def ppo_diagnostics(alg: Any, saturation_level: float = 0.99) -> dict[str, float]:
  """Policy movement, value fit and action composition over the rollout just learned."""
  storage = getattr(alg, "storage", None)
  # `update()` clears only the write cursor, so the rollout is still readable --
  # but it is gone one `act()` later, so this has to run before the next rollout.
  if storage is None or getattr(storage, "distribution_params", None) is None:
    return {}
  actor = alg.actor
  with torch.inference_mode():
    policy_mean = actor.distribution.deterministic_output(
      storage.distribution_params[0]
    )
    rollout_actions = storage.actions
    hidden_state = actor.get_hidden_state()
    if actor.is_recurrent:
      batch = next(storage.recurrent_mini_batch_generator(1, 1))
      actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      old_params = batch.old_distribution_params
      actions = batch.actions
      old_log_prob = batch.old_actions_log_prob.reshape(-1)
    else:
      old_params = tuple(p.flatten(0, 1) for p in storage.distribution_params)
      actions = rollout_actions.flatten(0, 1)
      old_log_prob = storage.actions_log_prob.flatten(0, 1).reshape(-1)
      actor(storage.observations.flatten(0, 1), stochastic_output=True)
    values = storage.values.flatten(0, 1).reshape(-1)
    returns = storage.returns.flatten(0, 1).reshape(-1)

    kl = actor.get_kl_divergence(old_params, actor.output_distribution_params)
    ratio = torch.exp(actor.get_output_log_prob(actions).reshape(-1) - old_log_prob)
    actor.reset(hidden_state=hidden_state)

    return {
      # End of the iteration, not the per-minibatch value the adaptive schedule
      # reacts to: this is how far the policy moved in total.
      "approx_kl": kl.mean().item(),
      "clip_fraction": ((ratio - 1.0).abs() > alg.clip_param).float().mean().item(),
      "explained_variance": _explained_variance(values, returns),
      "action_saturation": (actions.abs() >= saturation_level).float().mean().item(),
      "policy_mean_saturation": (policy_mean.abs() >= saturation_level)
      .float()
      .mean()
      .item(),
      "policy_mean_rms": _rms(policy_mean),
      "sampled_action_rms": _rms(rollout_actions),
      "exploration_rms": _rms(rollout_actions - policy_mean),
      "policy_mean_rate_rms": _temporal_rms(policy_mean, storage.dones),
      "sampled_action_rate_rms": _temporal_rms(rollout_actions, storage.dones),
      "schedule_kl": float(getattr(alg, "last_schedule_kl", float("nan"))),
    }


def _rms(values: torch.Tensor) -> float:
  """Root mean square over every component."""
  return torch.sqrt(torch.mean(torch.square(values))).item()


def _temporal_rms(values: torch.Tensor, dones: torch.Tensor) -> float:
  """Root mean square temporal change without crossing episode resets."""
  delta_sq = torch.square(values[1:] - values[:-1])
  valid = ~dones[:-1].bool()
  count = valid.sum() * values.shape[-1]
  if count == 0:
    return float("nan")
  return torch.sqrt((delta_sq * valid).sum() / count).item()


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
