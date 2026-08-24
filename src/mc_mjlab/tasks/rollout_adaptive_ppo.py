"""PPO with one full-rollout learning-rate decision after each update."""

from __future__ import annotations

import torch
from rsl_rl.algorithms import PPO


class RolloutAdaptivePPO(PPO):
  """Keep LR fixed within an update and adapt once from its full-rollout KL."""

  last_schedule_kl: float = float("nan")

  @staticmethod
  def next_learning_rate(rate: float, kl: float, desired_kl: float) -> float:
    """Apply rsl_rl's adaptive thresholds once to the following rollout."""
    if kl > desired_kl * 2.0:
      return max(1.0e-5, rate / 1.5)
    if 0.0 < kl < desired_kl / 2.0:
      return min(1.0e-2, rate * 1.5)
    return rate

  def update(self) -> dict[str, float]:
    """Optimize at fixed LR, measure all rollout samples, then schedule once."""
    adaptive = self.desired_kl is not None and self.schedule == "adaptive"
    original_schedule = self.schedule
    if adaptive:
      self.schedule = "fixed"
    try:
      losses = super().update()
    finally:
      self.schedule = original_schedule

    schedule_kl = self._full_rollout_kl()
    self.last_schedule_kl = schedule_kl
    losses["schedule_kl"] = schedule_kl
    if adaptive:
      self._adapt_learning_rate(schedule_kl)
    return losses

  def _full_rollout_kl(self) -> float:
    """Measure post-update KL over every valid sample in storage."""
    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(1, 1)
    else:
      generator = self.storage.mini_batch_generator(1, 1)
    hidden_state = self.actor.get_hidden_state()
    with torch.inference_mode():
      batch = next(generator)
      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      assert batch.old_distribution_params is not None
      kl = self.actor.get_kl_divergence(
        batch.old_distribution_params, self.actor.output_distribution_params
      )
      kl_mean = kl.mean()
      if self.is_multi_gpu:
        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
        kl_mean /= self.gpu_world_size
    self.actor.reset(hidden_state=hidden_state)
    return float(kl_mean)

  def _adapt_learning_rate(self, schedule_kl: float) -> None:
    """Update and synchronize the rate for the next rollout."""
    if self.gpu_global_rank == 0:
      self.learning_rate = self.next_learning_rate(
        self.learning_rate, schedule_kl, self.desired_kl
      )
    if self.is_multi_gpu:
      value = torch.tensor(self.learning_rate, device=self.device)
      torch.distributed.broadcast(value, src=0)
      self.learning_rate = float(value)
    for group in self.optimizer.param_groups:
      group["lr"] = self.learning_rate
