"""PPO with recovery-masked advantages and rollout-level LR adaptation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from rsl_rl.algorithms import PPO


def normalize_masked_advantages(
  advantages: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
  """Normalize selected advantages, zero the rest, and preserve gradient scale."""
  weights = mask.to(dtype=advantages.dtype)
  count = weights.sum()
  mean = (advantages * weights).sum() / count.clamp_min(1.0)
  variance = (torch.square(advantages - mean) * weights).sum()
  variance /= (count - 1.0).clamp_min(1.0)
  normalized = weights * (advantages - mean) / (torch.sqrt(variance) + 1.0e-8)
  return normalized * (weights.numel() / count.clamp_min(1.0))


class RolloutAdaptivePPO(PPO):
  """Train on effective actions and adapt LR once from full-rollout KL."""

  last_schedule_kl: float = float("nan")
  last_actor_update_fraction: float = 1.0

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    super().__init__(*args, **kwargs)
    self._actor_mask_source: Callable[[], torch.Tensor] | None = None
    self._actor_update_masks = torch.zeros(
      self.storage.num_transitions_per_env,
      self.storage.num_envs,
      1,
      dtype=torch.bool,
      device=self.device,
    )
    self._actor_mask_steps = 0

  def set_actor_update_mask_source(self, source: Callable[[], torch.Tensor]) -> None:
    """Read action effectiveness after each environment step."""
    if self.normalize_advantage_per_mini_batch:
      raise ValueError("actor masking requires rollout-level advantage normalization")
    self._actor_mask_source = source

  def process_env_step(
    self,
    obs: Any,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    """Record the applied-action mask beside the ordinary transition."""
    if self._actor_mask_source is not None:
      self.record_actor_update_mask(self._actor_mask_source() > 0.0)
    super().process_env_step(obs, rewards, dones, extras)

  def record_actor_update_mask(self, mask: torch.Tensor) -> None:
    """Store one rollout step's environments with nonzero action authority."""
    if self.storage.step != self._actor_mask_steps:
      raise RuntimeError("actor mask and rollout storage cursors diverged")
    if mask.numel() != self.storage.num_envs:
      raise ValueError("actor update mask must have one value per environment")
    self._actor_update_masks[self._actor_mask_steps].copy_(
      mask.to(device=self.device, dtype=torch.bool).reshape(-1, 1)
    )
    self._actor_mask_steps += 1

  def compute_returns(self, obs: Any) -> None:
    """Compute critic targets and select actor advantages by action authority."""
    super().compute_returns(obs)
    if self._actor_mask_source is None:
      return
    self._require_complete_actor_masks()
    self.storage.advantages.copy_(
      normalize_masked_advantages(self.storage.advantages, self._actor_update_masks)
    )

  @staticmethod
  def next_learning_rate(rate: float, kl: float, desired_kl: float) -> float:
    """Apply rsl_rl's adaptive thresholds once to the following rollout."""
    if kl > desired_kl * 2.0:
      return max(1.0e-5, rate / 1.5)
    if 0.0 < kl < desired_kl / 2.0:
      return min(1.0e-2, rate * 1.5)
    return rate

  def update(self) -> dict[str, float]:
    """Optimize at fixed LR, measure the rollout, then schedule once."""
    actor_masked = self._actor_mask_source is not None
    if actor_masked:
      self._require_complete_actor_masks()
      active = self._actor_update_masks.sum()
      self.last_actor_update_fraction = float(active / self._actor_update_masks.numel())
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
    if actor_masked:
      self._actor_mask_steps = 0
    return losses

  def _full_rollout_kl(self) -> float:
    """Measure post-update KL over every valid rollout sample."""
    generator = (
      self.storage.recurrent_mini_batch_generator(1, 1)
      if self.actor.is_recurrent or self.critic.is_recurrent
      else self.storage.mini_batch_generator(1, 1)
    )
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

  def _require_complete_actor_masks(self) -> None:
    """Reject partial rollouts instead of using stale mask entries."""
    expected = self.storage.num_transitions_per_env
    if self._actor_mask_steps != expected or self.storage.step != expected:
      raise RuntimeError(
        f"actor mask has {self._actor_mask_steps}/{expected} rollout steps"
      )

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
