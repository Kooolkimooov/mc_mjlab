"""Independent motion exploration with likelihoods only at motion decisions."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from mc_mjlab.rl.squashed_gaussian import SquashedGaussianDistribution, log_prob_floor


class MotionGaussianDistribution(SquashedGaussianDistribution):
  """Keep joint noise small and mask the independently sampled motion channel."""

  def __init__(
    self,
    output_dim: int,
    init_std: float = 0.1,
    std_range: tuple[float, float] = (0.05, 0.15),
    learn_std: bool = True,
    motion_init_std: float = 0.8,
    motion_std_range: tuple[float, float] = (0.5, 1.0),
  ) -> None:
    super().__init__(output_dim, init_std, std_range, learn_std)
    if output_dim < 1:
      raise ValueError("motion distribution requires at least one output")
    if output_dim == 1:
      self.std_param.requires_grad_(False)
    if not 0 < motion_std_range[0] <= motion_init_std <= motion_std_range[1]:
      raise ValueError("motion standard deviation must lie within positive bounds")
    self.motion_std_range = motion_std_range
    self.log_prob_floor = log_prob_floor(
      output_dim, min(std_range[0], motion_std_range[0])
    )
    self.motion_std_param = nn.Parameter(
      torch.tensor([motion_init_std]), requires_grad=learn_std
    )
    self._motion_active: torch.Tensor | None = None

  def set_motion_active(self, active: torch.Tensor) -> None:
    """Read the pre-action decision flag from stored, uncorrupted observations."""
    self._motion_active = active.bool().reshape(-1, 1)

  def update(self, mlp_output: torch.Tensor) -> None:
    """Build separate joint and motion latent standard deviations."""
    if (
      self._motion_active is None
      or self._motion_active.shape != mlp_output[..., -1:].shape
    ):
      raise ValueError("a matching pre-action motion mask is required")
    joint_std = self.std_param.clamp(*self.std_range).expand_as(mlp_output[..., :-1])
    motion_std = self.motion_std_param.clamp(*self.motion_std_range).expand_as(
      mlp_output[..., -1:]
    )
    self._distribution = Normal(mlp_output, torch.cat((joint_std, motion_std), dim=-1))
    self._latent_sample = None

  def sample(self) -> torch.Tensor:
    """Return a zero motion placeholder between decisions; the environment holds it."""
    sampled = super().sample()
    assert self._motion_active is not None
    return torch.cat(
      (sampled[..., :-1], torch.where(self._motion_active, sampled[..., -1:], 0.0)),
      dim=-1,
    )

  def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    """Count the motion likelihood once, alongside every-step joint likelihoods."""
    if self._distribution is None:
      raise RuntimeError("update() must precede log_prob()")
    eps = torch.finfo(outputs.dtype).eps
    latent = torch.atanh(outputs.clamp(-1 + eps, 1 - eps))
    gaussian = self._reduce(self._distribution.log_prob(latent), self._motion_active)
    jacobian = self._reduce(self._log_tanh_jacobian(latent), self._motion_active)
    return gaussian.clamp(min=self.log_prob_floor) - jacobian

  def kl_divergence(
    self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
  ) -> torch.Tensor:
    """Exclude motion KL on transitions that made no motion decision."""
    old = Normal(old_params[0], old_params[1])
    new = Normal(new_params[0], new_params[1])
    return self._reduce(
      torch.distributions.kl_divergence(old, new), new_params[2].bool()
    )

  @staticmethod
  def _reduce(values: torch.Tensor, active: torch.Tensor | None) -> torch.Tensor:
    """Sum joint components and the eligible motion component."""
    if active is None:
      raise RuntimeError("motion eligibility has not been set")
    return values[..., :-1].sum(dim=-1) + torch.where(
      active, values[..., -1:], 0.0
    ).squeeze(-1)

  @property
  def entropy(self) -> torch.Tensor:
    """Count transformed motion entropy only on actual decision transitions."""
    if self._distribution is None or self._latent_sample is None:
      raise RuntimeError("sample() must precede entropy")
    values = self._distribution.entropy() + self._log_tanh_jacobian(self._latent_sample)
    return self._reduce(values, self._motion_active)

  @property
  def params(self) -> tuple[torch.Tensor, ...]:
    """Store eligibility with the old means and standard deviations in PPO storage."""
    if self._motion_active is None:
      raise RuntimeError("motion eligibility has not been set")
    return (*super().params, self._motion_active.float())
