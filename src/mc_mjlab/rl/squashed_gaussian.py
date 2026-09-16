"""Tanh-squashed Gaussian output distribution for bounded residual actions."""

from __future__ import annotations

import math

import torch
from rsl_rl.modules.distribution import Distribution
from torch import nn
from torch.distributions import Normal

#: Bounds the log-ratio so `exp` cannot overflow. docs/ppo.md#LOG_PROB_FLOOR
LOG_PROB_FLOOR = -40.0


class _TanhDeterministicOutput(nn.Module):
  """Export-friendly deterministic tanh transform."""

  def forward(self, latent_mean: torch.Tensor) -> torch.Tensor:
    return torch.tanh(latent_mean)


class SquashedGaussianDistribution(Distribution):
  """Diagonal tanh-Gaussian with one bounded learned latent standard deviation."""

  def __init__(
    self,
    output_dim: int,
    init_std: float = 0.1,
    std_range: tuple[float, float] = (0.05, 0.30),
    learn_std: bool = True,
  ) -> None:
    super().__init__(output_dim)
    if not 0.0 < std_range[0] <= init_std <= std_range[1]:
      raise ValueError(
        f"expected 0 < min_std <= init_std <= max_std, got {std_range} and {init_std}"
      )
    self.std_range = std_range
    self.std_param = nn.Parameter(
      torch.tensor([init_std], dtype=torch.float32), requires_grad=learn_std
    )
    self._distribution: Normal | None = None
    self._latent_sample: torch.Tensor | None = None
    Normal.set_default_validate_args(False)

  def update(self, mlp_output: torch.Tensor) -> None:
    """Rebuild the latent Normal from this step's means and the learned std."""
    std = self.std_param.clamp(*self.std_range).expand_as(mlp_output)
    self._distribution = Normal(mlp_output, std)
    self._latent_sample = None

  def sample(self) -> torch.Tensor:
    """Draw one reparameterized latent sample and squash it into bounds."""
    if self._distribution is None:
      raise RuntimeError("update() must be called before sample()")
    self._latent_sample = self._distribution.rsample()
    return torch.tanh(self._latent_sample)

  def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
    """Squash latent means without sampling, for evaluation and export."""
    return torch.tanh(mlp_output)

  def as_deterministic_output_module(self) -> nn.Module:
    """The same squash as an exportable module."""
    return _TanhDeterministicOutput()

  def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    """Log density of squashed actions, floored so the PPO ratio cannot overflow."""
    if self._distribution is None:
      raise RuntimeError("update() must be called before log_prob()")
    eps = torch.finfo(outputs.dtype).eps
    bounded = outputs.clamp(min=-1.0 + eps, max=1.0 - eps)
    latent = torch.atanh(bounded)
    density = (
      self._distribution.log_prob(latent) - self._log_tanh_jacobian(latent)
    ).sum(dim=-1)
    # A saturated action is a ~40-sigma deviate here, so a large policy move can
    # push `exp(new - old)` past float32 in rsl_rl's ratio. Masked advantages are
    # exactly zero, and `0 * inf` is NaN, so one such sample kills the update.
    # docs/ppo.md#LOG_PROB_FLOOR
    return density.clamp(min=LOG_PROB_FLOOR)

  def kl_divergence(
    self,
    old_params: tuple[torch.Tensor, ...],
    new_params: tuple[torch.Tensor, ...],
  ) -> torch.Tensor:
    """KL between the latent Gaussians; the squash is shared and cancels."""
    old_mean, old_std = old_params
    new_mean, new_std = new_params
    old = Normal(old_mean, old_std)
    new = Normal(new_mean, new_std)
    return torch.distributions.kl_divergence(old, new).sum(dim=-1)

  @staticmethod
  def _log_tanh_jacobian(latent: torch.Tensor) -> torch.Tensor:
    """Log |d tanh / d latent|, in the numerically stable softplus form."""
    return 2.0 * (math.log(2.0) - latent - torch.nn.functional.softplus(-2.0 * latent))

  @property
  def input_dim(self) -> int:
    """Latent width the actor MLP must emit."""
    return self.output_dim

  @property
  def mean(self) -> torch.Tensor:
    """Squashed mean action; ``update`` must have run."""
    if self._distribution is None:
      raise RuntimeError("update() must be called before reading mean")
    return torch.tanh(self._distribution.mean)

  @property
  def std(self) -> torch.Tensor:
    """Latent standard deviation; ``update`` must have run."""
    if self._distribution is None:
      raise RuntimeError("update() must be called before reading std")
    return self._distribution.stddev

  @property
  def entropy(self) -> torch.Tensor:
    """Entropy of the squashed distribution; ``sample`` must have run."""
    if self._distribution is None or self._latent_sample is None:
      raise RuntimeError("sample() must be called before reading entropy")
    base = self._distribution.entropy()
    return (base + self._log_tanh_jacobian(self._latent_sample)).sum(dim=-1)

  @property
  def params(self) -> tuple[torch.Tensor, ...]:
    """Latent ``(mean, stddev)`` pair rsl_rl stores for its KL schedule."""
    if self._distribution is None:
      raise RuntimeError("update() must be called before reading params")
    return (self._distribution.mean, self._distribution.stddev)
