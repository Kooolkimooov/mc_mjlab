"""Assert bounded-policy and residual-safety contracts without running simulation."""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from mc_mjlab.residual_safety import project_residual
from mc_mjlab.tasks.squashed_gaussian import SquashedGaussianDistribution
from mc_mjlab.tasks.zero_init_actor import ZeroInitMLPModel, mean_head_magnitude


def verify_distribution() -> None:
  """Check bounds, density, latent KL, scalar std, and finite gradients."""
  torch.manual_seed(7)
  distribution = SquashedGaussianDistribution(4, init_std=0.1)
  mean = torch.zeros(4096, 4, requires_grad=True)
  distribution.update(mean)
  action = distribution.sample()
  assert bool((action.abs() < 1.0).all())
  latent = torch.atanh(action)
  expected_log_prob = (
    torch.distributions.Normal(mean, distribution.std).log_prob(latent)
    - torch.log1p(-action.square())
  ).sum(-1)
  assert torch.allclose(
    distribution.log_prob(action), expected_log_prob, atol=2e-5, rtol=2e-5
  )
  old = (torch.zeros(3, 4), torch.full((3, 4), 0.1))
  new = (torch.full((3, 4), 0.2), torch.full((3, 4), 0.15))
  expected_kl = torch.distributions.kl_divergence(
    torch.distributions.Normal(*old), torch.distributions.Normal(*new)
  ).sum(-1)
  assert torch.allclose(distribution.kl_divergence(old, new), expected_kl)
  loss = -(distribution.log_prob(action).mean() + 1e-3 * distribution.entropy.mean())
  loss.backward()
  assert distribution.std_param.shape == (1,)
  assert mean.grad is not None and math.isfinite(float(mean.grad.abs().max()))


def verify_zero_initialization() -> None:
  """Check deterministic output is exactly zero after actor construction."""
  observation = TensorDict({"actor": torch.randn(32, 7)}, batch_size=[32])
  actor = ZeroInitMLPModel(
    observation,
    {"actor": ["actor"]},
    "actor",
    4,
    hidden_dims=(16, 8),
    distribution_cfg={
      "class_name": ("mc_mjlab.tasks.squashed_gaussian:SquashedGaussianDistribution"),
      "init_std": 0.1,
      "std_range": (0.05, 0.30),
    },
  )
  assert mean_head_magnitude(actor, observation) == 0.0


def verify_projection() -> None:
  """Check target bounds and exact executed-residual accounting."""
  nominal = torch.tensor([[0.9, 1.2, -1.2, 0.0]])
  residual = torch.tensor([[0.2, -0.1, 0.1, -0.4]])
  lower = torch.full((1, 4), -1.0)
  upper = torch.full((1, 4), 1.0)
  target, executed, projected = project_residual(nominal, residual, lower, upper)
  assert bool((target >= lower).all() and (target <= upper).all())
  assert torch.allclose(executed, torch.tensor([[0.1, 0.0, 0.0, -0.4]]))
  assert torch.equal(projected, torch.tensor([[True, True, True, False]]))
  zero_target, zero_executed, _ = project_residual(
    nominal, torch.zeros_like(residual), lower, upper
  )
  assert torch.equal(zero_executed, torch.zeros_like(residual))
  assert torch.equal(zero_target, nominal.clamp(lower, upper))


def main() -> None:
  """Run every local improvement-contract assertion."""
  verify_distribution()
  verify_zero_initialization()
  verify_projection()
  print("improvement contract assertions passed")


if __name__ == "__main__":
  main()
