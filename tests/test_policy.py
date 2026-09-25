"""Deterministic policy contracts."""

from __future__ import annotations

import math

import pytest
import torch
from conftest import requires_controller
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mc_mjlab.mdp.rewards import (
  action_l2,
  requested_action_l2,
  requested_action_rate_l2,
)
from mc_mjlab.rl import runner as runner_module
from mc_mjlab.rl.rollout_adaptive_ppo import (
  RolloutAdaptivePPO,
  normalize_masked_advantages,
)
from mc_mjlab.rl.runner import McRtcResidualOnPolicyRunner
from mc_mjlab.rl.squashed_gaussian import (
  LOG_PROB_FLOOR,
  SquashedGaussianDistribution,
)
from mc_mjlab.rl.zero_init_actor import (
  ZeroInitMLPModel,
  mean_head_magnitude,
)
from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import (
  MC_RTC_YAML,
  make_residual_balance_env_cfg,
)
from mc_mjlab.tasks.residual_balance.residual_balance_ppo_cfg import (
  residual_balance_ppo_cfg,
)

pytestmark = requires_controller(MC_RTC_YAML)


def test_distribution() -> None:
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


def test_residual_balance_exploration_cfg() -> None:
  """Keep residual balance aligned with the powered ResidualMPC exploration fix."""
  cfg = residual_balance_ppo_cfg()
  assert cfg.actor.distribution_cfg["std_range"] == (0.05, 0.15)
  assert cfg.algorithm.entropy_coef == 0.00005


def test_zero_initialization() -> None:
  """Check deterministic output is exactly zero after actor construction."""
  observation = TensorDict({"actor": torch.randn(32, 7)}, batch_size=[32])
  actor = ZeroInitMLPModel(
    observation,
    {"actor": ["actor"]},
    "actor",
    4,
    hidden_dims=(16, 8),
    distribution_cfg={
      "class_name": ("mc_mjlab.rl.squashed_gaussian:SquashedGaussianDistribution"),
      "init_std": 0.1,
      "std_range": (0.05, 0.30),
    },
  )
  assert mean_head_magnitude(actor, observation) == 0.0


def test_rollout_schedule() -> None:
  """Check one-event rate decisions retain rsl_rl's thresholds and bounds."""
  update = RolloutAdaptivePPO.next_learning_rate
  assert update(1.0e-3, 0.05, 0.02) == 1.0e-3 / 1.5
  assert update(1.0e-3, 0.005, 0.02) == 1.5e-3
  assert update(1.0e-3, 0.02, 0.02) == 1.0e-3
  assert update(1.0e-5, 0.05, 0.02) == 1.0e-5
  assert update(1.0e-2, 0.005, 0.02) == 1.0e-2


def test_masked_policy_objective() -> None:
  """Check inactive samples have zero surrogate gradient without dilution."""
  advantages = torch.tensor([1.0, 3.0, 100.0, 200.0]).reshape(2, 2, 1)
  mask = torch.tensor([True, True, False, False]).reshape(2, 2, 1)
  normalized = normalize_masked_advantages(advantages, mask)
  assert torch.equal(normalized[~mask], torch.zeros(2))
  assert torch.isclose(normalized[mask].mean(), torch.tensor(0.0), atol=1.0e-6)
  ratio = torch.ones_like(normalized, requires_grad=True)
  (normalized * ratio).mean().backward()
  assert ratio.grad is not None
  assert torch.equal(ratio.grad[~mask], torch.zeros(2))
  expected = (advantages[mask] - advantages[mask].mean()) / advantages[mask].std()
  assert torch.allclose(ratio.grad[mask], expected / mask.sum())
  assert torch.equal(
    normalize_masked_advantages(advantages, torch.zeros_like(mask)),
    torch.zeros_like(advantages),
  )

  cfg = make_residual_balance_env_cfg("position")
  assert cfg.rewards["residual_magnitude"].func is requested_action_l2
  assert cfg.rewards["residual_rate"].func is requested_action_rate_l2
  assert cfg.metrics["executed_residual_l2"].func is action_l2
  assert cfg.metrics["requested_residual_l2"].func is requested_action_l2

  rollout_obs = TensorDict(
    {"actor": torch.randn(4, 3), "critic": torch.randn(4, 3)}, batch_size=[4]
  )
  obs_groups = {"actor": ["actor"], "critic": ["critic"]}
  distribution_cfg = {
    "class_name": "mc_mjlab.rl.squashed_gaussian:SquashedGaussianDistribution",
    "init_std": 0.1,
    "std_range": (0.05, 0.30),
  }
  actor = ZeroInitMLPModel(
    rollout_obs,
    obs_groups,
    "actor",
    2,
    hidden_dims=(8,),
    distribution_cfg=distribution_cfg,
  )
  critic = MLPModel(rollout_obs, obs_groups, "critic", 1, hidden_dims=(8,))
  storage = RolloutStorage("rl", 4, 3, rollout_obs, [2])
  algorithm = RolloutAdaptivePPO(
    actor,
    critic,
    storage,
    num_learning_epochs=1,
    num_mini_batches=1,
    schedule="fixed",
  )
  authority = torch.zeros(4)
  algorithm.set_actor_update_mask_source(lambda: authority)
  masks = torch.tensor(
    [
      [True, False, False, True],
      [False, True, False, False],
      [True, False, False, False],
    ]
  )
  for mask_row in masks:
    algorithm.act(rollout_obs)
    authority.copy_(mask_row)
    algorithm.process_env_step(
      rollout_obs, torch.randn(4), torch.zeros(4, dtype=torch.bool), {}
    )
  algorithm.compute_returns(rollout_obs)
  assert torch.equal(storage.advantages[~masks.unsqueeze(-1)], torch.zeros(8))
  losses = algorithm.update()
  assert all(math.isfinite(value) for value in losses.values())
  assert storage.step == 0
  assert math.isclose(algorithm.last_actor_update_fraction, 4 / 12, rel_tol=1.0e-6)


def test_log_ratio_cannot_overflow() -> None:
  """Check a saturated action cannot make `exp(new - old)` non-finite."""
  dist = SquashedGaussianDistribution(4, init_std=0.05, std_range=(0.05, 0.30))
  saturated = torch.full((3, 4), 1.0 - torch.finfo(torch.float32).eps)
  saturated[1] *= -1.0

  # Old policy centred, new policy moved far: the worst realistic ratio.
  dist.update(torch.zeros(3, 4))
  old = dist.log_prob(saturated)
  dist.update(torch.full((3, 4), 3.0))
  new = dist.log_prob(saturated)
  assert torch.isfinite(old).all() and torch.isfinite(new).all()
  assert torch.isfinite(torch.exp(new - old)).all(), torch.exp(new - old)
  assert float((new - old).abs().max()) < 88.0

  # The floor binds only where the density is already absurd.
  dist.update(torch.zeros(2, 4))
  ordinary = dist.log_prob(torch.zeros(2, 4))
  assert torch.isfinite(ordinary).all()
  assert float(ordinary.min()) > LOG_PROB_FLOOR

  # Exactly the arithmetic that killed the 2026-08-27 run.
  masked_advantage = torch.zeros(1)
  assert torch.isnan(masked_advantage * torch.tensor([float("inf")])).all()


class _StubWriter:
  """Collector standing in for the summary writer."""

  def __init__(self) -> None:
    self.scalars: dict[str, float] = {}

  def add_scalar(self, name: str, value: float, iteration: int) -> None:
    """Record one scalar the way the real writer would."""
    self.scalars[name] = value


class _StubLogger:
  """Logger whose one call per iteration the runner wraps."""

  def __init__(self) -> None:
    self.writer = _StubWriter()
    self.calls: list[int] = []

  def log(self, it: int) -> str:
    """Stand in for rsl_rl's per-iteration logging call."""
    self.calls.append(it)
    return "logged"


def test_shared_runner_logs_ppo_diagnostics(monkeypatch: pytest.MonkeyPatch) -> None:
  """Check every residual task records the diagnostics, not just residual_balance."""
  runner = McRtcResidualOnPolicyRunner.__new__(McRtcResidualOnPolicyRunner)
  runner.logger = _StubLogger()
  runner.last_ppo_diagnostics = {}
  runner.alg = object()
  wrapped = runner._log_with_diagnostics(runner.logger.log)

  # An algorithm with no rollout leaves the diagnostics empty rather than wrong,
  # and the wrapped call still has to reach the logger underneath.
  assert wrapped(7) == "logged"
  assert runner.logger.calls == [7]
  assert runner.last_ppo_diagnostics == {}
  assert runner.logger.writer.scalars == {}

  monkeypatch.setattr(
    runner_module, "ppo_diagnostics", lambda alg: {"schedule_kl": 0.25}
  )
  assert wrapped(8) == "logged"
  assert runner.logger.calls == [7, 8]
  assert runner.last_ppo_diagnostics == {"schedule_kl": 0.25}
  assert runner.logger.writer.scalars == {"Diagnostics/schedule_kl": 0.25}
