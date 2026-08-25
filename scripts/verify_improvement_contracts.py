"""Assert bounded-policy and residual-safety contracts without running simulation."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from tensordict import TensorDict

from mc_mjlab.recovery_authority import (
  RecoveryCalibration,
  RecoveryFilter,
  detector_target,
)
from mc_mjlab.residual_safety import project_residual
from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import _make_env_cfg
from mc_mjlab.tasks.rollout_adaptive_ppo import RolloutAdaptivePPO
from mc_mjlab.tasks.squashed_gaussian import SquashedGaussianDistribution
from mc_mjlab.tasks.zero_init_actor import (
  ZeroInitMLPModel,
  ZeroInitRNNModel,
  mean_head_magnitude,
)


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
  recurrent = ZeroInitRNNModel(
    observation,
    {"actor": ["actor"]},
    "actor",
    4,
    hidden_dims=(16, 8),
    rnn_type="gru",
    rnn_hidden_dim=16,
    distribution_cfg={
      "class_name": ("mc_mjlab.tasks.squashed_gaussian:SquashedGaussianDistribution"),
      "init_std": 0.1,
      "std_range": (0.05, 0.30),
    },
  )
  assert mean_head_magnitude(recurrent, observation) == 0.0


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


def verify_recovery_detector() -> None:
  """Check monotonic scoring, sensor onset, bounded duration, and exact cutoff."""
  path = Path(__file__).parents[1] / "etc" / "recovery_detector.json"
  calibration = RecoveryCalibration.from_json(path)
  centers = torch.tensor(calibration.centers)
  scales = torch.tensor(calibration.scales)
  nominal = centers.unsqueeze(0)
  score, target = detector_target(nominal, calibration)
  assert float(score) == 0.0 and float(target) == 0.0
  for index in range(len(centers)):
    disturbed = nominal.clone()
    disturbed[:, index] += scales[index] * (
      calibration.threshold + calibration.activation_span + 0.1
    )
    raised_score, raised_target = detector_target(disturbed, calibration)
    assert float(raised_score) > float(score)
    assert float(raised_target) == 1.0

  recovery_filter = RecoveryFilter(1, "cpu", calibration)
  dt = 0.02
  for _ in range(math.ceil(calibration.rearm_s / dt) + 1):
    authority = recovery_filter.update(score, nominal[:, 1], target, dt)
  assert float(authority) == 0.0
  onset = nominal[:, 1] + calibration.onset_delta + 0.01
  authority = recovery_filter.update(onset, onset, torch.ones(1), dt)
  assert 0.0 < float(authority) < 1.0
  for _ in range(math.ceil(calibration.max_active_s / dt)):
    authority = recovery_filter.update(onset, onset, torch.ones(1), dt)
  assert float(authority) == 0.0


def verify_rollout_schedule() -> None:
  """Check one-event rate decisions retain rsl_rl's thresholds and bounds."""
  update = RolloutAdaptivePPO.next_learning_rate
  assert update(1.0e-3, 0.05, 0.02) == 1.0e-3 / 1.5
  assert update(1.0e-3, 0.005, 0.02) == 1.5e-3
  assert update(1.0e-3, 0.02, 0.02) == 1.0e-3
  assert update(1.0e-5, 0.05, 0.02) == 1.0e-5
  assert update(1.0e-2, 0.005, 0.02) == 1.0e-2


def verify_environment_variants() -> None:
  """Check deployable observations and staged-randomization configuration."""
  standard = _make_env_cfg("position", num_envs=1, disturbance="none")
  actor = standard.observations["actor"].terms
  critic = standard.observations["critic"].terms
  for name in ("left_foot_lin_vel", "right_foot_lin_vel"):
    assert name not in actor and name in critic
  robust = _make_env_cfg(
    "position", num_envs=1, disturbance="none", randomization_stage=1
  )
  assert {
    "randomize_friction",
    "randomize_pd_gains",
    "randomize_strength",
  } <= robust.events.keys()
  assert all(
    term.delay_max_lag == 1 for term in robust.observations["actor"].terms.values()
  )
  assert all(
    term.delay_max_lag == 0 for term in robust.observations["critic"].terms.values()
  )
  assert robust.scene.entities["robot"].articulation is not None
  assert all(
    actuator.delay_max_lag == 2
    for actuator in robust.scene.entities["robot"].articulation.actuators
  )


def main() -> None:
  """Run every local improvement-contract assertion."""
  verify_distribution()
  verify_zero_initialization()
  verify_projection()
  verify_recovery_detector()
  verify_rollout_schedule()
  verify_environment_variants()
  print("improvement contract assertions passed")


if __name__ == "__main__":
  main()
