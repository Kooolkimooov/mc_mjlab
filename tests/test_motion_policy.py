"""Multi-rate PPO likelihoods and independently bounded motion exploration."""

from __future__ import annotations

import math
from typing import cast

import mjlab  # noqa: F401
import pytest
import torch
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mc_mjlab.rl.motion_actor import MotionActor
from mc_mjlab.rl.motion_distribution import MotionGaussianDistribution
from mc_mjlab.rl.rollout_adaptive_ppo import RolloutAdaptivePPO


def test_inactive_motion_has_no_density_entropy_kl_or_gradient() -> None:
  torch.manual_seed(42)
  distribution = MotionGaussianDistribution(3)
  distribution.set_motion_active(torch.tensor([False, True]))
  means = torch.zeros(2, 3, requires_grad=True)
  distribution.update(means)
  actions = distribution.sample().detach()
  assert actions[0, -1] == 0
  torch.testing.assert_close(distribution.std[0], torch.tensor([0.1, 0.1, 0.8]))
  latent = torch.atanh(actions)
  assert distribution._distribution is not None
  component = distribution._distribution.log_prob(latent) - torch.log1p(
    -actions.square()
  )
  expected = component[:, :-1].sum(-1) + torch.tensor([0, 1]) * component[:, -1]
  torch.testing.assert_close(distribution.log_prob(actions), expected)
  changed = actions.clone()
  changed[0, -1] = 0.99
  torch.testing.assert_close(distribution.log_prob(changed), expected)

  old = tuple(value.detach().clone() for value in distribution.params)
  shifted = (old[0].clone(), old[1].clone(), old[2].clone())
  shifted[0][:, -1] = 0.5
  kl = distribution.kl_divergence(old, shifted)
  torch.testing.assert_close(kl, torch.tensor([0.0, 0.5 * (0.5 / 0.8) ** 2]))

  objective = distribution.log_prob(actions) + 0.01 * distribution.entropy
  objective[0].backward(retain_graph=True)
  assert means.grad is not None
  assert distribution.motion_std_param.grad is not None
  assert distribution.std_param.grad is not None
  assert means.grad[0, -1] == 0
  assert distribution.motion_std_param.grad.item() == 0
  assert means.grad[0, :-1].abs().sum() > 0
  assert distribution.std_param.grad.abs().sum() > 0
  means.grad.zero_()
  distribution.motion_std_param.grad.zero_()
  objective[1].backward()
  assert means.grad[1, -1].abs() > 0
  assert distribution.motion_std_param.grad.abs().sum() > 0


def test_independent_exploration_reaches_slowdown_and_hold_without_large_joint_noise() -> (
  None
):
  torch.manual_seed(42)
  distribution = MotionGaussianDistribution(36)
  distribution.set_motion_active(torch.ones(16384))
  distribution.update(torch.zeros(16384, 36))
  actions = distribution.sample()
  assert 0.20 < (actions[:, -1] > 0.5).float().mean() < 0.30
  assert 0.005 < (actions[:, -1] >= 0.95).float().mean() < 0.02
  assert actions[:, :-1].square().mean().sqrt() < 0.11
  with torch.no_grad():
    distribution.motion_std_param.fill_(10)
    distribution.std_param.fill_(-1)
  distribution.update(torch.zeros(16384, 36))
  torch.testing.assert_close(distribution.std[:, -1], torch.ones(16384))
  torch.testing.assert_close(distribution.std[:, 0], torch.full((16384,), 0.05))
  with pytest.raises(ValueError, match="matching pre-action"):
    distribution.set_motion_active(torch.zeros(2))
    distribution.update(torch.zeros(3, 36))


@pytest.mark.parametrize("action_dim", [1, 3])
@pytest.mark.parametrize("make_decisions", [False, True])
def test_real_ppo_replays_decision_masks_in_shuffled_batches_and_across_rollouts(
  action_dim: int, make_decisions: bool
) -> None:
  torch.manual_seed(42)
  obs = TensorDict(
    {
      "actor": torch.randn(4, 3),
      "critic": torch.randn(4, 3),
      "motion_decision": torch.zeros(4, 3),
    },
    batch_size=[4],
  )
  groups = {
    "actor": ["actor", "motion_decision"],
    "critic": ["critic", "motion_decision"],
  }
  actor = MotionActor(
    obs,
    groups,
    "actor",
    action_dim,
    hidden_dims=(8,),
    obs_normalization=True,
    distribution_cfg={
      "class_name": "mc_mjlab.rl.motion_distribution:MotionGaussianDistribution"
    },
  )
  critic = MLPModel(obs, groups, "critic", 1, hidden_dims=(8,))
  storage = RolloutStorage("rl", 4, 5, obs, [action_dim])
  alg = RolloutAdaptivePPO(
    actor, critic, storage, num_learning_epochs=2, num_mini_batches=2
  )
  assert torch.equal(actor(obs), torch.zeros(4, action_dim))
  initial_actor = [p.detach().clone() for p in actor.parameters()]
  initial_critic = [p.detach().clone() for p in critic.parameters()]
  for rollout in range(2):
    for step in range(5):
      obs = obs.clone()
      obs["motion_decision"][:, 0] = (
        ((torch.arange(4) + step + rollout) % 3 == 0) & make_decisions
      ).float()
      with torch.inference_mode():
        actions = alg.act(obs)
      assert alg.transition.distribution_params is not None
      torch.testing.assert_close(
        alg.transition.distribution_params[2][:, 0], obs["motion_decision"][:, 0]
      )
      inactive = cast(torch.Tensor, obs["motion_decision"])[:, 0] == 0
      assert (actions[inactive, -1] == 0).all()
      alg.process_env_step(
        obs, torch.randn(4), torch.tensor([False, False, step == 2, False]), {}
      )
    alg.compute_returns(obs)
    for batch in storage.mini_batch_generator(2, 1):
      assert batch.old_distribution_params is not None
      assert batch.actions is not None and batch.old_actions_log_prob is not None
      actor(batch.observations, stochastic_output=True)
      torch.testing.assert_close(
        actor.output_distribution_params[2], batch.old_distribution_params[2]
      )
      torch.testing.assert_close(
        actor.get_output_log_prob(batch.actions), batch.old_actions_log_prob.squeeze(-1)
      )
    losses = alg.update()
    assert all(math.isfinite(value) for value in losses.values())
    assert storage.step == 0
    assert all(torch.isfinite(p).all() for p in actor.parameters())
  if action_dim == 1 and not make_decisions:
    for initial, parameter in zip(initial_actor, actor.parameters(), strict=True):
      torch.testing.assert_close(initial, parameter, rtol=0, atol=0)
    assert any(
      not torch.equal(initial, parameter)
      for initial, parameter in zip(initial_critic, critic.parameters(), strict=True)
    )


def test_braking_only_distribution_masks_entire_inactive_action() -> None:
  torch.manual_seed(42)
  distribution = MotionGaussianDistribution(1)
  assert not distribution.std_param.requires_grad
  distribution.set_motion_active(torch.tensor([False, True]))
  means = torch.zeros(2, 1, requires_grad=True)
  distribution.update(means)
  actions = distribution.sample().detach()
  assert actions.shape == (2, 1) and actions[0, 0] == 0
  torch.testing.assert_close(distribution.std, torch.full((2, 1), 0.8))
  assert distribution.log_prob(actions)[0] == 0
  assert distribution.entropy[0] == 0
  old = tuple(value.detach().clone() for value in distribution.params)
  shifted = (old[0] + 0.5, old[1], old[2])
  assert distribution.kl_divergence(old, shifted)[0] == 0
  objective = distribution.log_prob(actions) + 0.01 * distribution.entropy
  objective[0].backward(retain_graph=True)
  assert means.grad is not None and means.grad.count_nonzero() == 0
  assert distribution.motion_std_param.grad is not None
  assert distribution.motion_std_param.grad.item() == 0
  objective[1].backward()
  assert means.grad[1, 0] != 0
  assert distribution.motion_std_param.grad.item() != 0
