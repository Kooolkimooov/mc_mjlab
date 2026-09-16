"""Deterministic reward audit contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import torch
from evaluation.reward_audit import (
  RewardAuditRecorder,
  RewardAuditShapeError,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg


class _CountingReward:
  """Return fixed reward values while counting manager evaluations."""

  def __init__(self, values: torch.Tensor) -> None:
    self.values = values
    self.calls = 0

  def __call__(self, _env: Any) -> torch.Tensor:
    """Return the configured per-environment values."""
    self.calls += 1
    return self.values


def _reward_manager(terms: dict[str, RewardTermCfg]) -> RewardManager:
  """Use mjlab's real reward manager without creating a simulator."""
  env = cast(ManagerBasedRlEnv, SimpleNamespace(num_envs=4, device="cpu"))
  return RewardManager(terms, env)


def test_reward_audit() -> None:
  """Check single evaluation, zero-weight restoration, live weights, and shapes."""
  active = _CountingReward(torch.tensor([1.0, 2.0, 3.0, 4.0]))
  inactive = _CountingReward(torch.tensor([0.0, 0.5, 1.0, 1.5]))
  manager = _reward_manager(
    {
      "active": RewardTermCfg(func=active, weight=-2.0),
      "inactive": RewardTermCfg(func=inactive, weight=0.0),
    }
  )
  active = manager.get_term_cfg("active").func
  inactive = manager.get_term_cfg("inactive").func
  original_inactive = inactive
  audit = RewardAuditRecorder(
    manager, {"policy_zero": [0, 1], "checkpoint": [2, 3]}, 0.02
  )
  with audit:
    reward = manager.compute(0.02).clone()
    assert torch.allclose(reward, active.values * -2.0 * 0.02)
    assert inactive.calls == 1
    assert manager._term_cfgs[1].weight == 0.0
    assert bool((manager._episode_sums["inactive"] == 0.0).all())
    assert bool((manager._step_reward[:, 1] == 0.0).all())
    audit.capture_denominator("grounded", torch.tensor([1.0, 1.0, 0.0, 0.0]))
    manager._term_cfgs[0].weight = -4.0
    manager.compute(0.02)
  assert manager._term_cfgs[1].func is original_inactive
  report = audit.report()
  checkpoint = report["arms"]["checkpoint"]
  assert checkpoint["terms"]["active"]["raw"]["mean"] == 3.5
  assert checkpoint["terms"]["inactive"]["effective_weight"]["last"] == 0.0
  assert checkpoint["terms"]["active"]["effective_weight"]["distinct"] == [
    -4.0,
    -2.0,
  ]
  assert checkpoint["conditional_denominators"]["grounded"]["mean"] == 0.0
  assert audit.issues() == []

  bad = _reward_manager(
    {"bad": RewardTermCfg(func=lambda _env: torch.zeros(4, 1), weight=1.0)}
  )
  try:
    with RewardAuditRecorder(bad, {"all": [0, 1, 2, 3]}, 0.02):
      bad.compute(0.02)
  except RewardAuditShapeError:
    pass
  else:
    raise AssertionError("reward audit accepted a broadcastable (num_envs, 1) term")
