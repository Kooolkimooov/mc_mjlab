"""Deterministic reward audit contracts."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, cast

import torch
from evaluation.reward_audit import (
  RewardAuditRecorder,
  RewardAuditShapeError,
  kernel_error,
  placement_verdict,
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


def test_target_placement() -> None:
  """Check the kernel inversion, the band verdicts, and that a guard is exempt."""
  # Round-trip: the kernel's own output inverts back to the error behind it.
  for std in (0.05, 0.10, 0.25):
    for error in (0.01, 0.033, 0.10):
      score = math.exp(-((error / std) ** 2))
      assert math.isclose(kernel_error(score, std), error, rel_tol=1e-9)
  assert math.isnan(kernel_error(0.0, 0.10))
  assert math.isnan(kernel_error(1.5, 0.10))
  assert math.isnan(kernel_error(0.5, 0.0))

  def raw(q50: float, nonzero: float = 0.9, q99: float = 0.0) -> dict:
    return {
      "q50": q50,
      "q99": q99,
      "gated_q50": q50,
      "gated_q99": q99,
      "nonzero_fraction": nonzero,
    }

  # A target 1.4x the gated median is the band's centre.
  target = 0.10
  in_band = math.exp(-((target / 1.4 / target) ** 2))
  assert placement_verdict("kernel", target, raw(in_band))["verdict"] == "pass"
  # Below the measurement the ratio pins and the gradient dies.
  assert placement_verdict("kernel", target, raw(0.05))["verdict"] == "saturating"
  # Far above it the kernel floors and the gradient dies the other way.
  assert placement_verdict("kernel", target, raw(0.99))["verdict"] == "floored"

  # The gated quantile is what is judged, so a sparse gate is still measurable.
  sparse = placement_verdict("kernel", target, raw(in_band, nonzero=0.2))
  assert sparse["verdict"] == "pass"
  # A term with no admitted sample at all is not measurable.
  empty = placement_verdict("kernel", target, raw(float("nan"), nonzero=0.0))
  assert empty["verdict"] == "not_measured" and math.isnan(empty["ratio"])

  # A guard passes by reading zero, and the kernel band must not judge it.
  quiet = placement_verdict("guard", 0.8, raw(0.0, nonzero=0.0, q99=0.229))
  assert quiet["verdict"] == "pass"
  assert quiet["ratio"] > 3.0
  armed = placement_verdict("guard", 0.8, raw(0.5, nonzero=0.3, q99=0.58))
  assert armed["verdict"] == "armed"
