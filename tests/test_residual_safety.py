"""Deterministic residual safety contracts."""

from __future__ import annotations

import math
from typing import Any

import torch

from mc_mjlab.mdp.rewards import (
  requested_action_l2,
  requested_action_rate_l2,
)
from mc_mjlab.residuals.recovery_authority import (
  RecoveryCalibration,
  RecoveryFilter,
  detector_target,
)
from mc_mjlab.residuals.safety import project_residual
from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import (
  RECOVERY_DETECTOR_PATH,
)


class _RequestPricingEnv:
  """Expose only the action-manager lookup the request-pricing rewards perform."""

  def __init__(self, term: Any) -> None:
    self.action_manager = _RequestPricingActions(term)


class _RequestPricingActions:
  """Return one residual term for every action name."""

  def __init__(self, term: Any) -> None:
    self.term = term

  def get_term(self, name: str) -> Any:
    """Return the single fake residual term."""
    del name
    return self.term


def _request_pricing_env(
  request: torch.Tensor,
  previous_request: torch.Tensor,
  gate: torch.Tensor,
  previous_gate: torch.Tensor,
) -> _RequestPricingEnv:
  """Build the buffer surface the request-pricing rewards read."""
  # Imported here: `mc_mjlab.actions.__init__` pulls the subclasses, so a
  # module-level import of the base cycles through a partial package.
  from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase

  # `residual_term` isinstance-checks, so the double must be the real class.
  term = object.__new__(McRtcResidualActionBase)
  term._residual_raw_actions = request
  term._previous_residual_raw_actions = previous_request
  term._last_gate = gate
  term._previous_gate = previous_gate
  return _RequestPricingEnv(term)


def test_request_pricing() -> None:
  """Check burst onset is not charged the magnitude cost a second time."""
  request = torch.tensor([[0.3, -0.4], [0.3, -0.4], [0.3, -0.4], [0.3, -0.4]])
  previous = torch.tensor([[0.0, 0.0], [0.1, -0.2], [0.3, -0.4], [0.1, -0.2]])
  gate = torch.tensor([1.0, 1.0, 0.0, 0.4])
  previous_gate = torch.tensor([0.0, 1.0, 1.0, 1.0])
  env = _request_pricing_env(request, previous, gate, previous_gate)
  magnitude = requested_action_l2(env)  # ty: ignore[invalid-argument-type]
  rate = requested_action_rate_l2(env)  # ty: ignore[invalid-argument-type]
  assert torch.allclose(magnitude, torch.tensor([0.25, 0.25, 0.0, 0.25]))
  assert torch.allclose(rate, torch.tensor([0.0, 0.08, 0.0, 0.08]))
  assert float(rate[0]) == 0.0 and float(magnitude[0]) > 0.0


def test_projection() -> None:
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


def test_recovery_detector() -> None:
  """Check monotonic scoring, sensor onset, bounded duration, and exact cutoff."""
  calibration = RecoveryCalibration.from_json(RECOVERY_DETECTOR_PATH)
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
