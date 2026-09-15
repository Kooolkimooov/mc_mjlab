"""Deterministic ResidualMPC bridge and blending contracts."""

from __future__ import annotations

import torch

import mc_mjlab.tasks  # noqa: F401
from mc_mjlab.residual_mpc import (
  advance_action_history,
  contact_phases,
  paper_joint_action_scale,
  paper_torque_blend,
)
from mc_mjlab.residual_safety import project_residual


def test_phases() -> None:
  """Verify flat-foot duplication and the support-dependent half-cycle offset."""
  phases = contact_phases(
    torch.tensor([0.0, 0.5]),
    torch.tensor([1.0, 1.0]),
    torch.tensor([0.0, 1.0]),
  )
  expected = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.75, 0.75, 0.25, 0.25]])
  torch.testing.assert_close(phases, expected)
  assert bool(((phases >= 0.0) & (phases <= 1.0)).all())


def test_history() -> None:
  """Verify two physical action snapshots advance in temporal order."""
  current = torch.tensor([[3.0, 4.0]])
  previous = torch.tensor([[1.0, 2.0]])
  new_previous, new_second = advance_action_history(current, previous)
  torch.testing.assert_close(new_previous, current)
  torch.testing.assert_close(new_second, previous)


def test_blending() -> None:
  """Verify PD fallback, paper residual, lambda zero, scales, and projection."""
  # Deliberately distinct: eq (23) references the default posture while the PD
  # fallback tracks the controller, and a single tensor cannot catch a swap.
  controller_q = torch.tensor([[1.0, 2.0]])
  default_q = torch.tensor([[0.7, 1.9]])
  q = torch.tensor([[0.5, 1.5]])
  qd = torch.tensor([[0.2, -0.1]])
  controller_qd = torch.tensor([[0.3, 0.4]])
  controller_tau = torch.tensor([[0.0, 7.0]])
  action = torch.tensor([[0.1, -0.2]])
  kp = torch.tensor([[10.0, 20.0]])
  kd = torch.tensor([[2.0, 4.0]])
  mask = torch.ones_like(action)
  nominal, residual, blended = paper_torque_blend(
    controller_q,
    default_q,
    q,
    qd,
    controller_qd,
    controller_tau,
    action,
    kp,
    kd,
    torch.tensor([0.1]),
    mask,
  )
  torch.testing.assert_close(nominal, torch.tensor([[5.2, 7.0]]))
  # Against the default posture. Referencing the controller target instead --
  # the bug this replaces -- gives [[5.6, 6.4]].
  torch.testing.assert_close(residual, torch.tensor([[2.6, 4.4]]))
  torch.testing.assert_close(blended, torch.tensor([[0.26, 0.44]]))
  _, zero_residual, zero_blended = paper_torque_blend(
    controller_q,
    default_q,
    q,
    qd,
    controller_qd,
    controller_tau,
    torch.zeros_like(action),
    kp,
    kd,
    torch.zeros(1),
    mask,
  )
  assert bool((zero_residual != 0.0).all())
  assert bool((zero_blended == 0.0).all())
  scale = paper_joint_action_scale(
    torch.tensor([[100.0, 1.0]]), torch.tensor([[100.0, 100.0]]), 0.1
  )
  torch.testing.assert_close(scale, torch.tensor([[0.1, 0.02]]))
  effort, executed, projected = project_residual(
    torch.tensor([[9.0, 0.0]]),
    torch.tensor([[3.0, 1.0]]),
    torch.tensor([[-10.0, -10.0]]),
    torch.tensor([[10.0, 10.0]]),
  )
  torch.testing.assert_close(effort, torch.tensor([[10.0, 1.0]]))
  torch.testing.assert_close(executed, torch.tensor([[1.0, 1.0]]))
  assert projected.tolist() == [[True, False]]


def test_tracking_reward_discriminates() -> None:
  """Check the bare prior cannot already score the tracking reward's ceiling."""
  import math

  from mc_mjlab.tasks.residual_mpc.residual_mpc_env_cfg import (
    COMMAND_RANGES,
    LINEAR_TRACKING_SIGMA,
  )

  # Measured prior speed under the training disturbance. docs/residual-mpc.md#COMMAND_RANGES
  prior_speed = 0.243
  top = COMMAND_RANGES[0][1]
  error = ((top - prior_speed) / (1.0 + abs(top))) ** 2
  score = math.exp(-error / LINEAR_TRACKING_SIGMA)
  assert 0.5 < score < 0.8, (
    f"the prior scores {score:.3f} at the top of the command box; sigma "
    f"{LINEAR_TRACKING_SIGMA} leaves it nothing to earn"
  )
