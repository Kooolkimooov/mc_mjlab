#!/usr/bin/env python3
"""Verify deterministic ResidualMPC bridge and blending contracts."""

from __future__ import annotations

import numpy as np
import torch

import mc_mjlab.tasks  # noqa: F401
from mc_mjlab.actions.mc_rtc_controller_host import (
  ControllerHost,
  IoLayout,
  _read_scalar_output,
)
from mc_mjlab.residual_mpc import (
  advance_action_history,
  contact_phases,
  paper_joint_action_scale,
  paper_torque_blend,
)
from mc_mjlab.residual_safety import project_residual


class _Vector:
  """Minimal Eigen Vector3 stand-in for datastore command tests."""

  def __init__(self, x: float, y: float, z: float):
    self._values = (x, y, z)

  def x(self) -> float:
    return self._values[0]

  def y(self) -> float:
    return self._values[1]

  def z(self) -> float:
    return self._values[2]


class _Datastore:
  """Small callable datastore with vector getter/setter pairs."""

  def __init__(self):
    self.values = {
      "get_delta": _Vector(1.0, 2.0, 3.0),
      "get_absolute": _Vector(4.0, 5.0, 6.0),
      "ismpc_walking::support_foot_name": "LeftFootCenter",
    }

  def call(self, name: str, value=None):
    if value is None:
      return self.values[name]
    self.values[name.removeprefix("set_").join(("get_", ""))] = value


class _Control:
  """Controller wrapper exposing the fake datastore."""

  def __init__(self, datastore: _Datastore):
    self._datastore = datastore

  def controller(self):
    return self

  def datastore(self) -> _Datastore:
    return self._datastore


def verify_layout() -> None:
  """Verify scalar output columns do not overlap status or command telemetry."""
  layout = IoLayout(
    num_targets=2,
    named_routing=True,
    has_floating_base_sensor=True,
    use_reset=True,
    feed_accel_fallback=False,
    output_channels=("q", "alpha", "tau"),
    output_vectors=("planned_zmp",),
    output_scalars=("objective", "time", "duration", "support"),
    datastore_scalar_commands=(("get", "set"),),
  )
  assert layout.status_off == 6
  assert layout.vector_off == 7
  assert layout.scalar_output_off == 10
  assert layout.scalar_command_output_off == 14
  assert layout.out_width == 16


def verify_datastore_commands() -> None:
  """Verify baseline-relative delta and absolute vector commands side by side."""
  layout = IoLayout(
    num_targets=1,
    named_routing=False,
    has_floating_base_sensor=False,
    use_reset=False,
    feed_accel_fallback=False,
    datastore_vector_commands=(
      ("get_delta", "set_delta"),
      ("get_absolute", "set_absolute"),
    ),
    datastore_vector_command_is_absolute=(False, True),
  )
  datastore = _Datastore()
  host = object.__new__(ControllerHost)
  host._layout = layout
  host._command_baselines = [[None, None]]
  host._scalar_command_baselines = [[]]
  host._vector_command_is_absolute = (False, True)
  row = np.zeros(layout.in_width)
  row[layout.command_off : layout.command_off + 8] = (
    1.0,
    0.5,
    -0.5,
    1.0,
    1.0,
    -1.0,
    -2.0,
    -3.0,
  )
  host._apply_datastore_commands(_Control(datastore), 0, row)
  delta = datastore.values["get_delta"]
  absolute = datastore.values["get_absolute"]
  assert isinstance(delta, _Vector)
  assert isinstance(absolute, _Vector)
  assert (delta.x(), delta.y(), delta.z()) == (1.5, 1.5, 4.0)
  assert (absolute.x(), absolute.y(), absolute.z()) == (-1.0, -2.0, -3.0)
  row[layout.command_off] = 0.0
  host._apply_datastore_commands(_Control(datastore), 0, row)
  restored = datastore.values["get_delta"]
  assert isinstance(restored, _Vector)
  assert (restored.x(), restored.y(), restored.z()) == (1.0, 2.0, 3.0)
  assert _read_scalar_output(datastore, "ismpc_walking::support_foot_name") == 1.0


def verify_phases() -> None:
  """Verify flat-foot duplication and the support-dependent half-cycle offset."""
  phases = contact_phases(
    torch.tensor([0.0, 0.5]),
    torch.tensor([1.0, 1.0]),
    torch.tensor([0.0, 1.0]),
  )
  expected = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.75, 0.75, 0.25, 0.25]])
  torch.testing.assert_close(phases, expected)
  assert bool(((phases >= 0.0) & (phases <= 1.0)).all())


def verify_history() -> None:
  """Verify two physical action snapshots advance in temporal order."""
  current = torch.tensor([[3.0, 4.0]])
  previous = torch.tensor([[1.0, 2.0]])
  new_previous, new_second = advance_action_history(current, previous)
  torch.testing.assert_close(new_previous, current)
  torch.testing.assert_close(new_second, previous)


def verify_blending() -> None:
  """Verify PD fallback, paper residual, lambda zero, scales, and projection."""
  q_hat = torch.tensor([[1.0, 2.0]])
  q = torch.tensor([[0.5, 1.5]])
  qd = torch.tensor([[0.2, -0.1]])
  controller_qd = torch.tensor([[0.3, 0.4]])
  controller_tau = torch.tensor([[0.0, 7.0]])
  action = torch.tensor([[0.1, -0.2]])
  kp = torch.tensor([[10.0, 20.0]])
  kd = torch.tensor([[2.0, 4.0]])
  mask = torch.ones_like(action)
  nominal, residual, blended = paper_torque_blend(
    q_hat,
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
  torch.testing.assert_close(residual, torch.tensor([[5.6, 6.4]]))
  torch.testing.assert_close(blended, torch.tensor([[0.56, 0.64]]))
  _, zero_residual, zero_blended = paper_torque_blend(
    q_hat,
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


def verify_tracking_reward_discriminates() -> None:
  """Check the bare prior cannot already score the tracking reward's ceiling."""
  import math

  from mc_mjlab.tasks.residual_mpc.residual_mpc_env_cfg import (
    COMMAND_RANGES,
    LINEAR_TRACKING_SIGMA,
  )

  # Measured settled speed of the ISMPC prior. docs/residual-mpc.md#COMMAND_RANGES
  prior_speed = 0.269
  top = COMMAND_RANGES[0][1]
  error = ((top - prior_speed) / (1.0 + abs(top))) ** 2
  score = math.exp(-error / LINEAR_TRACKING_SIGMA)
  assert 0.5 < score < 0.8, (
    f"the prior scores {score:.3f} at the top of the command box; sigma "
    f"{LINEAR_TRACKING_SIGMA} leaves it nothing to earn"
  )


def main() -> None:
  """Run every deterministic ResidualMPC contract."""
  verify_layout()
  verify_datastore_commands()
  verify_phases()
  verify_history()
  verify_blending()
  verify_tracking_reward_discriminates()
  print("ResidualMPC deterministic contracts: PASS")


if __name__ == "__main__":
  main()
