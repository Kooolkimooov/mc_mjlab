"""Verify native action wiring and scheduling without external controller adapters."""

from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import patch

import mujoco
import numpy as np
import torch

import mc_mjlab.tasks  # noqa: F401
import mc_rtc_interface as native
from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionAction,
)
from mc_mjlab.actions.walking_reference_action import (
  WALKING_REF_VEL_GETTER,
  WALKING_REF_VEL_SETTER,
  GatedWalkingReferenceDeltaAction,
)
from mc_mjlab.residual_printer import ResidualPrinter
from mc_mjlab.sim_controller_bridge import SimControllerBridge


def verify_layout() -> tuple[SimControllerBridge, NS, NS]:
  """Exercise reference-order scatter, quaternion conversion and sensor routing."""
  model = mujoco.MjModel.from_xml_string("""
  <mujoco><worldbody><body name="robot/base"><freejoint name="robot/root"/>
  <geom size=".1" mass="1"/><site name="robot/imu"/>
  <body><joint name="robot/b"/><geom size=".1" mass="1"/>
  <body><joint name="robot/a"/><geom size=".1" mass="1"/></body></body>
  </body></worldbody><sensor>
  <gyro name="robot/IMU_gyro" site="robot/imu"/>
  <accelerometer name="robot/IMU_accelerometer" site="robot/imu"/>
  <force name="robot/foot_fsensor" site="robot/imu"/>
  <torque name="robot/foot_tsensor" site="robot/imu"/>
  </sensor></mujoco>""")
  data = NS(
    qpos=torch.tensor([[11.0, 22.0, 3.0, 0.5, 0.5, 0.5, 0.5, 8.0, 9.0]]).repeat(2, 1),
    qvel=torch.arange(16.0).reshape(2, 8),
    qacc=torch.arange(16.0, 32.0).reshape(2, 8),
    qfrc_actuator=torch.arange(32.0, 48.0).reshape(2, 8),
    sensordata=torch.arange(24.0).reshape(2, 12),
  )
  env = NS(
    num_envs=2,
    device="cpu",
    cfg=NS(frameskip=2, decimation=4),
    sim=NS(mj_model=model, data=data),
    scene=NS(env_origins=torch.tensor([[10.0, 20.0, 0.0]]).repeat(2, 1)),
  )
  entity = NS(
    joint_names=("b", "a"),
    data=NS(
      joint_pos_biased=torch.tensor([[4.0, 5.0], [6.0, 7.0]]),
      joint_vel=torch.tensor([[8.0, 9.0], [10.0, 11.0]]),
    ),
  )

  def sensor(name: str) -> NS:
    return NS(name=lambda: name)

  module = NS(
    bodySensors=lambda: [sensor("FloatingBase"), sensor("IMU")],
    forceSensors=lambda: [sensor("foot")],
  )
  with (
    patch(
      "mc_mjlab.sim_controller_bridge.robots.get_ref_joint_order",
      return_value=("a", "missing", "b"),
    ),
    patch(
      "mc_mjlab.sim_controller_bridge.robots.get_default_joint_positions",
      return_value={"missing": 0.7},
    ),
    patch(
      "mc_mjlab.sim_controller_bridge.robots.get_robot_module", return_value=module
    ),
  ):
    bridge = SimControllerBridge(
      env,  # ty: ignore[invalid-argument-type]
      entity,  # ty: ignore[invalid-argument-type]
      ["b", "a"],
      torch.tensor([0, 1]),
      "test",
      ("q", "alpha", "tau"),
      "robot",
    )
  rows = np.zeros((2, bridge.layout.input_size))
  bridge.set_feedback_offset(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
  bridge.set_joint_velocity_offset(torch.tensor([[0.2, 0.3], [0.4, 0.5]]))
  bridge.set_wrench_offset(torch.ones(2, 6))
  bridge.fill_controller_input(rows)
  layout = bridge.layout.input
  np.testing.assert_allclose(rows[:, :3], [[5.2, 0.7, 4.1], [7.4, 0.7, 6.3]])
  np.testing.assert_allclose(rows[:, 3:6], [[9.3, 0, 8.2], [11.5, 0, 10.4]])
  np.testing.assert_allclose(rows[:, 6:9], [[39, 0, 38], [47, 0, 46]])
  ro = layout.root_offset()
  np.testing.assert_allclose(rows[:, ro : ro + 3], [[1, 2, 3]] * 2)
  data.qpos[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
  bridge.fill_controller_input(rows)
  np.testing.assert_allclose(rows[:, ro + 3 : ro + 7], [[0, 0, 0, 1]] * 2)
  bo = layout.body_sensors_offset()
  np.testing.assert_allclose(rows[:, bo : bo + 3], data.qvel[:, 3:6])
  np.testing.assert_allclose(rows[:, bo + 3 : bo + 6], data.qacc[:, :3])
  np.testing.assert_allclose(rows[:, bo + 6 : bo + 12], data.sensordata[:, :6])
  fo = layout.force_sensors_offset()
  np.testing.assert_allclose(rows[:, fo : fo + 6], data.sensordata[:, 6:12] + 1)
  bridge.set_root_pose_offset(torch.ones(2, 3), torch.tensor([[0.0, 0.1, 0.0]] * 2))
  bridge.fill_controller_input(rows)
  np.testing.assert_allclose(rows[:, ro : ro + 3], [[2, 3, 4]] * 2)
  assert not np.allclose(rows[:, bo + 9 : bo + 12], data.sensordata[:, 3:6])
  np.testing.assert_allclose(np.linalg.norm(rows[:, ro + 3 : ro + 7], axis=1), 1)
  out = np.arange(2.0 * bridge.layout.output_size).reshape(2, -1)
  block = bridge.upload_controller_output(out)
  for channel, offset in (("q", 0), ("alpha", 3), ("tau", 6)):
    np.testing.assert_allclose(
      bridge.read_controller_output(block)[channel], out[:, [offset + 2, offset]]
    )
  return bridge, env, entity


def verify_walking_reference_feed() -> None:
  """Check the nominal latch, the hold while perturbing and the absolute mode."""
  action = object.__new__(GatedWalkingReferenceDeltaAction)
  action._env = NS(num_envs=2)  # ty: ignore[invalid-assignment]
  zero = torch.zeros(2, 3)
  action._walking_reference_executed = zero.clone()
  action._previous_walking_reference_executed = zero.clone()
  action._walking_reference_nominal = zero.clone()
  action._walking_reference_fed = zero.clone()
  action._datastore_output_fresh = torch.ones(2, dtype=torch.bool)
  nominal = torch.tensor([[0.1, 0.0, 0.0]] * 2)
  action._datastore_vector_outputs = {WALKING_REF_VEL_GETTER: nominal.clone()}
  action._datastore_vector_input_columns = {WALKING_REF_VEL_SETTER: 0}
  action._datastore_vector_input_feed = torch.zeros(2, 1, 3)

  fed = action._datastore_vector_input_feed[:, 0]
  action._walking_reference_executed.copy_(torch.tensor([[0.05, 0.0, 0.0]] * 2))
  action._feed_walking_reference()
  torch.testing.assert_close(fed, torch.tensor([[0.15, 0.0, 0.0]] * 2))

  # The getter mirrors what the term fed, so the nominal must not follow it.
  action._datastore_vector_outputs[WALKING_REF_VEL_GETTER].copy_(fed)
  action._walking_reference_executed.copy_(torch.tensor([[0.02, 0.0, 0.0]] * 2))
  action._feed_walking_reference()
  torch.testing.assert_close(action._walking_reference_nominal, nominal)

  # A reset zeroes the readouts; without fresh output the nominal still holds.
  action._walking_reference_fed.zero_()
  action._datastore_output_fresh.zero_()
  action._datastore_vector_outputs[WALKING_REF_VEL_GETTER].zero_()
  action._feed_walking_reference()
  torch.testing.assert_close(action._walking_reference_nominal, nominal)

  # Mid-policy-step: the controller's own change is latched at the next collect,
  # not lost to the cached write the intervening periods would repeat.
  action._datastore_output_fresh.fill_(True)
  action._walking_reference_executed.zero_()
  action._feed_walking_reference()
  moved = torch.tensor([[0.2, 0.0, 0.0]] * 2)
  action._datastore_vector_outputs[WALKING_REF_VEL_GETTER].copy_(moved)
  action._feed_walking_reference()
  torch.testing.assert_close(action._walking_reference_nominal, moved)
  torch.testing.assert_close(fed, moved)

  absolute = type("Absolute", (GatedWalkingReferenceDeltaAction,), {})
  absolute.walking_reference_is_absolute = True
  action.__class__ = absolute
  action._walking_reference_executed.copy_(torch.tensor([[0.02, 0.0, 0.0]] * 2))
  action._feed_walking_reference()
  torch.testing.assert_close(fed, torch.tensor([[0.02, 0.0, 0.0]] * 2))


class Manager:
  """Deterministic manager that distinguishes failed rows from completed rows."""

  def __init__(self, action: Any) -> None:
    self.action = action
    self.calls = 0
    self.failure = []
    self.inputs = []
    self.resets = []

  def dispatch(self, command: native.Command) -> None:
    assert command == native.Command.Step
    self.calls += 1
    self.inputs.append(self.action._in_np.copy())

  def collect(self) -> list[int]:
    out = self.action._out_np
    out[:] = self.calls * 4
    out[:, self.action._bridge.layout.output.status_offset()] = 0
    return self.failure

  def respawn(self, reset_row_ids: list[int]) -> None:
    self.resets.append(list(reset_row_ids))


def verify_pipeline() -> None:
  """Keep interpolation delayed through partial resets and exclude failed rows."""
  bridge, env, entity = verify_layout()
  action = object.__new__(McRtcResidualJointPositionAction)
  action._env = env  # ty: ignore[invalid-assignment]
  action._entity = entity  # ty: ignore[invalid-assignment]
  action.cfg = NS(  # ty: ignore[invalid-assignment]
    frameskip=2,
    datastore_vectors_outputs=(),
    datastore_scalar_outputs=(),
    datastore_vectors_inputs=(),
    datastore_scalar_inputs=(),
  )
  action._num_targets = 2
  action._target_ids = torch.tensor([0, 1])
  action._bridge = bridge
  bridge._output_channels = action.output_channels
  action._setup_datastore_outputs(action.cfg)
  action._setup_datastore_inputs(action.cfg)
  action._alloc_interpolation_buffers()
  action._alloc_failure_latches()
  action._in_np = np.zeros((2, bridge.layout.input_size))
  action._out_np = np.zeros((2, bridge.layout.output_size))
  action._manager = Manager(action)
  action._pending_dispatch = False
  action._pending_reset = np.zeros(2, dtype=bool)
  action._dispatch_resets = np.zeros(2, dtype=bool)
  action._substep = 0
  action._processed_actions = torch.zeros(2, 2)
  action._last_gate = torch.ones(2)
  action._torque_peak = torch.zeros(2, 2)
  entity.data.qfrc_actuator = torch.zeros(2, 2)
  action._residual_ids = None
  action._executed_physical = torch.zeros(2, 2)
  action._projection_mask = torch.zeros(2, 2, dtype=torch.bool)
  action._printer = ResidualPrinter(0, [], None, "")
  action._raw_actions = torch.zeros(2, 2)
  action._previous_executed_physical = torch.zeros(2, 2)
  action._residual_raw_actions = torch.zeros(2, 2)
  action._previous_residual_raw_actions = torch.zeros(2, 2)
  action._walking_reference_requested = torch.empty(2, 0)
  action._previous_walking_reference_executed = torch.empty(2, 0)
  action._previous_gate = torch.ones(2)
  action._recovery_authority = None
  applied = []

  def apply(
    control: dict[str, torch.Tensor], residual: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    applied.append(control["q"].clone())
    return residual, torch.zeros_like(residual, dtype=torch.bool)

  action._apply_control = apply  # ty: ignore[invalid-assignment]
  for _ in range(4):
    action.apply_actions()
  np.testing.assert_allclose([v[0, 0] for v in applied], [0, 0, 2, 4])
  action.reset(torch.tensor([1]))
  assert action._manager.resets == [[1]]
  assert not action._pending_dispatch
  assert action._has_staged_control.tolist() == [True, False]
  np.testing.assert_allclose(
    action._next_control["q"][1], entity.data.joint_pos_biased[1]
  )
  assert not action._next_control["alpha"][1].any()
  assert action._substep == 4
  action.apply_actions()
  action.apply_actions()
  action._manager.failure = [1]
  action._collect_controller_output()
  assert action.controller_worker_failed.tolist() == [False, True]
  assert action._pending_reset.tolist() == [False, True]
  assert action._has_staged_control.tolist() == [True, False]
  assert not action.controller_failed.any()
  action._manager.failure = []
  action.apply_actions()
  assert action._manager.inputs[-1][:, bridge.layout.input.reset_offset()].tolist() == [
    0,
    1,
  ]
  action.apply_actions()
  action._collect_controller_output()
  assert not action._pending_reset.any()
  assert action.controller_worker_failed[1]
  assert action._has_staged_control.all()
  action._manager.failure = [0, 1]
  action.apply_actions()
  action.apply_actions()
  action._collect_controller_output()
  assert not action._has_staged_control.any()


def main() -> None:
  """Run contracts that do not require native recovery or adapter callbacks."""
  verify_layout()
  verify_walking_reference_feed()
  verify_pipeline()
  print("Native action deterministic contracts: PASS")


if __name__ == "__main__":
  main()
