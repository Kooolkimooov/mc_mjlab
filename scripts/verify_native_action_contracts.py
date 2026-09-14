"""Verify native action wiring and scheduling without external controller adapters."""

from types import SimpleNamespace as NS
from unittest.mock import patch

import mujoco
import numpy as np
import torch

import mc_mjlab.tasks  # noqa: F401
import mc_rtc_interface as native
from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionAction,
)
from mc_mjlab.controller_datastore import (
  DatastoreCommands,
  output_columns,
  read_outputs,
)
from mc_mjlab.controller_io import ControllerIoBinding


class UsageInput:
  """Test the forthcoming per-callback flags without modifying native bindings."""

  def __init__(self):
    self.datastore_scalar = []
    self.datastore_vector3 = []

  def datastore_scalar_offset(self):
    return 10

  def datastore_vector3_offset(self):
    return 20

  def use_datastore_scalar_offset(self):
    return 40

  def use_datastore_vector3_offset(self):
    return 50


def verify_layout():
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

  def sensor(name):
    return NS(name=lambda: name)

  module = NS(
    bodySensors=lambda: [sensor("FloatingBase"), sensor("IMU")],
    forceSensors=lambda: [sensor("foot")],
  )
  with (
    patch(
      "mc_mjlab.controller_io.robots.get_ref_joint_order",
      return_value=("a", "missing", "b"),
    ),
    patch(
      "mc_mjlab.controller_io.robots.get_default_joint_positions",
      return_value={"missing": 0.7},
    ),
    patch("mc_mjlab.controller_io.robots.get_robot_module", return_value=module),
  ):
    io = ControllerIoBinding(
      env,
      entity,
      ["b", "a"],
      torch.tensor([0, 1]),
      "test",
      ("q", "alpha", "tau"),
      "robot",
    )
  rows = np.zeros((2, io.layout.input_size))
  io.set_feedback_offset(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
  io.set_joint_velocity_offset(torch.tensor([[0.2, 0.3], [0.4, 0.5]]))
  io.set_wrench_offset(torch.ones(2, 6))
  io.fill_controller_input(rows)
  layout = io.layout.input
  np.testing.assert_allclose(rows[:, :3], [[5.2, 0.7, 4.1], [7.4, 0.7, 6.3]])
  np.testing.assert_allclose(rows[:, 3:6], [[9.3, 0, 8.2], [11.5, 0, 10.4]])
  np.testing.assert_allclose(rows[:, 6:9], [[39, 0, 38], [47, 0, 46]])
  ro = layout.root_offset()
  np.testing.assert_allclose(rows[:, ro : ro + 3], [[1, 2, 3]] * 2)
  data.qpos[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])
  io.fill_controller_input(rows)
  np.testing.assert_allclose(rows[:, ro + 3 : ro + 7], [[0, 0, 0, 1]] * 2)
  bo = layout.body_sensors_offset()
  np.testing.assert_allclose(rows[:, bo : bo + 3], data.qvel[:, 3:6])
  np.testing.assert_allclose(rows[:, bo + 3 : bo + 6], data.qacc[:, :3])
  np.testing.assert_allclose(rows[:, bo + 6 : bo + 12], data.sensordata[:, :6])
  fo = layout.force_sensors_offset()
  np.testing.assert_allclose(rows[:, fo : fo + 6], data.sensordata[:, 6:12] + 1)
  io.set_root_pose_offset(torch.ones(2, 3), torch.tensor([[0.0, 0.1, 0.0]] * 2))
  io.fill_controller_input(rows)
  np.testing.assert_allclose(rows[:, ro : ro + 3], [[2, 3, 4]] * 2)
  assert not np.allclose(rows[:, bo + 9 : bo + 12], data.sensordata[:, 3:6])
  np.testing.assert_allclose(np.linalg.norm(rows[:, ro + 3 : ro + 7], axis=1), 1)
  out = np.arange(2.0 * io.layout.output_size).reshape(2, -1)
  block = io.upload_controller_output(out)
  for channel, offset in (("q", 0), ("alpha", 3), ("tau", 6)):
    np.testing.assert_allclose(
      io.read_controller_output(block)[channel], out[:, [offset + 2, offset]]
    )
  return io, env, entity


def verify_datastore_commands():
  """Check independent gates, live getters, restoration and reset warmup."""
  layout = NS(input=UsageInput(), output=native.OutputLayout())
  scalar = DatastoreCommands(
    layout, 2, (("get_a", "set_a"), ("get_b", "set_b")), "scalar"
  )
  vector = DatastoreCommands(
    layout, 2, (("get_v", "set_v"), ("get_abs", "set_abs")), "vector3", (False, True)
  )
  rows = np.zeros((2, 60))
  out = np.zeros((2, layout.output.size()))
  off = layout.output.datastore_scalar_offset()
  out[:, off : off + 2] = [[3, 4], [5, 6]]
  vo = layout.output.datastore_vector3_offset()
  out[:, vo : vo + 6] = [[1, 2, 3, 4, 5, 6]] * 2
  active = torch.tensor([[True, False], [False, True]])
  values = torch.ones(2, 2)
  scalar.write(rows, active, values)
  assert not rows[:, 40:42].any()
  scalar.collect(out, [0, 1])
  scalar.write(rows, active, values)
  np.testing.assert_array_equal(rows[:, 40:42], active)
  np.testing.assert_allclose(rows[[0, 1], [10, 11]], [4, 7])
  out[:, off : off + 2] = [[4, 40], [50, 7]]
  scalar.collect(out, [0, 1])
  scalar.write(rows, ~active, values * 2)
  np.testing.assert_allclose(rows[:, 10:12], [[3, 42], [52, 6]])
  assert rows[:, 40:42].all()
  scalar.write(rows, ~active, values * 2)
  np.testing.assert_array_equal(rows[:, 40:42], ~active)
  columns = output_columns(layout, {"public": "get_a"}, "scalar")
  scalars = read_outputs(torch.from_numpy(out), columns, "scalar")
  np.testing.assert_allclose(scalars["public"], [4, 50])
  vector.write(rows, torch.ones(2, 2, dtype=torch.bool), torch.ones(2, 6) * 9)
  np.testing.assert_array_equal(rows[:, 50:52], [[0, 1]] * 2)
  np.testing.assert_allclose(rows[:, 23:26], 9)
  vector.collect(out, [0, 1])
  vector.write(rows, torch.ones(2, 2, dtype=torch.bool), torch.ones(2, 6))
  np.testing.assert_allclose(rows[:, 20:26], [[2, 3, 4, 1, 1, 1]] * 2)
  scalar.reset([0])
  scalar.write(rows, torch.ones(2, 2, dtype=torch.bool), values)
  assert not rows[0, 40:42].any()
  out[0, :2] = [100, 200]
  scalar.collect(out, [0])
  scalar.write(rows, torch.ones(2, 2, dtype=torch.bool), values)
  np.testing.assert_allclose(rows[0, 10:12], [101, 201])
  try:
    DatastoreCommands(native.IoLayout(), 1, (("get", "set"),), "scalar")
  except RuntimeError as error:
    assert "use_datastore_scalar_offset" in str(error)
  else:
    assert hasattr(native.InputLayout(), "use_datastore_scalar_offset")


class Manager:
  """Deterministic manager that distinguishes failed rows from completed rows."""

  def __init__(self, action):
    self.action = action
    self.calls = 0
    self.failure = []
    self.inputs = []
    self.resets = []

  def dispatch(self, command):
    assert command == native.Command.Step
    self.calls += 1
    self.inputs.append(self.action._in_np.copy())

  def collect(self):
    out = self.action._out_np
    out[:] = self.calls * 4
    out[:, self.action._io.layout.output.status_offset()] = 0
    return self.failure

  def respawn(self, reset_row_ids):
    self.resets.append(list(reset_row_ids))


def verify_pipeline():
  """Keep interpolation delayed through partial resets and exclude failed rows."""
  io, env, entity = verify_layout()
  action = object.__new__(McRtcResidualJointPositionAction)
  action._env = env
  action._entity = entity
  action.cfg = NS(frameskip=2, controller_vectors=(), controller_scalars=())
  action._num_targets = 2
  action._target_ids = torch.tensor([0, 1])
  action._io = io
  io._output_channels = action.output_channels
  action._vector_aliases = action._scalar_aliases = {}
  action._datastore_scalar_commands = ()
  action._vector_commands = DatastoreCommands(io.layout, 2, (), "vector3")
  action._scalar_commands = DatastoreCommands(io.layout, 2, (), "scalar")
  action._vector_columns = output_columns(io.layout, {}, "vector3")
  action._scalar_columns = output_columns(io.layout, {}, "scalar")
  action._alloc_interpolation_buffers()
  action._in_np = np.zeros((2, io.layout.input_size))
  action._out_np = np.zeros((2, io.layout.output_size))
  action._manager = Manager(action)
  action._pending_dispatch = False
  action._pending_reset = np.zeros(2, dtype=bool)
  action._dispatch_resets = np.zeros(2, dtype=bool)
  action._substep = 0
  action._walking_reference_active = torch.zeros(2, dtype=torch.bool)
  action._walking_reference_executed = torch.empty(2, 0)
  action._datastore_scalar_active = torch.empty(2, 0, dtype=torch.bool)
  action._datastore_scalar_delta = torch.empty(2, 0)
  action._processed_actions = torch.zeros(2, 2)
  action._last_gate = torch.ones(2)
  action._torque_peak = torch.zeros(2, 2)
  entity.data.qfrc_actuator = torch.zeros(2, 2)
  action._residual_ids = None
  action._executed_physical = torch.zeros(2, 2)
  action._projection_mask = torch.zeros(2, 2, dtype=torch.bool)
  action._print_pending = False
  action._raw_actions = torch.zeros(2, 2)
  action._previous_executed_physical = torch.zeros(2, 2)
  action._residual_raw_actions = torch.zeros(2, 2)
  action._previous_residual_raw_actions = torch.zeros(2, 2)
  action._walking_reference_requested = torch.empty(2, 0)
  action._previous_walking_reference_executed = torch.empty(2, 0)
  action._previous_gate = torch.ones(2)
  action._datastore_scalar_holds = ()
  action._datastore_scalar_hold_values = torch.empty(0)
  action._recovery_authority = None
  applied = []

  def apply(control, residual):
    applied.append(control["q"].clone())
    return residual, torch.zeros_like(residual, dtype=torch.bool)

  action._apply_control = apply
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
  assert action._manager.inputs[-1][:, io.layout.input.reset_offset()].tolist() == [
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


def main():
  """Run contracts that do not require native recovery or adapter callbacks."""
  verify_layout()
  verify_datastore_commands()
  verify_pipeline()
  print("Native action deterministic contracts: PASS")


if __name__ == "__main__":
  main()
