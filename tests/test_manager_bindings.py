"""Run native manager commands against Python-owned shared-memory rows."""

import os
from collections.abc import Iterator
from pathlib import Path

import mc_rbdyn
import numpy as np
import pytest

import mc_rtc_interface as native
from mc_mjlab.bridge.shared_memory import ShmHandle, create_shm, row_window

SharedIo = tuple[native.WorkerStartMessage, ShmHandle, ShmHandle]

DIRECT_JOINTS = 17

# Not native.__file__'s dir: an editable install resolves that to the install
# tree, which carries only the module and the worker, never the probe.
PROBE_DIR = Path(
  os.environ.get(
    "MC_RTC_INSTANCE_PROBE_DIR",
    Path(__file__).resolve().parents[1] / "build",
  )
)


def make_layout() -> native.IoLayout:
  """Prepare the worker layout from the robot module's complete reference order."""
  robot = mc_rbdyn.get_robot_module("HRP5P")
  layout = native.IoLayout()
  layout.set_joint_order(robot.ref_joint_order())
  layout.input.body_sensors = [sensor.name().decode() for sensor in robot.bodySensors()]
  layout.input.force_sensors = [
    sensor.name().decode() for sensor in robot.forceSensors()
  ]
  return layout


def make_configuration(tmp_path: Path) -> str:
  """Write the probe controller configuration with isolated plugin loading."""
  config = tmp_path / "mc_rtc.yaml"
  config.write_text(
    "MainRobot: HRP5P\nEnabled: InstanceProbe\nTimestep: 0.002\n"
    "GUIServer: {Enable: false}\nLog: false\n"
    "ClearGlobalPluginPath: true\nGlobalPluginPaths: []\nPlugins: []\n"
    f"ControllerModulePaths: ['{PROBE_DIR}']\n"
  )
  return str(config)


@pytest.fixture
def shared_io() -> Iterator[SharedIo]:
  """Allocate a batch whose uneven worker split exercises row offsets."""
  layout = make_layout()
  layout.input.datastore_scalar = ["set_scalar", "set_hang"]
  layout.output.datastore_scalar = ["get_scalar"]
  inputs = create_shm((3, layout.input_size))
  outputs = create_shm((3, layout.output_size))
  root = layout.input.root_offset()
  inputs.arr[:, root + 2] = 0.8
  inputs.arr[:, root + 6] = 1.0
  configuration = native.WorkerStartMessage(
    layout,
    native.SharedMemoryDescription(*row_window(inputs, 0, 3)),
    native.SharedMemoryDescription(*row_window(outputs, 0, 3)),
  )
  try:
    yield configuration, inputs, outputs
  finally:
    inputs.unlink()
    outputs.unlink()


def test_manager_steps_python_owned_rows(tmp_path: Path, shared_io: SharedIo) -> None:
  configuration, inputs, outputs = shared_io
  layout = configuration.layout
  inputs.arr[:, :DIRECT_JOINTS] = np.arange(3)[:, None] * 0.05
  inputs.arr[:, layout.input.datastore_scalar_offset()] = [1.0, 2.0, 3.0]
  with native.ControllersManager(
    make_configuration(tmp_path),
    3,
    2,
    configuration,
    timeout_ms=5000,
  ) as manager:
    manager.dispatch(native.Command.Initialize)
    assert manager.collect() == []
    manager.dispatch(native.Command.Step)
    assert manager.collect() == []
    np.testing.assert_allclose(
      outputs.arr[:, :DIRECT_JOINTS], inputs.arr[:, :DIRECT_JOINTS]
    )
    np.testing.assert_array_equal(
      outputs.arr[:, layout.output.datastore_scalar_offset()], [1.0, 2.0, 3.0]
    )
    inputs.arr[1, layout.input.reset_offset()] = 1.0
    manager.dispatch(native.Command.Step)
    assert manager.collect() == []
    assert (outputs.arr[:, layout.output.status_offset()] == 0).all()
    inputs.arr[:, layout.input.reset_offset()] = 0.0
    manager.dispatch(native.Command.Reset)
    assert manager.collect() == []
  manager.close()
  with pytest.raises(RuntimeError, match="closed"):
    manager.collect()
  outputs.arr[:] = 42.0
  assert (outputs.arr == 42.0).all()


def test_uninitialized_step_only_writes_row_status(
  tmp_path: Path, shared_io: SharedIo
) -> None:
  configuration, inputs, outputs = shared_io
  status = configuration.layout.output.status_offset()
  outputs.arr[:] = 42.0
  original_inputs = inputs.arr.copy()
  with native.ControllersManager(
    make_configuration(tmp_path), 3, 2, configuration, timeout_ms=5000
  ) as manager:
    manager.dispatch(native.Command.Step)
    assert manager.collect() == []
    np.testing.assert_array_equal(inputs.arr, original_inputs)
    assert (np.delete(outputs.arr, status, axis=1) == 42.0).all()
    assert (
      outputs.arr[:, status] == int(native.OutputLayout.Status.WORKER_FAILED)
    ).all()


def test_manager_respawns_wedged_worker(tmp_path: Path, shared_io: SharedIo) -> None:
  configuration, inputs, outputs = shared_io
  layout = configuration.layout
  scalar = layout.input.datastore_scalar_offset()
  hang = scalar + 1
  status = layout.output.status_offset()
  inputs.arr[:, scalar] = [1.0, 2.0, 3.0]
  with native.ControllersManager(
    make_configuration(tmp_path), 3, 2, configuration, timeout_ms=5000
  ) as manager:
    manager.dispatch(native.Command.Initialize)
    assert manager.collect() == []
    outputs.arr[:2] = 42.0
    outputs.arr[:, status] = int(native.OutputLayout.Status.WORKER_FAILED)
    inputs.arr[0, hang] = 1.0
    manager.dispatch(native.Command.Step)
    assert manager.collect() == [0, 1]
    assert (np.delete(outputs.arr[:2], status, axis=1) == 42.0).all()
    assert outputs.arr[2, status] == int(native.OutputLayout.Status.OK)
    assert outputs.arr[2, layout.output.datastore_scalar_offset()] == 3.0

    inputs.arr[0, hang] = 0.0
    inputs.arr[:2, layout.input.reset_offset()] = 1.0
    manager.respawn([0, 1])
    manager.dispatch(native.Command.Step)
    assert manager.collect() == []
    assert (outputs.arr[:, status] == int(native.OutputLayout.Status.OK)).all()
    np.testing.assert_array_equal(
      outputs.arr[:, layout.output.datastore_scalar_offset()], [1.0, 2.0, 3.0]
    )


def test_qp_failure_latches_until_row_reset(
  tmp_path: Path, shared_io: SharedIo
) -> None:
  configuration, inputs, outputs = shared_io
  layout = configuration.layout
  scalar = layout.input.datastore_scalar_offset()
  status = layout.output.status_offset()
  with native.ControllersManager(
    make_configuration(tmp_path), 3, 2, configuration, timeout_ms=5000
  ) as manager:
    manager.dispatch(native.Command.Initialize)
    assert manager.collect() == []
    inputs.arr[0, scalar] = -1.0
    manager.dispatch(native.Command.Step)
    assert manager.collect() == []
    np.testing.assert_array_equal(outputs.arr[:, status], [1, 0, 0])
    inputs.arr[0, scalar] = 3.0
    manager.dispatch(native.Command.Step)
    assert manager.collect() == []
    np.testing.assert_array_equal(outputs.arr[:, status], [1, 0, 0])
    inputs.arr[0, layout.input.reset_offset()] = 1.0
    manager.dispatch(native.Command.Step)
    assert manager.collect() == []
    np.testing.assert_array_equal(outputs.arr[:, status], [0, 0, 0])
    assert outputs.arr[0, layout.output.datastore_scalar_offset()] == 3.0


def test_context_manager_closes_on_python_exception(
  tmp_path: Path, shared_io: SharedIo
) -> None:
  configuration, _, _ = shared_io
  with pytest.raises(ValueError, match="probe"):
    with native.ControllersManager(
      make_configuration(tmp_path), 3, 2, configuration
    ) as manager:
      raise ValueError("probe")
  with pytest.raises(RuntimeError, match="closed"):
    manager.dispatch(native.Command.Step)
