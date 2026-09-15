"""Object layout, MuJoCo conventions and native measured/reference separation."""

import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from test_manager_bindings import make_configuration, make_layout

import mc_rtc_interface as native_typed
from mc_mjlab.controller_objects import object_state
from utils.shared_memory import create_shm, row_window

native = cast(Any, native_typed)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_object_layout_preserves_earlier_columns(count):
  layout = make_layout()
  start = layout.input.datastore_scalar_offset()
  root = layout.input.root_offset()
  layout.input.objects = [f"object_{i}" for i in range(count)]
  layout.input.datastore_scalar = ["command"]
  layout.input.datastore_vector3 = ["velocity"]
  assert layout.input.objects_offset() == start
  assert layout.input.root_offset() == root
  assert layout.input.datastore_scalar_offset() == start + 13 * count
  assert layout.input.reset_offset() == start + 13 * count + 4


def test_object_pose_and_velocity_conventions():
  half = math.sqrt(0.5)
  qpos = torch.tensor([[10.75, 3, 0, half, 0, 0, half], [0.75, 0, 0, 1, 0, 0, 0]])
  qvel = torch.tensor([[1.0, 2, 3, 1, 0, 0], [1.0, 2, 3, 1, 0, 0]])
  origins = torch.tensor([[10.0, 3, 0], [0.0, 0, 0]])
  actual = object_state(qpos, qvel, origins)
  torch.testing.assert_close(actual[:, :3], torch.tensor([[0.75, 0, 0], [0.75, 0, 0]]))
  torch.testing.assert_close(actual[0, 3:7], torch.tensor([0.0, 0, half, half]))
  torch.testing.assert_close(actual[:, 7:10], qvel[:, :3])
  torch.testing.assert_close(
    actual[:, 10:13], torch.tensor([[0.0, 1, 0], [1.0, 0, 0]]), atol=2e-7, rtol=1e-6
  )


@pytest.mark.parametrize("recover_worker", [False, True])
def test_object_feedback_reset_and_worker_recovery(tmp_path, recover_worker):
  layout = make_layout()
  layout.input.objects = ["obj"]
  layout.input.datastore_scalar = ["set_hang"]
  getters = [
    "object_measured",
    "object_reference",
    "object_velocity",
    "object_angular",
    "object_axis",
    "object_reset_reference",
    "object_reset_measured",
  ]
  layout.output.datastore_vector3 = getters
  path = Path(make_configuration(tmp_path))
  path.write_text(path.read_text() + "ProbeObject: true\n")
  inputs, outputs = (
    create_shm((2, layout.input_size)),
    create_shm((2, layout.output_size)),
  )
  try:
    inputs.arr[:, layout.input.root_offset() + 2] = 0.8
    inputs.arr[:, layout.input.root_offset() + 6] = 1.0
    off = layout.input.objects_offset()
    inputs.arr[:, off : off + 13] = [
      0.75,
      0.2,
      0.1,
      0,
      0,
      math.sqrt(0.5),
      math.sqrt(0.5),
      1,
      2,
      3,
      4,
      5,
      6,
    ]
    message = native.WorkerStartMessage(
      layout,
      native.SharedMemoryDescription(*row_window(inputs, 0, 2)),
      native.SharedMemoryDescription(*row_window(outputs, 0, 2)),
    )

    def get(name):
      start = layout.output.datastore_vector3_offset() + 3 * getters.index(name)
      return outputs.arr[:, start : start + 3].copy()

    with native.ControllersManager(
      str(path), 2, 1, message, timeout_ms=5000
    ) as manager:
      manager.dispatch(native.Command.Initialize)
      assert manager.collect() == []
      for name in (
        "object_measured",
        "object_reference",
        "object_reset_reference",
        "object_reset_measured",
      ):
        np.testing.assert_allclose(get(name), inputs.arr[:, off : off + 3])
      inputs.arr[0, off] = 1.25
      manager.dispatch(native.Command.Step)
      assert manager.collect() == []
      np.testing.assert_allclose(get("object_measured"), inputs.arr[:, off : off + 3])
      np.testing.assert_allclose(get("object_reference")[:, 0], [0.75, 0.75])
      np.testing.assert_allclose(get("object_velocity"), [[1, 2, 3]] * 2)
      np.testing.assert_allclose(get("object_angular"), [[4, 5, 6]] * 2)
      np.testing.assert_allclose(get("object_axis"), [[0, 1, 0]] * 2, atol=1e-12)
      if recover_worker:
        inputs.arr[0, layout.input.datastore_scalar_offset()] = 1
        manager.dispatch(native.Command.Step)
        assert manager.collect() == [0, 1]
        manager.respawn([0, 1])
        inputs.arr[:, layout.input.datastore_scalar_offset()] = 0
        inputs.arr[:, layout.input.reset_offset()] = 1
      else:
        inputs.arr[0, layout.input.reset_offset()] = 1
      manager.dispatch(native.Command.Step)
      assert manager.collect() == []
      np.testing.assert_allclose(get("object_reference"), inputs.arr[:, off : off + 3])
      np.testing.assert_allclose(
        get("object_reset_measured"), inputs.arr[:, off : off + 3]
      )
  finally:
    inputs.unlink()
    outputs.unlink()
