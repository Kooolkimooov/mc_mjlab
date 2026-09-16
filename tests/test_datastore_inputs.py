"""Python datastore validation and feed-buffer contracts, without native transport."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv

from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionAction,
)
from mc_mjlab.bridge.controller_datastore import write_inputs


def test_setter_validation_readback_and_python_scatter() -> None:
  """Setters isolate configured columns, copy inputs and reject invalid requests."""
  action = object.__new__(McRtcResidualJointPositionAction)
  action._env = cast(ManagerBasedRlEnv, SimpleNamespace(num_envs=2))
  action._datastore_scalar_input_columns = {"second": 8, "first": 4}
  action._datastore_vector_input_columns = {"vector": 12}
  action._datastore_scalar_input_feed = torch.zeros(2, 2)
  action._datastore_vector_input_feed = torch.zeros(2, 1, 3)
  values = torch.tensor([2.0, 3.0])
  action.set_datastore_scalar_input("first", values)
  values.zero_()
  assert action.datastore_scalar_input("first").tolist() == [2.0, 3.0]
  assert action.datastore_scalar_input("second").tolist() == [0.0, 0.0]
  vectors = torch.arange(6.0).reshape(2, 3)
  action.set_datastore_vector_input("vector", vectors)
  rows = np.full((2, 18), -1.0)
  write_inputs(
    rows,
    action._datastore_scalar_input_columns,
    action._datastore_scalar_input_feed,
    "scalar",
  )
  write_inputs(
    rows,
    action._datastore_vector_input_columns,
    action._datastore_vector_input_feed,
    "vector3",
  )
  np.testing.assert_array_equal(rows[:, 4], [2, 3])
  np.testing.assert_array_equal(rows[:, 8], [0, 0])
  np.testing.assert_array_equal(rows[:, 12:15], vectors.numpy())
  assert (np.delete(rows, [4, 8, 12, 13, 14], axis=1) == -1).all()
  for setter, bad in (
    (action.set_datastore_scalar_input, torch.zeros(2, 1)),
    (action.set_datastore_vector_input, torch.zeros(2, 2)),
  ):
    name = "first" if bad.shape[-1] == 1 else "vector"
    with pytest.raises(ValueError, match="shape"):
      setter(name, bad)
  for getter in (action.datastore_scalar_input, action.datastore_vector_input):
    with pytest.raises(KeyError, match="not configured"):
      getter("missing")
  with pytest.raises(KeyError, match="not configured"):
    action.set_datastore_scalar_input("missing", torch.zeros(2))
  with pytest.raises(KeyError, match="not configured"):
    action.set_datastore_vector_input("missing", torch.zeros(2, 3))
