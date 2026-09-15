"""Numeric datastore getter and setter columns of the shared-memory blocks."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import torch

import mc_rtc_interface as native

DatastoreKind = Literal["vector3", "scalar"]

# Getters the mc_mjlab adapter registers on every controller build; a task names
# them directly, there is no alias layer. docs/coupling.md
PLANNED_ZMP = "mc_mjlab::planned_zmp"
CONTROL_COM = "mc_mjlab::control_com"
CONTROL_COM_VEL = "mc_mjlab::control_com_vel"
SUPPORT_FOOT = "mc_mjlab::support_foot"


def input_columns(
  layout: native.IoLayout, names: Iterable[str], kind: DatastoreKind
) -> dict[str, int]:
  """Resolve configured setter names to their native input columns, once."""
  callbacks = list(getattr(layout.input, "datastore_" + kind))
  off = getattr(layout.input, "datastore_" + kind + "_offset")()
  width = 3 if kind == "vector3" else 1
  return {name: off + width * callbacks.index(name) for name in names}


def write_inputs(
  rows: np.ndarray, columns: dict[str, int], values: torch.Tensor, kind: DatastoreKind
) -> None:
  """Copy one tensor of per-environment setter values into the input block."""
  if not columns:
    return
  width = 3 if kind == "vector3" else 1
  payload = values.detach().cpu().numpy().reshape(rows.shape[0], -1, width)
  for index, start in enumerate(columns.values()):
    rows[:, start : start + width] = payload[:, index]


def output_columns(
  layout: native.IoLayout, names: Iterable[str], kind: DatastoreKind
) -> dict[str, int]:
  """Resolve configured getter names to their native output columns, once."""
  callbacks = list(getattr(layout.output, "datastore_" + kind))
  off = getattr(layout.output, "datastore_" + kind + "_offset")()
  width = 3 if kind == "vector3" else 1
  return {name: off + width * callbacks.index(name) for name in names}


def read_outputs(
  block: torch.Tensor, columns: dict[str, int], kind: DatastoreKind
) -> dict[str, torch.Tensor]:
  """Slice the uploaded output block into one tensor per configured getter."""
  width = 3 if kind == "vector3" else 1
  dtype = torch.get_default_dtype()
  return {
    name: (block[:, start : start + width] if width == 3 else block[:, start]).to(dtype)
    for name, start in columns.items()
  }
