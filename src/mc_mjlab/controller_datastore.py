"""Numeric datastore getters and independently gated baseline-relative commands."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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


class DatastoreCommands:
  """Capture collected baselines and restore each setter once on deactivation."""

  def __init__(
    self,
    layout: native.IoLayout,
    count: int,
    pairs: Sequence[tuple[str, str]],
    kind: DatastoreKind,
    absolute: Sequence[bool] = (),
  ) -> None:
    self.pairs = tuple(pairs)
    self.width = 3 if kind == "vector3" else 1
    self.absolute = np.array(absolute or (False,) * len(pairs), dtype=bool)
    if len(self.absolute) != len(pairs):
      raise ValueError("one absolute-command flag is required per callback pair")
    setters = [setter for _, setter in pairs]
    if len(set(setters)) != len(setters):
      raise ValueError("datastore setters must be unique within each numeric type")
    fed = list(getattr(layout.input, "datastore_" + kind))
    clashing = [setter for setter in setters if setter in fed]
    if clashing:
      raise ValueError(
        f"datastore setters {clashing} are already declared as unconditional "
        f"{kind} inputs; a setter is written by one path only"
      )
    # Appended, not assigned: the unconditional inputs hold the columns before
    # these, and `write` addresses this block from the first appended setter.
    setattr(layout.input, "datastore_" + kind, [*fed, *setters])
    self.setter_offset = len(fed)
    getters = list(getattr(layout.output, "datastore_" + kind))
    getters = list(dict.fromkeys([*getters, *(getter for getter, _ in pairs)]))
    setattr(layout.output, "datastore_" + kind, getters)
    self.getter_indices = [getters.index(getter) for getter, _ in pairs]
    self.kind = kind
    self.layout = layout
    self.active = np.zeros((count, len(pairs)), dtype=bool)
    self.ready = np.zeros(count, dtype=bool)
    self.latest = np.zeros((count, len(pairs), self.width))
    self.baseline = np.zeros_like(self.latest)
    method = "use_datastore_" + kind + "_offset"
    if pairs and not callable(getattr(layout.input, method, None)):
      raise RuntimeError(
        f"native InputLayout.{method}() is required for independently gated "
        "datastore setters; install the native usage-flags dependency"
      )

  def reset(self, indices: list[int]) -> None:
    """Forget baselines when the native controller is rebuilt."""
    self.ready[indices] = False
    self.active[indices] = False
    self.baseline[indices] = 0.0

  def collect(self, rows: np.ndarray, indices: list[int]) -> None:
    """Accept fresh getter outputs even when no setter is active."""
    off = getattr(self.layout.output, "datastore_" + self.kind + "_offset")()
    for i, getter in enumerate(self.getter_indices):
      start = off + self.width * getter
      self.latest[indices, i] = rows[indices, start : start + self.width]
      inactive = np.asarray(indices)[~self.active[indices, i]]
      self.baseline[inactive, i] = self.latest[inactive, i]
    self.ready[indices] = True

  def write(self, rows: np.ndarray, active: torch.Tensor, values: torch.Tensor) -> None:
    """Send absolute values with a usage flag for each callback and environment."""
    count, commands = self.active.shape
    if not commands:
      return
    requested = active.detach().cpu().numpy().reshape(count, commands)
    values = values.detach().cpu().numpy().reshape(count, commands, self.width)
    enabled = requested & (self.ready[:, None] | self.absolute[None, :])
    activated = enabled & ~self.active
    self.baseline[activated] = self.latest[activated]
    restoring = ~enabled & self.active & ~self.absolute[None, :]
    payload = np.where(self.absolute[None, :, None], values, self.baseline + values)
    payload[restoring] = self.baseline[restoring]
    off = getattr(self.layout.input, "datastore_" + self.kind + "_offset")()
    off += self.width * self.setter_offset
    usage = getattr(self.layout.input, "use_datastore_" + self.kind + "_offset")()
    usage += self.setter_offset
    rows[:, off : off + commands * self.width] = payload.reshape(count, -1)
    rows[:, usage : usage + commands] = enabled | restoring
    self.active[:] = enabled


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
