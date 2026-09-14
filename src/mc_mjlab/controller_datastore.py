"""Numeric datastore aliases and independently gated baseline-relative commands."""

from __future__ import annotations

import numpy as np
import torch

VECTOR_CALLBACKS = {
  "planned_zmp": "mc_mjlab::planned_zmp",
  "control_com": "mc_mjlab::control_com",
  "control_com_vel": "mc_mjlab::control_com_vel",
  "walking_ref_vel": "ismpc_walking::get_ref_vel",
}
SCALAR_CALLBACKS = {
  "ismpc_walking::support_foot_name": "mc_mjlab::support_foot",
  "support_foot": "mc_mjlab::support_foot",
}


class DatastoreCommands:
  """Capture collected baselines and restore each setter once on deactivation."""

  def __init__(self, layout, count, pairs, kind, absolute=()):
    self.pairs = tuple(pairs)
    self.width = 3 if kind == "vector3" else 1
    self.absolute = np.array(absolute or (False,) * len(pairs), dtype=bool)
    if len(self.absolute) != len(pairs):
      raise ValueError("one absolute-command flag is required per callback pair")
    setters = [setter for _, setter in pairs]
    if len(set(setters)) != len(setters):
      raise ValueError("datastore setters must be unique within each numeric type")
    setattr(layout.input, "datastore_" + kind, setters)
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

  def reset(self, indices):
    """Forget baselines when the native controller is rebuilt."""
    self.ready[indices] = False
    self.active[indices] = False
    self.baseline[indices] = 0.0

  def collect(self, rows, indices):
    """Accept fresh getter outputs even when no setter is active."""
    off = getattr(self.layout.output, "datastore_" + self.kind + "_offset")()
    for i, getter in enumerate(self.getter_indices):
      start = off + self.width * getter
      self.latest[indices, i] = rows[indices, start : start + self.width]
      inactive = np.asarray(indices)[~self.active[indices, i]]
      self.baseline[inactive, i] = self.latest[inactive, i]
    self.ready[indices] = True

  def write(self, rows, active, values):
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
    usage = getattr(self.layout.input, "use_datastore_" + self.kind + "_offset")()
    rows[:, off : off + commands * self.width] = payload.reshape(count, -1)
    rows[:, usage : usage + commands] = enabled | restoring
    self.active[:] = enabled


def output_columns(layout, aliases, kind):
  """Resolve public aliases to supported native getter columns, once."""
  callbacks = list(getattr(layout.output, "datastore_" + kind))
  off = getattr(layout.output, "datastore_" + kind + "_offset")()
  width = 3 if kind == "vector3" else 1
  return {alias: off + width * callbacks.index(cb) for alias, cb in aliases.items()}


def read_outputs(block, columns, kind):
  """Slice the uploaded output block into public alias tensors."""
  width = 3 if kind == "vector3" else 1
  dtype = torch.get_default_dtype()
  return {
    alias: (block[:, start : start + width] if width == 3 else block[:, start]).to(
      dtype
    )
    for alias, start in columns.items()
  }
