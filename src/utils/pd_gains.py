"""Reading and overwriting an mjlab entity's PD gains."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def apply_reference_pd_gains(
  entity,
  ref_joint_order: Sequence[str],
  target_names: Sequence[str],
  path: str,
) -> None:
  """Set the entity's PD gains from an mc_mujoco ``PDgains_sim.dat``."""
  with open(path) as f:
    rows = [line.split() for line in f if line.strip()]
  gains = {
    name: (float(row[0]), float(row[1]))
    for name, row in zip(ref_joint_order, rows, strict=True)
    if len(row) >= 2
  }

  matched = 0
  for act in entity.actuators:
    stiffness = getattr(act, "stiffness", None)
    damping = getattr(act, "damping", None)
    if stiffness is None or damping is None:
      continue
    for j, name in enumerate(act.target_names):
      if name in gains:
        kp, kd = gains[name]
        stiffness[:, j] = kp
        damping[:, j] = kd
        matched += 1
        print(
          f"[mc_rtc] applied reference PD gains {kp=} and {kd=} from {path.split('/')[-1]} to {name}"
        )


def _actuator_gain_columns(
  entity, target_names: Sequence[str]
) -> list[tuple[int, torch.Tensor, torch.Tensor, int]]:
  """``(target index, stiffness, damping, column)`` per PD-driven target joint."""
  index_of = {name: i for i, name in enumerate(target_names)}
  columns: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []
  for act in entity.actuators:
    stiffness = getattr(act, "stiffness", None)
    damping = getattr(act, "damping", None)
    if stiffness is None or damping is None:
      continue
    for j, name in enumerate(act.target_names):
      i = index_of.get(name)
      if i is not None:
        columns.append((i, stiffness, damping, j))
  return columns


def read_pd_gains(
  entity, target_names: Sequence[str], num_envs: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
  """``(kp, kd)`` copies, each ``(num_envs, num_targets)`` in target order."""
  kp = torch.zeros(num_envs, len(target_names), device=device)
  kd = torch.zeros_like(kp)
  for i, stiffness, damping, j in _actuator_gain_columns(entity, target_names):
    kp[:, i] = stiffness[:, j]
    kd[:, i] = damping[:, j]
  return kp, kd


def zero_pd_gains(entity, target_names: Sequence[str]) -> int:
  """Zero the entity's PD gains for the target joints; returns the count."""
  columns = _actuator_gain_columns(entity, target_names)
  for _, stiffness, damping, j in columns:
    stiffness[:, j] = 0.0
    damping[:, j] = 0.0
  return len(columns)
