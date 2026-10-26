"""Position residual whose few policy outputs each drive a named joint group."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.string import resolve_matching_names

from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionAction,
  McRtcResidualJointPositionActionCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class GroupedResidualJointPositionActionCfg(McRtcResidualJointPositionActionCfg):
  """One normalized output per group, copied to every residual joint it names."""

  residual_groups: tuple[tuple[str, ...], ...] = ()
  """Actuator regexes per output; together they must cover each residual joint once."""

  def build(self, env: ManagerBasedRlEnv) -> GroupedResidualJointPositionAction:
    return GroupedResidualJointPositionAction(self, env)


class GroupedResidualJointPositionAction(McRtcResidualJointPositionAction):
  """Expand group outputs to joint requests before the shared scale and projection."""

  cfg: GroupedResidualJointPositionActionCfg

  def __init__(
    self, cfg: GroupedResidualJointPositionActionCfg, env: ManagerBasedRlEnv
  ) -> None:
    # mjlab's base sizes joint buffers from action_dim; stay joint-level until grouped.
    self._group_index: torch.Tensor | None = None
    self._num_groups = len(cfg.residual_groups)
    super().__init__(cfg, env)
    owners = group_of_joint(self.residual_names, cfg.residual_groups)
    self._group_index = torch.tensor(owners, device=self.device, dtype=torch.long)
    self._policy_actions = torch.zeros(
      self.num_envs, self._num_groups, device=self.device
    )

  def process_actions(self, actions: torch.Tensor) -> None:
    """Copy each group's output onto its joints, then process joint requests."""
    assert self._group_index is not None
    groups = actions[:, : self._num_groups]
    self._policy_actions.copy_(groups)
    expanded = torch.cat(
      (groups[:, self._group_index], actions[:, self._num_groups :]), 1
    )
    super().process_actions(expanded)

  def expand(self, groups: torch.Tensor) -> torch.Tensor:
    """Joint-level normalized requests for group-level outputs."""
    assert self._group_index is not None
    return groups[:, self._group_index]

  @property
  def action_dim(self) -> int:
    if self._group_index is None:
      return self._action_dim
    return self._num_groups + self._action_dim - self._residual_action_dim

  @property
  def policy_action(self) -> torch.Tensor:
    """The group-level outputs as the policy emitted them."""
    return self._policy_actions

  @property
  def group_index(self) -> torch.Tensor:
    """Policy output feeding each residual joint, in residual column order."""
    assert self._group_index is not None
    return self._group_index


def group_of_joint(
  residual_names: tuple[str, ...], groups: tuple[tuple[str, ...], ...]
) -> list[int]:
  """Each residual joint's group, rejecting gaps and overlaps."""
  if not groups:
    raise ValueError("grouped residual needs at least one group")
  owners: dict[int, int] = {}
  for group, patterns in enumerate(groups):
    indices, _ = resolve_matching_names(patterns, list(residual_names))
    for index in indices:
      if index in owners:
        raise ValueError(f"{residual_names[index]} belongs to two residual groups")
      owners[index] = group
  missing = [name for i, name in enumerate(residual_names) if i not in owners]
  if missing:
    raise ValueError(f"residual joints outside every group: {missing}")
  return [owners[i] for i in range(len(residual_names))]
