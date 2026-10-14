"""Joint residuals with explicit controller-side slowdown and hold authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionAction,
  McRtcResidualJointPositionActionCfg,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionCfg

MOTION_SCALE_SETTER = "Locomanip::setMotionScale"
MOTION_SCALE_GETTER = "Locomanip::motionScale"
MOTION_STATE_GETTER = "Locomanip::motionState"
MOTION_LIMIT_GETTER = "Locomanip::motionScaleLimit"


@dataclass(kw_only=True)
class LocomanipMotionActionCfg(McRtcResidualJointPositionActionCfg):
  """Append a normalized slowdown request to the joint-position residual."""

  motion_phase_getters: tuple[str, str]
  motion_hold_phase: float
  hold_action_threshold: float = 0.95
  minimum_motion_scale: float = 0.1

  def __post_init__(self) -> None:
    super().__post_init__()
    self.datastore_scalar_inputs = tuple(
      dict.fromkeys((*self.datastore_scalar_inputs, MOTION_SCALE_SETTER))
    )
    self.datastore_scalar_outputs = tuple(
      dict.fromkeys(
        (
          *self.datastore_scalar_outputs,
          *self.motion_phase_getters,
          MOTION_SCALE_GETTER,
          MOTION_STATE_GETTER,
          MOTION_LIMIT_GETTER,
        )
      )
    )
    if not 0 < self.hold_action_threshold <= 1:
      raise ValueError("hold_action_threshold must be in (0, 1]")
    if not 0.1 <= self.minimum_motion_scale <= 1:
      raise ValueError("minimum_motion_scale must be in [0.1, 1]")

  def build(self, env: ManagerBasedRlEnv) -> LocomanipMotionAction:
    return LocomanipMotionAction(self, env)


class LocomanipMotionAction(McRtcResidualJointPositionAction):
  """Feed a gated motion-scale request beside the bounded joint residual."""

  cfg: LocomanipMotionActionCfg

  def process_actions(self, actions: torch.Tensor) -> None:
    """Keep zero action at nominal speed and map positive requests to slowdown."""
    self._motion_raw.copy_(actions[:, -1].clamp(-1.0, 1.0))
    slowdown = self._motion_raw.clamp_min(0.0)
    self._motion_requested.copy_(
      torch.where(
        slowdown >= self.cfg.hold_action_threshold,
        0.0,
        (1.0 - slowdown).clamp_min(self.cfg.minimum_motion_scale),
      )
    )
    super().process_actions(actions[:, : self._residual_action_dim])

  def _setup_action_extensions(self, cfg: McRtcResidualActionCfg) -> None:
    """Allocate one extra policy channel without changing joint penalty units."""
    super()._setup_action_extensions(cfg)
    self._action_dim += 1
    self._motion_raw = torch.zeros(self.num_envs, device=self.device)
    self._motion_requested = torch.ones_like(self._motion_raw)
    self._motion_dispatched = torch.ones_like(self._motion_raw)

  def _advance_action_extensions(self) -> None:
    """Dispatch authority only with fresh bilateral Hold and a healthy controller."""
    super()._advance_action_extensions()
    enabled = (
      self._datastore_output_fresh
      & ~self.controller_failed
      & ~self.controller_worker_failed
    )
    for getter in self.cfg.motion_phase_getters:
      enabled &= self.datastore_scalar_output(getter) == self.cfg.motion_hold_phase
    self._motion_dispatched.copy_(torch.where(enabled, self._motion_requested, 1.0))
    self.set_datastore_scalar_input(MOTION_SCALE_SETTER, self._motion_dispatched)

  def _reset_action_extensions(self, env_ids: torch.Tensor | slice) -> None:
    """Clear row-local requests and restore nominal speed before native reset."""
    super()._reset_action_extensions(env_ids)
    self._motion_raw[env_ids] = 0.0
    self._motion_requested[env_ids] = 1.0
    self._motion_dispatched[env_ids] = 1.0
    self.set_datastore_scalar_input(MOTION_SCALE_SETTER, self._motion_dispatched)

  @property
  def motion_scale_requested(self) -> torch.Tensor:
    """Motion scale requested before phase gating, with zero meaning hold."""
    return self._motion_requested

  @property
  def motion_scale_dispatched(self) -> torch.Tensor:
    """Scale sent to the controller on the latest native dispatch."""
    return self._motion_dispatched

  @property
  def raw_action(self) -> torch.Tensor:
    """Joint requests followed by the normalized slowdown channel."""
    return torch.cat((self._raw_actions, self._motion_raw.unsqueeze(-1)), dim=1)
