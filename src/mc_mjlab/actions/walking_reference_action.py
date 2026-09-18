"""Policy control of the ismpc walking reference, beside the joint residual."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mc_mjlab.actions.mc_rtc_residual_action import (
  McRtcResidualActionBase,
  McRtcResidualActionCfg,
)

#: The walking controller's own reference pair, not the mc_mjlab adapter's.
WALKING_REF_VEL_GETTER = "ismpc_walking::get_ref_vel"
WALKING_REF_VEL_SETTER = "ismpc_walking::set_ref_vel"


@dataclass(kw_only=True)
class WalkingReferenceActionCfg(McRtcResidualActionCfg):
  """Shared configuration for actions that drive the walking reference."""

  required_controller: str = "LogisticController_ismpc"
  """Only this controller provides the ismpc walking-reference calls."""

  def __post_init__(self) -> None:
    """Declare the reference callbacks; `_validate_cfg` owns the controller check."""
    super().__post_init__()
    self.datastore_vectors_inputs = tuple(
      dict.fromkeys((*self.datastore_vectors_inputs, WALKING_REF_VEL_SETTER))
    )
    self.datastore_vectors_outputs = tuple(
      dict.fromkeys((*self.datastore_vectors_outputs, WALKING_REF_VEL_GETTER))
    )


class WalkingReferenceActionBase(McRtcResidualActionBase):
  """Buffers and absolute reference feed for walking actions."""

  cfg: WalkingReferenceActionCfg

  def _setup_action_extensions(self, cfg: McRtcResidualActionCfg) -> None:
    super()._setup_action_extensions(cfg)
    self._walking_reference_executed = torch.zeros(self.num_envs, 3, device=self.device)

  def _advance_action_extensions(self) -> None:
    super()._advance_action_extensions()
    self._feed_walking_reference()

  def _feed_walking_reference(self) -> None:
    """Send the executed absolute walking target."""
    self.set_datastore_vector_input(
      WALKING_REF_VEL_SETTER, self._walking_reference_executed
    )

  def _reset_action_extensions(self, env_ids: torch.Tensor | slice) -> None:
    super()._reset_action_extensions(env_ids)
    self._walking_reference_executed[env_ids] = 0.0

  @property
  def walking_reference_velocity(self) -> torch.Tensor:
    """Executed ``(vx, vy, yaw_rate)`` target in physical units."""
    return self._walking_reference_executed


@dataclass(kw_only=True)
class AbsoluteWalkingReferenceActionCfg(WalkingReferenceActionCfg):
  """Configuration for driving the walking reference from a command-manager term."""

  walking_velocity_command_name: str = "base_velocity"
  """Command-manager term sent as an absolute ISMPC ``set_ref_vel`` target."""


class AbsoluteWalkingReferenceActionBase(WalkingReferenceActionBase):
  """Send a command-manager term as the controller's walking-velocity target."""

  cfg: AbsoluteWalkingReferenceActionCfg

  def _setup_action_extensions(self, cfg: McRtcResidualActionCfg) -> None:
    super()._setup_action_extensions(cfg)
    name = self.cfg.walking_velocity_command_name
    command = self._env.command_manager.get_command(name)
    if command is None or tuple(command.shape) != (self.num_envs, 3):
      raise ValueError(f"walking command {name!r} must have shape ({self.num_envs}, 3)")

  def _process_action_extensions(self, actions: torch.Tensor) -> None:
    super()._process_action_extensions(actions)
    command = self._env.command_manager.get_command(
      self.cfg.walking_velocity_command_name
    )
    assert command is not None
    self._walking_reference_executed.copy_(command)
