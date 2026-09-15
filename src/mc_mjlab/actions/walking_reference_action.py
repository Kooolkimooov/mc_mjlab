"""Policy control of the ismpc walking reference, beside the joint residual."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mc_mjlab.actions.mc_rtc_residual_action import (
  McRtcResidualActionBase,
  McRtcResidualActionCfg,
)
from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionAction,
  McRtcResidualJointPositionActionCfg,
)
from utils.mc_rtc_config import get_controller_name

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

#: The walking controller's own reference pair, not the mc_mjlab adapter's.
WALKING_REF_VEL_GETTER = "ismpc_walking::get_ref_vel"
WALKING_REF_VEL_SETTER = "ismpc_walking::set_ref_vel"


@dataclass(kw_only=True)
class WalkingReferenceActionCfg(McRtcResidualActionCfg):
  """Shared configuration for actions that drive the walking reference."""

  walking_controller: str = "LogisticController_ismpc"
  """Enabled controller that must provide the ismpc walking-reference calls."""

  def __post_init__(self) -> None:
    """Declare the reference callbacks and reject a controller lacking them."""
    enabled = get_controller_name(Path(self.mc_rtc_config_path))
    if enabled != self.walking_controller:
      raise ValueError(
        f"{WALKING_REF_VEL_SETTER} needs {self.walking_controller!r}, but "
        f"{self.mc_rtc_config_path} enables {enabled!r}"
      )
    self.datastore_vectors_inputs = tuple(
      dict.fromkeys((*self.datastore_vectors_inputs, WALKING_REF_VEL_SETTER))
    )
    self.datastore_vectors_outputs = tuple(
      dict.fromkeys((*self.datastore_vectors_outputs, WALKING_REF_VEL_GETTER))
    )


class WalkingReferenceMixin(McRtcResidualActionBase):
  """Buffers, reference feed and readouts shared by both drive modes."""

  cfg: WalkingReferenceActionCfg

  walking_reference_is_absolute: bool = False
  """Whether the setter takes a target rather than an offset from the nominal."""

  def _setup_action_extensions(self, cfg: McRtcResidualActionCfg) -> None:
    super()._setup_action_extensions(cfg)
    self._walking_reference_requested = torch.zeros(
      self.num_envs, 3, device=self.device
    )
    self._walking_reference_executed = torch.zeros_like(
      self._walking_reference_requested
    )
    self._previous_walking_reference_executed = torch.zeros_like(
      self._walking_reference_requested
    )
    self._walking_reference_nominal = torch.zeros_like(
      self._walking_reference_requested
    )
    self._walking_reference_active = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def _feed_walking_reference(self) -> None:
    """Send the executed reference, as a target or an offset from the nominal."""
    if self.walking_reference_is_absolute:
      self.set_datastore_vector_input(
        WALKING_REF_VEL_SETTER, self._walking_reference_executed
      )
      return
    # The getter reports what this term last wrote, so the nominal is only
    # readable while the offset is zero. docs/walking-reference.md
    idle = (self._previous_walking_reference_executed == 0.0).all(dim=1)
    hold = (idle & self._datastore_output_fresh).unsqueeze(-1)
    self._walking_reference_nominal.copy_(
      torch.where(
        hold,
        self.datastore_vector_output(WALKING_REF_VEL_GETTER),
        self._walking_reference_nominal,
      )
    )
    self.set_datastore_vector_input(
      WALKING_REF_VEL_SETTER,
      self._walking_reference_nominal + self._walking_reference_executed,
    )

  def _reset_action_extensions(self, env_ids: torch.Tensor | slice) -> None:
    super()._reset_action_extensions(env_ids)
    self._walking_reference_requested[env_ids] = 0.0
    self._walking_reference_executed[env_ids] = 0.0
    self._previous_walking_reference_executed[env_ids] = 0.0
    self._walking_reference_active[env_ids] = False
    # The nominal survives: the rebuilt controller sets the same reference, and
    # feeding zero for the period before the first fresh getter stops the walk.

  @property
  def walking_reference_velocity(self) -> torch.Tensor:
    """Executed ``(vx, vy, yaw_rate)`` target or delta in physical units."""
    return self._walking_reference_executed

  @property
  def walking_reference_normalized(self) -> torch.Tensor:
    """Executed walking reference in normalized action coordinates."""
    return self._walking_reference_executed

  @property
  def previous_walking_reference_normalized(self) -> torch.Tensor:
    """Previous executed walking reference in normalized coordinates."""
    return self._previous_walking_reference_executed


@dataclass(kw_only=True)
class GatedWalkingReferenceDeltaActionCfg(
  WalkingReferenceActionCfg, McRtcResidualJointPositionActionCfg
):
  """Configuration for recovery-gated walking-velocity deltas on joint position."""

  walking_reference_velocity_scale: tuple[float, float, float] = (0.20, 0.15, 0.30)
  """Maximum recovery-gated walking velocity delta. docs/walking-reference.md"""

  walking_reference_velocity_slew_rate: tuple[float, float, float] = (2.0, 1.5, 3.0)
  """Maximum physical command change per second while authority is nonzero."""

  def __post_init__(self) -> None:
    super().__post_init__()
    scale = self.walking_reference_velocity_scale
    if len(scale) != 3 or any(value <= 0.0 for value in scale):
      raise ValueError(
        "walking reference velocity scales must be three positive values"
      )
    slew = self.walking_reference_velocity_slew_rate
    if len(slew) != 3 or any(value <= 0.0 for value in slew):
      raise ValueError("walking reference slew rates must be three positive values")

  def build(self, env: ManagerBasedRlEnv) -> "GatedWalkingReferenceDeltaAction":
    return GatedWalkingReferenceDeltaAction(self, env)


class GatedWalkingReferenceDeltaAction(
  WalkingReferenceMixin, McRtcResidualJointPositionAction
):
  """Three extra action dimensions offsetting the walk, scaled by recovery authority."""

  cfg: GatedWalkingReferenceDeltaActionCfg

  def _setup_action_extensions(self, cfg: McRtcResidualActionCfg) -> None:
    super()._setup_action_extensions(cfg)
    self._walking_reference_scale = torch.tensor(
      self.cfg.walking_reference_velocity_scale, device=self.device
    ).unsqueeze(0)
    self._walking_reference_slew = (
      torch.tensor(
        self.cfg.walking_reference_velocity_slew_rate, device=self.device
      ).unsqueeze(0)
      * self._env.step_dt
    )
    self._action_dim += 3
    self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)

  def _process_action_extensions(self, actions: torch.Tensor) -> None:
    super()._process_action_extensions(actions)
    self._previous_walking_reference_executed.copy_(self._walking_reference_executed)
    normalized = actions[:, self._residual_action_dim :].clamp(-1.0, 1.0)
    self._walking_reference_requested.copy_(normalized * self._walking_reference_scale)
    target = self._walking_reference_requested * self._last_gate.unsqueeze(-1)
    delta = (target - self._walking_reference_executed).clamp(
      -self._walking_reference_slew, self._walking_reference_slew
    )
    self._walking_reference_executed.add_(delta)
    # Zero authority must restore the nominal exactly, not ramp towards it.
    self._walking_reference_executed[self._last_gate == 0.0] = 0.0
    self._walking_reference_active.copy_(
      self._walking_reference_executed.abs().amax(dim=1) > 1.0e-6
    )
    self._feed_walking_reference()

  @property
  def walking_reference_normalized(self) -> torch.Tensor:
    """Executed walking-reference delta in normalized action coordinates."""
    return self._walking_reference_executed / self._walking_reference_scale

  @property
  def previous_walking_reference_normalized(self) -> torch.Tensor:
    """Previous executed walking-reference delta in normalized coordinates."""
    return self._previous_walking_reference_executed / self._walking_reference_scale


@dataclass(kw_only=True)
class AbsoluteWalkingReferenceActionCfg(WalkingReferenceActionCfg):
  """Configuration for driving the walking reference from a command-manager term."""

  walking_velocity_command_name: str = "base_velocity"
  """Command-manager term sent as an absolute ISMPC ``set_ref_vel`` target."""


class AbsoluteWalkingReferenceMixin(WalkingReferenceMixin):
  """Send a command-manager term as the controller's walking-velocity target."""

  cfg: AbsoluteWalkingReferenceActionCfg

  walking_reference_is_absolute = True

  def _setup_action_extensions(self, cfg: McRtcResidualActionCfg) -> None:
    super()._setup_action_extensions(cfg)
    name = self.cfg.walking_velocity_command_name
    command = self._env.command_manager.get_command(name)
    if command is None or tuple(command.shape) != (self.num_envs, 3):
      raise ValueError(f"walking command {name!r} must have shape ({self.num_envs}, 3)")

  def _process_action_extensions(self, actions: torch.Tensor) -> None:
    super()._process_action_extensions(actions)
    self._previous_walking_reference_executed.copy_(self._walking_reference_executed)
    command = self._env.command_manager.get_command(
      self.cfg.walking_velocity_command_name
    )
    assert command is not None
    self._walking_reference_requested.copy_(command)
    self._walking_reference_executed.copy_(command)
    self._walking_reference_active.fill_(True)
    self._feed_walking_reference()
