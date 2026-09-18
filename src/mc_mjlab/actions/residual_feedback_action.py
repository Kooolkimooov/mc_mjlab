"""Residual feedback learning: steer the controller's input, not only its output."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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
from mc_mjlab.actions.residual_mpc_joint_torque_action import (
  ResidualMpcJointTorqueAction,
  ResidualMpcJointTorqueActionCfg,
)
from mc_mjlab.residuals.printer import ResidualPrinter

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

#: Root pose carries 3 translation and 3 rotation channels; joint position is
#: sized by the residual joints. docs/residual-feedback.md#feedback_modalities
ROOT_POSE_DIM = 6
SUPPORTED_MODALITIES = ("joint_position", "joint_velocity", "root_pose", "wrench")

#: A force sensor's slot is a force triple then a moment triple.
WRENCH_STRIDE = 6
TRIPLE = 3

FORCE_AXES = ("Fx", "Fy", "Fz")
MOMENT_AXES = ("Mx", "My", "Mz")
ROOT_POSE_AXES = ("x", "y", "z", "rx", "ry", "rz")


@dataclass(kw_only=True)
class ResidualFeedbackActionCfg(McRtcResidualActionCfg):
  """Shared configuration for actions that offset the state mc_rtc reads."""

  feedback_modalities: tuple[str, ...] = field(
    default_factory=lambda: ("joint_position",)
  )
  """Feedback spaces the policy may offset, in action order."""

  feedback_scale: float = 0.02
  """Radians of encoder offset at a saturated joint_position action."""

  root_translation_scale: float = 0.0025
  """Metres of root offset at a saturated root_pose translation action."""

  root_rotation_scale: float = 0.02
  """Radians of root tilt at a saturated root_pose rotation action."""

  joint_velocity_scale: float = 0.05
  """Rad/s of encoder-velocity offset at a saturated joint_velocity action."""

  wrench_force_scale: float = 50.0
  """Newtons of force offset at a saturated wrench action."""

  wrench_torque_scale: float = 20.0
  """Newton-metres of moment offset at a saturated wrench action."""

  wrench_sensor_names: tuple[str, ...] = ()
  """Force sensors the wrench modality offsets; empty offsets every one of them."""

  wrench_force_only: bool = False
  """Offset each selected sensor's force triple and leave its moments measured."""

  gate_scalar_outputs: tuple[str, ...] = ()
  """Datastore scalars that must all read ``gate_value`` before feedback goes out."""

  gate_value: float = 0.0

  def __post_init__(self) -> None:
    """Reject an unusable feedback spec here, not minutes into a training run."""
    super().__post_init__()
    if self.feedback_scale <= 0.0:
      raise ValueError("feedback_scale must be positive")
    if not math.isfinite(self.wrench_force_scale) or self.wrench_force_scale <= 0.0:
      raise ValueError("wrench_force_scale must be finite and positive")
    if not self.feedback_modalities:
      raise ValueError("at least one feedback modality is required")

    unknown = set(self.feedback_modalities) - set(SUPPORTED_MODALITIES)
    if unknown:
      raise ValueError(f"unsupported feedback modalities: {sorted(unknown)}")
    if len(set(self.feedback_modalities)) != len(self.feedback_modalities):
      raise ValueError("feedback modalities must be unique")
    if len(set(self.wrench_sensor_names)) != len(self.wrench_sensor_names):
      raise ValueError("wrench_sensor_names must be distinct")
    if self.gate_scalar_outputs and self.recovery_detector_path is not None:
      raise ValueError("a datastore gate and recovery authority both drive the gate")

    self.datastore_scalar_outputs = tuple(
      dict.fromkeys((*self.datastore_scalar_outputs, *self.gate_scalar_outputs))
    )


class ResidualFeedbackActionBase(McRtcResidualActionBase):
  """Add a residual to the state mc_rtc reads, so it replans with the offset."""

  cfg: ResidualFeedbackActionCfg

  def process_actions(self, actions: torch.Tensor) -> None:
    """Keep the trailing feedback block and pass the residual head down."""
    actions = actions.clamp(-1.0, 1.0)
    width = actions.shape[1] - self._feedback_dim

    self._feedback_previous_requested.copy_(self._feedback_requested)
    self._feedback_previous_executed.copy_(self._feedback_executed)
    self._feedback_requested.copy_(actions[:, width:])
    self._feedback_physical.copy_(self._feedback_requested * self._feedback_scale)

    super().process_actions(actions[:, :width])

  def _build_bridge(self, cfg: McRtcResidualActionCfg) -> None:
    """Size the blocks against the live layout, then bind them to the bridge."""
    super()._build_bridge(cfg)
    ids = self._residual_ids
    joint_dim = self._num_targets if ids is None else int(ids.numel())
    self._resolve_wrench_columns()

    self._modality_dims = {
      "joint_position": joint_dim,
      "joint_velocity": joint_dim,
      "root_pose": ROOT_POSE_DIM,
      "wrench": int(self._wrench_columns.numel()),
    }
    self._modalities = tuple(self.cfg.feedback_modalities)
    if "wrench" in self._modalities and not self._modality_dims["wrench"]:
      raise ValueError("wrench feedback needs force sensors, the model has none")

    self._feedback_dim = sum(self._modality_dims[m] for m in self._modalities)
    # Only the env-facing width grows: `_raw_actions` keeps the base class's
    # size because `process_actions` hands it the block ahead of the feedback.
    self._action_dim += self._feedback_dim

    self._alloc_feedback_buffers(joint_dim)
    self._bind_feedback_offsets()

    print(
      f"[mc_rtc] ResidualFeedback: {self._feedback_dim} channel(s) across "
      f"{list(self._modalities)}."
    )

  def _setup_residual_printer(self, cfg: McRtcResidualActionCfg) -> None:
    """Print the feedback block when there is no joint residual to print."""
    if self._residual_action_dim:
      super()._setup_residual_printer(cfg)
      return
    self._printer = ResidualPrinter(
      cfg.print_residual_every,
      list(self.feedback_names),
      self._feedback_scale[0].abs().cpu().tolist(),
      self.feedback_unit,
    )

  def _advance_action_extensions(self) -> None:
    """Refresh the gate on the collected outputs, then write the next offsets."""
    super()._advance_action_extensions()
    self._refresh_feedback_gate()

    gate = self._last_gate.unsqueeze(-1)
    self._feedback_executed.copy_(self._feedback_physical * gate)
    blocks = self._modality_slices(self._feedback_executed)

    if "joint_position" in blocks:
      self._scatter_joint_block(blocks["joint_position"], self._feedback_offset)
    if "joint_velocity" in blocks:
      self._scatter_joint_block(blocks["joint_velocity"], self._joint_velocity_offset)
    if "root_pose" in blocks:
      root = blocks["root_pose"]
      self._root_translation.copy_(root[:, :3])
      self._root_rotation.copy_(root[:, 3:])
    if "wrench" in blocks:
      self._wrench_offset.zero_()
      self._wrench_offset[:, self._wrench_columns] = blocks["wrench"]

  def _reset_action_extensions(self, env_ids: torch.Tensor | slice) -> None:
    """Clear this row's offsets, including the tensors already bound to the bridge."""
    super()._reset_action_extensions(env_ids)
    for values in (
      self._feedback_requested,
      self._feedback_previous_requested,
      self._feedback_physical,
      self._feedback_executed,
      self._feedback_previous_executed,
      self._feedback_offset,
      self._joint_velocity_offset,
      self._root_translation,
      self._root_rotation,
      self._wrench_offset,
    ):
      values[env_ids] = 0.0

    if self.cfg.gate_scalar_outputs:
      # Fresh outputs have to re-open the gate; the base opened it on reset.
      self._last_gate[env_ids] = 0.0
      self._previous_gate[env_ids] = 0.0

  def _resolve_wrench_columns(self) -> None:
    """Map the selected sensors onto columns of the layout's full wrench block."""
    sensors = list(self._bridge.layout.input.force_sensors)
    selected = self.cfg.wrench_sensor_names or tuple(sensors)
    width = TRIPLE if self.cfg.wrench_force_only else WRENCH_STRIDE

    columns = []
    for name in selected:
      if name not in sensors:
        raise ValueError(f"controller robot has no force sensor {name!r}")
      columns.extend(
        WRENCH_STRIDE * sensors.index(name) + axis for axis in range(width)
      )

    self._wrench_sensors = tuple(selected)
    self._wrench_columns = torch.tensor(columns, device=self.device, dtype=torch.long)

  def _alloc_feedback_buffers(self, joint_dim: int) -> None:
    """Allocate the policy-facing block and one offset tensor per modality."""
    self._feedback_scale = torch.tensor(
      [scale for name in self._modalities for scale in self._modality_scales(name)],
      device=self.device,
    ).expand(self.num_envs, self._feedback_dim)

    self._feedback_requested = torch.zeros(
      self.num_envs, self._feedback_dim, device=self.device
    )
    self._feedback_previous_requested = torch.zeros_like(self._feedback_requested)
    self._feedback_physical = torch.zeros_like(self._feedback_requested)
    self._feedback_executed = torch.zeros_like(self._feedback_requested)
    self._feedback_previous_executed = torch.zeros_like(self._feedback_requested)
    self._feedback_projection = torch.zeros_like(
      self._feedback_requested, dtype=torch.bool
    )

    self._feedback_offset = torch.zeros(
      self.num_envs, self._num_targets, device=self.device
    )
    self._joint_velocity_offset = torch.zeros_like(self._feedback_offset)
    self._root_translation = torch.zeros(self.num_envs, 3, device=self.device)
    self._root_rotation = torch.zeros_like(self._root_translation)
    self._wrench_offset = torch.zeros(
      self.num_envs,
      WRENCH_STRIDE * len(self._bridge.layout.input.force_sensors),
      device=self.device,
    )

    if self.cfg.gate_scalar_outputs:
      self._last_gate.zero_()
      self._previous_gate.zero_()
      self._actor_update_gate.zero_()

  def _bind_feedback_offsets(self) -> None:
    """Hand the bridge only the modalities this action owns; the rest stay measured."""
    if "joint_position" in self._modalities:
      self._bridge.set_feedback_offset(self._feedback_offset)
    if "joint_velocity" in self._modalities:
      self._bridge.set_joint_velocity_offset(self._joint_velocity_offset)
    if "root_pose" in self._modalities:
      self._bridge.set_root_pose_offset(self._root_translation, self._root_rotation)
    if "wrench" in self._modalities:
      self._bridge.set_wrench_offset(self._wrench_offset)

  def _modality_scales(self, name: str) -> list[float]:
    """Physical units per saturated action, one entry per column of a modality."""
    cfg = self.cfg
    width = self._modality_dims[name]
    if name == "joint_position":
      return [cfg.feedback_scale] * width
    if name == "joint_velocity":
      return [cfg.joint_velocity_scale] * width
    if name == "root_pose":
      return [cfg.root_translation_scale] * 3 + [cfg.root_rotation_scale] * 3

    force = [cfg.wrench_force_scale] * TRIPLE
    if cfg.wrench_force_only:
      return force * len(self._wrench_sensors)
    return (force + [cfg.wrench_torque_scale] * TRIPLE) * len(self._wrench_sensors)

  def _refresh_feedback_gate(self) -> None:
    """Close the gate unless every named scalar reads the value the cfg wants."""
    if not self.cfg.gate_scalar_outputs:
      return

    active = (
      self._datastore_output_fresh
      & ~self.controller_failed
      & ~self.controller_worker_failed
    )
    for name in self.cfg.gate_scalar_outputs:
      active = active & (self.datastore_scalar_output(name) == self.cfg.gate_value)

    self._last_gate.copy_(active)
    self._actor_update_gate.copy_(self._last_gate)

  def _scatter_joint_block(self, block: torch.Tensor, out: torch.Tensor) -> None:
    """Widen a residual-joint block to every target, leaving the rest unbiased."""
    if self._residual_ids is None:
      out.copy_(block)
      return
    out.zero_()
    out[:, self._residual_ids] = block

  def _modality_slices(self, feedback: torch.Tensor) -> dict[str, torch.Tensor]:
    """Split the feedback block into its modalities, in cfg order."""
    out: dict[str, torch.Tensor] = {}
    start = 0

    for name in self._modalities:
      width = self._modality_dims[name]
      out[name] = feedback[:, start : start + width]
      start += width
    return out

  @property
  def feedback_names(self) -> tuple[str, ...]:
    """Column labels of the feedback block, in action order."""
    names: list[str] = []
    for modality in self._modalities:
      if modality == "joint_position":
        names.extend(f"{joint}/q" for joint in self.residual_joint_names)
      elif modality == "joint_velocity":
        names.extend(f"{joint}/qd" for joint in self.residual_joint_names)
      elif modality == "root_pose":
        names.extend(f"root/{axis}" for axis in ROOT_POSE_AXES)
      else:
        axes = FORCE_AXES if self.cfg.wrench_force_only else FORCE_AXES + MOMENT_AXES
        names.extend(f"{s}/{a}" for s in self._wrench_sensors for a in axes)
    return tuple(names)

  @property
  def feedback_unit(self) -> str:
    """The printout's unit, where every enabled modality shares one."""
    if self._modalities == ("wrench",) and self.cfg.wrench_force_only:
      return "N"
    if self._modalities in (("joint_position",), ("root_pose",)):
      return "rad"
    return ""

  @property
  def residual_joint_names(self) -> tuple[str, ...]:
    """Target names the joint modalities offset, in their own column order."""
    if self._residual_ids is None:
      return tuple(self._target_names)
    return tuple(self._target_names[index] for index in self._residual_ids.tolist())


@dataclass(kw_only=True)
class ResidualFeedbackJointTorqueActionCfg(
  ResidualFeedbackActionCfg, ResidualMpcJointTorqueActionCfg
):
  """Configuration for Ranjbar-style residual feedback beside the torque residual."""

  torque_channel: bool = True
  """Keep the torque residual; False gives the paper's feedback-only variant."""

  def build(self, env: ManagerBasedRlEnv) -> ResidualFeedbackJointTorqueAction:
    return ResidualFeedbackJointTorqueAction(self, env)


class ResidualFeedbackJointTorqueAction(
  ResidualFeedbackActionBase, ResidualMpcJointTorqueAction
):
  """Feedback offsets beside the MPC torque residual the paper blends."""

  cfg: ResidualFeedbackJointTorqueActionCfg

  def process_actions(self, actions: torch.Tensor) -> None:
    """Silence the torque residual first, so its own history records the zeros."""
    if not self.cfg.torque_channel:
      actions = actions.clone()
      actions[:, : self._residual_action_dim] = 0.0
    super().process_actions(actions)


@dataclass(kw_only=True)
class ResidualFeedbackJointPositionActionCfg(
  ResidualFeedbackActionCfg, McRtcResidualJointPositionActionCfg
):
  """Configuration for feedback offsets around a position-controlled robot."""

  residual_actuator_names: tuple[str, ...] | None = ()
  """Empty by default: this action exists to offset feedback, not joint targets."""

  def __post_init__(self) -> None:
    """Refuse the joint residual whose accounting this action would misreport."""
    super().__post_init__()
    if self.residual_actuator_names != ():
      raise ValueError("feedback beside an empty joint residual only, on this action")

  def build(self, env: ManagerBasedRlEnv) -> ResidualFeedbackJointPositionAction:
    return ResidualFeedbackJointPositionAction(self, env)


class ResidualFeedbackJointPositionAction(
  ResidualFeedbackActionBase, McRtcResidualJointPositionAction
):
  """Offset the controller's feedback while mc_rtc keeps producing the targets."""

  cfg: ResidualFeedbackJointPositionActionCfg

  @property
  def raw_action(self) -> torch.Tensor:
    """The feedback block is the whole action here."""
    return self._feedback_requested

  @property
  def requested_normalized_action(self) -> torch.Tensor:
    """Bounded feedback requests, which are what the action penalties cost."""
    return self._feedback_requested

  @property
  def previous_requested_normalized_action(self) -> torch.Tensor:
    """The preceding policy step's normalized feedback requests."""
    return self._feedback_previous_requested

  @property
  def requested_physical_action(self) -> torch.Tensor:
    """Feedback offsets before the gate, in each modality's physical units."""
    return self._feedback_physical

  @property
  def executed_physical_action(self) -> torch.Tensor:
    """Feedback offsets sent on the most recent dispatch."""
    return self._feedback_executed

  @property
  def executed_normalized_action(self) -> torch.Tensor:
    """Dispatched feedback in normalized policy coordinates."""
    return self._feedback_executed / self._feedback_scale

  @property
  def previous_executed_normalized_action(self) -> torch.Tensor:
    """Normalized feedback at the end of the preceding policy step."""
    return self._feedback_previous_executed / self._feedback_scale

  @property
  def projection_mask(self) -> torch.Tensor:
    """Feedback has no actuator feasibility projection."""
    return self._feedback_projection

  @property
  def residual_names(self) -> tuple[str, ...]:
    """Feedback column labels, since there is no joint residual to name."""
    return self.feedback_names

  @property
  def residual_scale(self) -> torch.Tensor:
    """Physical units per unit action for each feedback column."""
    return self._feedback_scale
