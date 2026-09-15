"""Reusable base for residual action terms backed by per-env mc_rtc controllers."""

from __future__ import annotations

import abc
import os
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.utils.lab_api.string import resolve_matching_names

import mc_rtc_interface as native
from mc_mjlab.controller_datastore import (
  DatastoreCommands,
  input_columns,
  output_columns,
  read_outputs,
  write_inputs,
)
from mc_mjlab.recovery_authority import RecoveryAuthority
from mc_mjlab.robots import mc_rtc_robot_configuration as mc_rtc
from mc_mjlab.sim_controller_bridge import SimControllerBridge
from utils.pd_gains import apply_reference_pd_gains
from utils.shared_memory import ShmHandle, create_shm, row_window

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class McRtcResidualActionCfg(BaseActionCfg):
  """Shared configuration for mc_rtc residual action terms."""

  mc_rtc_config_path: str
  """Path to the mc_rtc configuration file."""

  mc_rtc_robot_name: str = "jvrc1"
  """Name of the robot in mc_rtc."""

  frameskip: int = 1
  """Physics substeps per controller step (e.g. 5ms control / 1ms physics -> 5)."""

  num_workers: int | None = None
  """Worker process count; ``None`` = ``min(num_envs, cpu_count - 2)``."""

  pd_gains_path: str | None = None
  """Optional reference-order mc_mujoco PD gains overriding entity defaults."""

  residual_actuator_names: tuple[str, ...] | None = None
  """Actuator regexes receiving residuals; ``None`` selects all controlled joints."""

  datastore_scalar_inputs: tuple[str, ...] = ()
  """Native double setters written every control period. docs/coupling.md"""

  datastore_scalar_outputs: tuple[str, ...] = ()
  """Native double getter callbacks collected after each controller step."""

  datastore_vectors_inputs: tuple[str, ...] = ()
  """Native Vector3d setters written every control period. docs/coupling.md"""

  datastore_vectors_outputs: tuple[str, ...] = ()
  """Native Vector3d getters collected each period, without interpolation."""

  controller_timeout_ms: int = 60000
  """Native collection timeout, in milliseconds."""

  print_residual_every: int = 0
  """Policy steps between printing env 0's residual to the terminal; 0 disables."""

  console_output: Literal["none", "single", "all"] = "none"
  """Native row logging: none, environment zero (single), or all."""

  recovery_detector_path: str | None = None
  """Calibration JSON for recovery authority; ``None`` grants full authority."""


class McRtcResidualActionBase(BaseAction):
  """mc_rtc residual action base: steps controllers via a native manager, adds RL residual."""

  cfg: McRtcResidualActionCfg

  output_channels: tuple[str, ...] = ()
  """Public joint output channels; native qd is exposed as alpha."""

  residual_unit: str = ""
  """Unit the residual is expressed in, for the printout. Set by the subclass."""

  def __init__(self, cfg: McRtcResidualActionCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg=cfg, env=env)

    self._mc_rtc_robot_name = cfg.mc_rtc_robot_name
    self._num_targets = len(self._target_names)
    # Every env shares one dispatch phase (nothing resets this per env), so a
    # host int keeps the substep loop free of device synchronisation.
    self._substep = 0

    self._validate_cfg(cfg)

    self._setup_residual(cfg)
    self._setup_datastore_vector_input_commands(cfg)
    self._setup_action_extensions(cfg)
    self._setup_residual_printing(cfg)

    self._build_bridge(cfg)

    self._manager = None
    self._finalizer = None
    self._input_memory = self._output_memory = None
    self._in_np = self._out_np = None
    self._pending_dispatch = False
    self._pending_reset = np.zeros(self.num_envs, dtype=bool)
    self._dispatch_resets = np.zeros(self.num_envs, dtype=bool)

    try:
      self._start_controllers(cfg)
      self._finish_initialization(env, cfg)
    except BaseException:
      self.close()
      raise

  def _validate_cfg(self, cfg: McRtcResidualActionCfg) -> None:
    """Reject configurations the shared-memory pipeline cannot honour."""
    if cfg.frameskip <= 0 or self._env.cfg.decimation % cfg.frameskip:
      raise ValueError("environment decimation must be divisible by positive frameskip")

    if cfg.controller_timeout_ms < 0:
      raise ValueError("controller_timeout_ms must be nonnegative")

    if cfg.console_output not in ("none", "single", "all"):
      raise ValueError(f"invalid console_output: {cfg.console_output!r}")

  def _build_bridge(self, cfg: McRtcResidualActionCfg) -> None:
    """Bind the simulation bridge, declare its datastore columns and load PD gains."""
    self._bridge = SimControllerBridge(
      self._env,
      self._entity,
      self._target_names,
      self._target_ids,
      cfg.mc_rtc_robot_name,
      self.output_channels,
      cfg.entity_name,
    )

    self._bridge.layout.output.datastore_scalar = list(
      dict.fromkeys(cfg.datastore_scalar_outputs)
    )

    self._bridge.layout.output.datastore_vector3 = list(
      dict.fromkeys(cfg.datastore_vectors_outputs)
    )

    # Declared before the command pairs, which append their setters after these.
    self._bridge.layout.input.datastore_scalar = list(
      dict.fromkeys(cfg.datastore_scalar_inputs)
    )

    self._bridge.layout.input.datastore_vector3 = list(
      dict.fromkeys(cfg.datastore_vectors_inputs)
    )

    self._datastore_vector_input_commands = DatastoreCommands(
      self._io.layout,
      self.num_envs,
      self._datastore_vector_input_command_pairs,
      "vector3",
      self._datastore_vector_input_command_is_absolute,
    )

    # After the command pair appended its own getter to the layout.
    self._setup_datastore_outputs(cfg)
    self._setup_datastore_inputs(cfg)

    if cfg.pd_gains_path is not None:
      apply_reference_pd_gains(
        self._entity,
        self._bridge.layout.input.joint_order,
        self._target_names,
        cfg.pd_gains_path,
      )

  def _start_controllers(self, cfg: McRtcResidualActionCfg) -> None:
    """Map the shared-memory blocks, start the worker pool and initialize every row."""
    # Both sizes count the datastore columns declared above; keep this after them.
    self._input_memory = create_shm((self.num_envs, self._bridge.layout.input_size))
    self._output_memory = create_shm((self.num_envs, self._bridge.layout.output_size))
    self._in_np = self._input_memory.arr
    self._out_np = self._output_memory.arr
    self._bridge.fill_controller_input(self._in_np)

    configuration = native.WorkerStartMessage(
      self._bridge.layout,
      native.SharedMemoryDescription(*row_window(self._input_memory, 0, self.num_envs)),
      native.SharedMemoryDescription(
        *row_window(self._output_memory, 0, self.num_envs)
      ),
    )

    workers = (
      cfg.num_workers
      if cfg.num_workers is not None
      else min(self.num_envs, max(1, (os.cpu_count() or 1) - 2))
    )

    self._manager = native.ControllersManager(
      cfg.mc_rtc_config_path,
      self.num_envs,
      workers,
      configuration,
      cfg.controller_timeout_ms,
    )

    self._finalizer = weakref.finalize(
      self,
      self._release_controller,
      self._manager,
      self._input_memory,
      self._output_memory,
    )

    self._manager.dispatch(native.Command.Initialize)
    failed = self._manager.collect()
    status = self._out_np[:, self._bridge.layout.output.status_offset()]

    if failed or np.any(status != int(native.OutputLayout.Status.OK)):
      raise RuntimeError(
        "native controller initialization failed; verify configured numeric "
        "datastore callbacks and the mc_mjlab controller adapter "
        f"(vectors={list(self._bridge.layout.output.datastore_vector3)}, "
        f"scalars={list(self._bridge.layout.output.datastore_scalar)})"
      )

    all_envs = list(range(self.num_envs))
    self._datastore_vector_input_commands.collect(self._out_np, all_envs)

  def _finish_initialization(
    self, env: ManagerBasedRlEnv, cfg: McRtcResidualActionCfg
  ) -> None:
    """Allocate simulation buffers while native resources remain guarded."""
    self._alloc_interpolation_buffers()
    self._alloc_failure_latches()

    self._recovery_authority = (
      RecoveryAuthority(env, self, cfg.recovery_detector_path)
      if cfg.recovery_detector_path is not None
      else None
    )
    if self._recovery_authority is not None:
      self._last_gate.zero_()
      self._previous_gate.zero_()
      self._actor_update_gate.zero_()

  @staticmethod
  def _release_controller(
    manager: native.ControllersManager,
    input_memory: ShmHandle,
    output_memory: ShmHandle,
  ) -> None:
    """Release workers before the memory they may still be accessing."""
    manager.close()
    input_memory.unlink()
    output_memory.unlink()

  def close(self) -> None:
    """Stop the manager before releasing either shared-memory block."""
    if self._finalizer is not None:
      self._finalizer()
      self._finalizer = None
      self._input_memory = self._output_memory = None
    elif self._manager is not None:
      self._manager.close()

    self._manager = None
    self._bridge.release_views()
    self._in_np = self._out_np = None

    for name in ("_input_memory", "_output_memory"):
      memory = getattr(self, name, None)
      if memory is not None:
        memory.unlink()
        setattr(self, name, None)

  def _setup_residual(self, cfg: McRtcResidualActionCfg) -> None:
    """Slice scale/offset/clip down to the residual actuator subset."""
    self._residual_ids: torch.Tensor | None = None
    self._last_gate = torch.ones(self.num_envs, device=self.device)
    self._previous_gate = torch.ones_like(self._last_gate)
    self._actor_update_gate = torch.ones_like(self._last_gate)
    self._torque_peak = torch.zeros(
      self.num_envs, self._num_targets, device=self.device
    )

    if cfg.residual_actuator_names is not None:
      self._restrict_to_residual_actuators(cfg, cfg.residual_actuator_names)

    self._physical_scale = (
      self._scale.abs().clone()
      if isinstance(self._scale, torch.Tensor)
      else torch.full_like(self._raw_actions, abs(self._scale))
    )
    if bool((self._physical_scale <= 0.0).any()):
      raise ValueError("residual action scales must be non-zero")

    self._executed_physical = torch.zeros_like(self._raw_actions)
    self._previous_executed_physical = torch.zeros_like(self._raw_actions)
    self._projection_mask = torch.zeros_like(self._raw_actions, dtype=torch.bool)

    # An extension may append dimensions after these, so the residual slice is
    # recorded before `_setup_action_extensions` can grow `_action_dim`.
    self._residual_action_dim = self._action_dim
    self._residual_raw_actions = self._raw_actions
    self._previous_residual_raw_actions = torch.zeros_like(self._residual_raw_actions)

    self._setup_hardware_bounds()

  def _restrict_to_residual_actuators(
    self, cfg: McRtcResidualActionCfg, names: tuple[str, ...]
  ) -> None:
    """Narrow the action buffers and affine terms to the matched actuators."""
    ids, _ = resolve_matching_names(names, self._target_names)
    self._residual_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
    self._action_dim = len(ids)
    self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)

    if isinstance(self._scale, torch.Tensor):
      self._scale = self._scale[:, ids]
    if isinstance(self._offset, torch.Tensor):
      self._offset = self._offset[:, ids]
    if cfg.clip is not None:
      self._clip = self._clip[:, ids]

    self._residual_full = torch.zeros(
      self.num_envs, self._num_targets, device=self.device
    )

  def _setup_hardware_bounds(self) -> None:
    """Resolve RobotModule position, velocity, and effort bounds to target order."""
    position = mc_rtc.get_position_bounds(self._mc_rtc_robot_name)
    velocity = mc_rtc.get_velocity_bounds(self._mc_rtc_robot_name)
    effort = mc_rtc.get_effort_bounds(self._mc_rtc_robot_name)
    self._position_lower, self._position_upper = self._bound_tensors(*position)
    self._velocity_lower, self._velocity_upper = self._bound_tensors(*velocity)
    self._effort_lower, self._effort_upper = self._bound_tensors(*effort)

  def _bound_tensors(
    self, lower: dict[str, float], upper: dict[str, float]
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert lower/upper maps to broadcastable tensors in target order."""
    required = (
      self._target_names
      if self._residual_ids is None
      else [self._target_names[i] for i in self._residual_ids.tolist()]
    )
    missing = [n for n in required if n not in lower or n not in upper]
    if missing:
      raise KeyError(
        f"the mc_rtc RobotModule reports no bounds for residual joints {missing}; "
        "feasibility cannot be enforced"
      )
    return (
      torch.tensor(
        [lower.get(n, -float("inf")) for n in self._target_names], device=self.device
      ).unsqueeze(0),
      torch.tensor(
        [upper.get(n, float("inf")) for n in self._target_names], device=self.device
      ).unsqueeze(0),
    )

  def _setup_datastore_vector_input_commands(self, cfg: McRtcResidualActionCfg) -> None:
    """Declare no vector datastore commands; an extension may replace these."""
    self._datastore_vector_input_command_pairs: tuple[tuple[str, str], ...] = ()
    self._datastore_vector_input_command_is_absolute: tuple[bool, ...] = ()
    self._datastore_vector_input_active = torch.zeros(
      self.num_envs, 0, dtype=torch.bool, device=self.device
    )
    self._datastore_vector_input_values = torch.zeros(
      self.num_envs, 0, device=self.device
    )

  def _setup_datastore_outputs(self, cfg: McRtcResidualActionCfg) -> None:
    """Resolve the collected getter columns and their latched readouts."""
    self._datastore_vector_output_columns = output_columns(
      self._bridge.layout, cfg.datastore_vectors_outputs, "vector3"
    )
    self._datastore_scalar_output_columns = output_columns(
      self._bridge.layout, cfg.datastore_scalar_outputs, "scalar"
    )
    # Whole-controller vectors: latched as collected, no ramp (see the cfg).
    self._datastore_vector_outputs = {
      getter: torch.zeros(self.num_envs, 3, device=self.device)
      for getter in self._datastore_vector_output_columns
    }
    self._datastore_scalar_outputs = {
      getter: torch.zeros(self.num_envs, device=self.device)
      for getter in self._datastore_scalar_output_columns
    }

  def _setup_datastore_inputs(self, cfg: McRtcResidualActionCfg) -> None:
    """Resolve the unconditionally fed setter columns and their value buffers."""
    self._datastore_scalar_input_columns = input_columns(
      self._bridge.layout, dict.fromkeys(cfg.datastore_scalar_inputs), "scalar"
    )
    self._datastore_vector_input_columns = input_columns(
      self._bridge.layout, dict.fromkeys(cfg.datastore_vectors_inputs), "vector3"
    )
    # Every declared setter is written each period from the first step on, so a
    # task that declares one owns its value from then on. docs/coupling.md
    self._datastore_scalar_input_feed = torch.zeros(
      self.num_envs, len(self._datastore_scalar_input_columns), device=self.device
    )
    self._datastore_vector_input_feed = torch.zeros(
      self.num_envs, len(self._datastore_vector_input_columns), 3, device=self.device
    )

  def _datastore_input_index(
    self, columns: dict[str, int], setter: str, kind: str
  ) -> int:
    """Position of one configured setter in its input value buffer."""
    try:
      return list(columns).index(setter)
    except ValueError:
      raise KeyError(
        f"datastore {kind} input {setter!r} is not configured; add it to the "
        f"action term's `datastore_{kind}_inputs` (have: {sorted(columns)})"
      ) from None

  def datastore_scalar_input(self, setter: str) -> torch.Tensor:
    """Value currently fed to one configured scalar datastore setter."""
    index = self._datastore_input_index(
      self._datastore_scalar_input_columns, setter, "scalar"
    )
    return self._datastore_scalar_input_feed[:, index]

  def datastore_vector_input(self, setter: str) -> torch.Tensor:
    """Value currently fed to one configured vector datastore setter."""
    index = self._datastore_input_index(
      self._datastore_vector_input_columns, setter, "vectors"
    )
    return self._datastore_vector_input_feed[:, index]

  def set_datastore_scalar_input(self, setter: str, values: torch.Tensor) -> None:
    """Feed one scalar datastore setter; the value holds until set again."""
    index = self._datastore_input_index(
      self._datastore_scalar_input_columns, setter, "scalar"
    )
    if tuple(values.shape) != (self.num_envs,):
      raise ValueError(
        f"datastore scalar input shape {tuple(values.shape)}, "
        f"expected {(self.num_envs,)}"
      )
    self._datastore_scalar_input_feed[:, index].copy_(values)

  def set_datastore_vector_input(self, setter: str, values: torch.Tensor) -> None:
    """Feed one vector datastore setter; the value holds until set again."""
    index = self._datastore_input_index(
      self._datastore_vector_input_columns, setter, "vectors"
    )
    if tuple(values.shape) != (self.num_envs, 3):
      raise ValueError(
        f"datastore vector input shape {tuple(values.shape)}, "
        f"expected {(self.num_envs, 3)}"
      )
    self._datastore_vector_input_feed[:, index].copy_(values)

  def _setup_residual_printing(self, cfg: McRtcResidualActionCfg) -> None:
    """Resolve the printout interval and the column labels it prints once."""
    every = cfg.print_residual_every
    override = os.environ.get("MC_MJLAB_PRINT_RESIDUAL")
    if override is not None:
      every = int(override)

    self._print_every = max(0, every)
    self._print_countdown = 0
    self._print_header_pending = True
    self._print_pending = False
    if not self._print_every:
      return

    ids = self._residual_ids
    self._print_names = (
      list(self._target_names)
      if ids is None
      else [self._target_names[i] for i in ids.tolist()]
    )
    # Per-joint clip magnitude, for the saturation marker. The tasks set a
    # symmetric bound from `residual_scale`, so the upper one describes both.
    self._print_limit = (
      self._clip[0, :, 1].abs().cpu().tolist() if cfg.clip is not None else None
    )

  def _zero_channels(self) -> dict[str, torch.Tensor]:
    """One zeroed ``(num_envs, num_targets)`` buffer per output channel."""
    return {
      channel: torch.zeros(self.num_envs, self._num_targets, device=self.device)
      for channel in self.output_channels
    }

  def _alloc_interpolation_buffers(self) -> None:
    """Per-channel ramp endpoints plus the one-period-behind staging buffer."""
    self._previous_control = self._zero_channels()
    self._next_control = self._zero_channels()
    self._staged_control = self._zero_channels()
    # Reused every substep: allocating the blend and the masked swap in the loop
    # would churn the caching allocator for no reason.
    self._interpolated = self._zero_channels()
    self._swap_scratch = self._zero_channels()

    self._has_staged_control = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def _alloc_failure_latches(self) -> None:
    """Allocate the per-episode controller and worker failure flags."""
    # Latched per env until reset; read by the `controller_failed` termination
    # term so a QP giving up ends that episode instead of the whole run.
    self.controller_failed = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    # Apart from the QP latch: losing a worker is exogenous, so the task truncates.
    self.controller_worker_failed = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def _setup_action_extensions(self, cfg: McRtcResidualActionCfg) -> None:
    """Append extra action dimensions past the residual; the base has none."""

  def _process_action_extensions(self, actions: torch.Tensor) -> None:
    """Consume the action dimensions past ``_residual_action_dim``."""

  def _reset_action_extensions(self, env_ids: torch.Tensor | slice) -> None:
    """Clear extension state for the given (reset) envs."""

  @abc.abstractmethod
  def _seed_interpolation(self, env_ids: torch.Tensor) -> None:
    """Seed the interpolation endpoints for the given (reset) envs."""
    raise NotImplementedError

  @abc.abstractmethod
  def _apply_control(
    self, interpolated_control: dict[str, torch.Tensor], residual: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Write targets and return full-width executed residual and projection mask."""
    raise NotImplementedError

  def controller_reference(self, channel: str) -> torch.Tensor:
    """Latest raw controller output for ``channel``, residual excluded."""
    return self._next_control[channel]

  def datastore_vector_output(self, getter: str) -> torch.Tensor:
    """Latest ``(num_envs, 3)`` value of one collected vector datastore getter."""
    try:
      return self._datastore_vector_outputs[getter]
    except KeyError:
      raise KeyError(
        f"datastore vector output {getter!r} is not collected; add it to the "
        f"action term's `datastore_vectors_outputs` "
        f"(have: {sorted(self._datastore_vector_outputs)})"
      ) from None

  def datastore_scalar_output(self, getter: str) -> torch.Tensor:
    """Latest scalar datastore getter value collected from each controller."""
    try:
      return self._datastore_scalar_outputs[getter]
    except KeyError:
      raise KeyError(
        f"datastore scalar output {getter!r} is not configured; add it to the "
        f"action term's `datastore_scalar_outputs` "
        f"(have: {sorted(self._datastore_scalar_outputs)})"
      ) from None

  def consume_torque_peak(self) -> torch.Tensor:
    """Peak |joint torque| since the last call, over the target joints; resets it."""
    peak = self._torque_peak.clone()
    self._torque_peak.zero_()
    return peak

  @property
  def processed_action(self) -> torch.Tensor:
    """Requested physical residual after scale and clip."""
    return self._processed_actions

  @property
  def requested_normalized_action(self) -> torch.Tensor:
    """Policy request in normalized action coordinates."""
    return self._residual_raw_actions

  @property
  def previous_requested_normalized_action(self) -> torch.Tensor:
    """Normalized residual request from the preceding policy step."""
    return self._previous_residual_raw_actions

  @property
  def requested_physical_action(self) -> torch.Tensor:
    """Physical request after affine processing and clip."""
    return self._processed_actions

  @property
  def executed_physical_action(self) -> torch.Tensor:
    """Physical residual delivered after authority and feasibility projection."""
    return self._executed_physical

  @property
  def executed_normalized_action(self) -> torch.Tensor:
    """Executed physical residual divided by its configured scale."""
    return self._executed_physical / self._physical_scale

  @property
  def previous_executed_normalized_action(self) -> torch.Tensor:
    """Executed normalized residual from the preceding policy step."""
    return self._previous_executed_physical / self._physical_scale

  @property
  def projection_mask(self) -> torch.Tensor:
    """Per-residual-joint mask where feasibility changed the gated request."""
    return self._projection_mask

  @property
  def last_gate(self) -> torch.Tensor:
    """Most recent recovery authority in 0..1."""
    return self._last_gate

  @property
  def previous_gate(self) -> torch.Tensor:
    """Recovery authority from the preceding policy step."""
    return self._previous_gate

  @property
  def actor_update_gate(self) -> torch.Tensor:
    """Authority of the last processed action, retained across an auto-reset."""
    return self._actor_update_gate

  @property
  def detector_score(self) -> torch.Tensor:
    """Most recent calibrated detector score, zeros when disabled."""
    if self._recovery_authority is None:
      return torch.zeros_like(self._last_gate)
    return self._recovery_authority.score

  @property
  def recovery_dcm_error_vector(self) -> torch.Tensor:
    """Latest deployable signed horizontal DCM error, in metres."""
    if self._recovery_authority is None:
      return torch.zeros(self.num_envs, 2, device=self.device)
    return self._recovery_authority.dcm_error

  @property
  def residual_ids(self) -> torch.Tensor | None:
    """Columns of the target arrays carrying the residual; ``None`` = all."""
    return self._residual_ids

  @property
  def residual_names(self) -> tuple[str, ...]:
    """Actuator names in residual action-column order."""
    if self._residual_ids is None:
      return tuple(self._target_names)
    return tuple(self._target_names[index] for index in self._residual_ids.tolist())

  @property
  def residual_scale(self) -> torch.Tensor:
    """Absolute physical authority per normalized residual action column."""
    return self._physical_scale

  @property
  def residual_effort_ratio(self) -> torch.Tensor:
    """Current actuator effort divided by the RobotModule limit per residual joint."""
    effort = self._entity.data.qfrc_actuator[:, self._target_ids].abs()
    limit = torch.maximum(self._effort_lower.abs(), self._effort_upper.abs())
    if self._residual_ids is not None:
      effort = effort[:, self._residual_ids]
      limit = limit[:, self._residual_ids]
    return effort / limit.clamp_min(torch.finfo(effort.dtype).eps)

  def process_actions(self, actions: torch.Tensor) -> None:
    self._previous_executed_physical.copy_(self._executed_physical)
    self._previous_residual_raw_actions.copy_(self._residual_raw_actions)
    self._previous_gate.copy_(self._last_gate)

    self._raw_actions.copy_(actions)
    residual = actions[:, : self._residual_action_dim]
    self._residual_raw_actions.copy_(residual)
    self._processed_actions.copy_(residual * self._scale + self._offset)
    if self.cfg.clip is not None:
      self._processed_actions.copy_(
        torch.clamp(
          self._processed_actions,
          min=self._clip[:, :, 0],
          max=self._clip[:, :, 1],
        )
      )

    if self._recovery_authority is not None:
      self._last_gate.copy_(self._recovery_authority.update())
    self._actor_update_gate.copy_(self._last_gate)

    self._process_action_extensions(actions)

    if self._print_every:
      self._print_pending = True

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids=env_ids)

    self._collect_controller_output()

    if env_ids is None:
      env_ids = slice(None)

    self._torque_peak[env_ids] = 0.0
    self._last_gate[env_ids] = 0.0 if self._recovery_authority is not None else 1.0
    self._previous_gate[env_ids] = self._last_gate[env_ids]
    # PPO reads the transition gate after terminal environments auto-reset.
    self._executed_physical[env_ids] = 0.0
    self._previous_executed_physical[env_ids] = 0.0
    self._residual_raw_actions[env_ids] = 0.0
    self._previous_residual_raw_actions[env_ids] = 0.0
    self._projection_mask[env_ids] = False
    self._reset_action_extensions(env_ids)

    if self._recovery_authority is not None:
      self._recovery_authority.reset(env_ids)

    if isinstance(env_ids, slice):
      env_indices = list(range(self.num_envs))[env_ids]
    else:
      env_indices = env_ids.tolist()

    self._manager.respawn(env_indices)
    self._pending_reset[env_indices] = True
    self._datastore_vector_input_commands.reset(env_indices)

    # Seed interpolation (subclass-specific rest value per channel) and discard
    # any staged output for the reset envs; they restart from that seed.
    rows = torch.tensor(env_indices, device=self.device, dtype=torch.long)
    self._seed_interpolation(rows)
    self._has_staged_control[rows] = False

    for readouts in (self._datastore_vector_outputs, self._datastore_scalar_outputs):
      for values in readouts.values():
        values[rows] = 0.0

    # Episode latches clear now; the queued native reset still has to complete.
    self.controller_failed[rows] = False
    self.controller_worker_failed[rows] = False

  def apply_actions(self) -> None:
    substep_in_period = self._substep % self.cfg.frameskip
    self._substep += 1
    if substep_in_period == 0:
      self._advance_control_period()

    # coef=1 on the last substep gives the full new target, matching mc_mujoco.
    interpolation_coef = (substep_in_period + 1) / self.cfg.frameskip
    interpolated_control = {}
    for channel in self.output_channels:
      torch.lerp(
        self._previous_control[channel],
        self._next_control[channel],
        interpolation_coef,
        out=self._interpolated[channel],
      )
      interpolated_control[channel] = self._interpolated[channel]

    # Peak-held across the decimation window: what sizes an actuator is the worst
    # instant, and apply_actions runs before sim.step so this trails by a substep.
    torch.maximum(
      self._torque_peak,
      self._entity.data.qfrc_actuator[:, self._target_ids].abs(),
      out=self._torque_peak,
    )

    residual = self._processed_actions * self._last_gate.unsqueeze(-1)
    if self._residual_ids is not None:
      # Scatter into the full target width; non-matched joints get 0, i.e. pure
      # mc_rtc tracking.
      self._residual_full.zero_()
      self._residual_full[:, self._residual_ids] = residual
      residual = self._residual_full

    executed, projected = self._apply_control(interpolated_control, residual)
    if self._residual_ids is None:
      self._executed_physical.copy_(executed)
      self._projection_mask.copy_(projected)
    else:
      self._executed_physical.copy_(executed[:, self._residual_ids])
      self._projection_mask.copy_(projected[:, self._residual_ids])

    if self._print_pending:
      self._print_residual()
      self._print_pending = False

  def _advance_control_period(self) -> None:
    """Collect the finished solve, roll the ramp endpoints and dispatch the next."""
    assert self._manager is not None
    assert self._in_np is not None and self._out_np is not None

    # Collect the previous period's dispatch (it solved while the intervening
    # sim substeps ran) before reusing the shared I/O blocks.
    self._collect_controller_output()

    # Masked, not indexed: a boolean gather's data-dependent shape would sync.
    fresh = self._has_staged_control.unsqueeze(-1)
    for channel in self.output_channels:
      scratch = self._swap_scratch[channel]
      torch.where(
        fresh, self._next_control[channel], self._previous_control[channel], out=scratch
      )
      self._previous_control[channel].copy_(scratch)
      torch.where(
        fresh, self._staged_control[channel], self._next_control[channel], out=scratch
      )
      self._next_control[channel].copy_(scratch)
    self._has_staged_control.zero_()

    # Sample the current state and dispatch this period's solve without
    # blocking; it overlaps the next `frameskip` substeps of sim.
    self._bridge.fill_controller_input(self._in_np)
    write_inputs(
      self._in_np,
      self._datastore_scalar_input_columns,
      self._datastore_scalar_input_feed,
      "scalar",
    )
    write_inputs(
      self._in_np,
      self._datastore_vector_input_columns,
      self._datastore_vector_input_feed,
      "vector3",
    )
    self._datastore_vector_input_commands.write(
      self._in_np,
      self._datastore_vector_input_active,
      self._datastore_vector_input_values,
    )

    self._dispatch_resets[:] = self._pending_reset
    self._in_np[:, self._bridge.layout.input.reset_offset()] = self._dispatch_resets
    # An unserviced row must never look like a fresh successful result.
    self._out_np[:, self._bridge.layout.output.status_offset()] = int(
      native.OutputLayout.Status.WORKER_FAILED
    )
    self._manager.dispatch(native.Command.Step)
    self._pending_dispatch = True

  def _collect_controller_output(self) -> None:
    """Await the outstanding async step (if any) and stage its outputs."""
    if not self._pending_dispatch:
      return
    assert self._manager is not None and self._out_np is not None

    failed = self._manager.collect()
    self._pending_dispatch = False

    # Merge the worker failures into the block itself, so the single upload
    # below carries the final status and no mask has to cross separately.
    status_column = self._out_np[:, self._bridge.layout.output.status_offset()]
    status_column[failed] = int(native.OutputLayout.Status.WORKER_FAILED)
    status = status_column.copy()

    worker_failed = status == int(native.OutputLayout.Status.WORKER_FAILED)
    self._pending_reset[self._dispatch_resets & ~worker_failed] = False
    self._pending_reset[worker_failed] = True

    failed_indices = np.flatnonzero(worker_failed).tolist()
    self._datastore_vector_input_commands.reset(failed_indices)
    env_indices = np.flatnonzero(status == int(native.OutputLayout.Status.OK)).tolist()

    block = self._bridge.upload_controller_output(self._out_np)
    status_t = block[:, self._bridge.layout.output.status_offset()]
    ok = status_t == int(native.OutputLayout.Status.OK)
    self.controller_failed |= status_t == int(native.OutputLayout.Status.QP_FAILED)
    self.controller_worker_failed |= status_t == int(
      native.OutputLayout.Status.WORKER_FAILED
    )

    # Every status is OK, QP_FAILED or WORKER_FAILED: fresh output means OK.
    self._has_staged_control.copy_(ok)
    fresh = ok.unsqueeze(-1)
    for channel, values in self._bridge.read_controller_output(block).items():
      staged = self._staged_control[channel]
      staged.copy_(torch.where(fresh, values, staged))

    for kind, columns, destination in (
      (
        "vector3",
        self._datastore_vector_output_columns,
        self._datastore_vector_outputs,
      ),
      ("scalar", self._datastore_scalar_output_columns, self._datastore_scalar_outputs),
    ):
      mask = fresh if kind == "vector3" else ok
      for name, value in read_outputs(block, columns, kind).items():
        destination[name].copy_(torch.where(mask, value, destination[name]))

    self._datastore_vector_input_commands.collect(self._out_np, env_indices)

  def _print_residual(self) -> None:
    """One line of env 0's residual, throttled by ``print_residual_every``."""
    if self._print_countdown:
      self._print_countdown -= 1
      return
    self._print_countdown = self._print_every - 1

    values = self._executed_physical[0].detach().cpu().tolist()
    if self._print_header_pending:
      # Lazily, on the first line: a header printed at construction would be
      # buried under mc_rtc's own startup logging long before the first frame.
      unit = f" [{self.residual_unit}]" if self.residual_unit else ""
      print(
        f"[residual] env 0, every {self._print_every} policy step(s){unit}; * = clipped"
      )
      print("[residual] " + " ".join(f"{n:>6s} " for n in self._print_names) + "   |r|")
      self._print_header_pending = False

    limit = self._print_limit
    cells = [
      f"{v:+.3f}" + ("*" if limit is not None and abs(v) >= 0.999 * limit[j] else " ")
      for j, v in enumerate(values)
    ]
    norm = sum(v * v for v in values) ** 0.5
    # Flushed: Python block-buffers into a pipe while spdlog writes to fd 1, so
    # unflushed the two interleave wrongly. docs/coupling.md#console-output
    print("[residual] " + " ".join(cells) + f" {norm:6.3f}", flush=True)
