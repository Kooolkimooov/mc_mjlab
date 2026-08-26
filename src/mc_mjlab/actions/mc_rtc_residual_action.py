"""Reusable base for residual action terms backed by per-env mc_rtc controllers."""

from __future__ import annotations

import abc
import math
import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.utils.lab_api.string import resolve_matching_names

from mc_mjlab.actions.mc_rtc_controller_host import STATUS_OK, HostMetadata
from mc_mjlab.actions.mc_rtc_controller_io_binding import (
  ControllerIoBinding,
  apply_reference_pd_gains,
)
from mc_mjlab.actions.mc_rtc_controller_pool import ControllerPool
from mc_mjlab.recovery_authority import RecoveryAuthority
from mc_mjlab.robots import mc_rtc_robot_configuration as mc_rtc

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


WALKING_REFERENCE_CALLBACKS = (
  "ismpc_walking::get_ref_vel",
  "ismpc_walking::set_ref_vel",
)


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
  """Worker process count; ``None`` = ``min(num_envs, cpu_count - 2)``.

  With ``use_worker_processes=False``: thread count of the in-process pool."""

  use_worker_processes: bool = True
  """When False, host controllers in-process (serial or threaded) -- mainly for
  debugging, since only ``run()`` releases the GIL."""

  pd_gains_path: str | None = None
  """Optional mc_mujoco ``PDgains_sim.dat`` (one ``kp kd`` row per refJointOrder
  joint) overriding the entity's PD gains. Without the real gains a walking
  controller's trajectory is not tracked and the robot falls."""

  residual_actuator_names: tuple[str, ...] | None = None
  """Actuator names (regex) receiving the RL residual; ``None`` = all controlled
  joints. Non-matched joints track the raw mc_rtc output."""

  controller_vectors: tuple[str, ...] = ()
  """Whole-controller 3-vectors to read off every controller step, named after
  ``mc_rtc_controller_host.VECTOR_OUTPUTS`` (e.g. ``"planned_zmp"``). Unlike the
  joint channels these are not per-joint and are not interpolated across
  substeps; mdp terms read the latest value through ``controller_vector``."""

  use_controller_reset: bool = True
  """Reset via ``MCGlobalController.reset()`` (mc_mujoco parity). Requires the
  locally patched mc_rtc (stock fsm::Controller segfaults on destruction, see
  the GUI/StateBuilder fix). When False, resets re-run ``init()``, which raises
  for plugins that register datastore entries."""

  print_residual_every: int = 0
  """Policy steps between printing env 0's residual to the terminal; 0 disables.

  For watching a `play` session: what the policy is actually adding to the
  controller, per joint, in the control channel's own unit, with a ``*`` on any
  joint sitting at its clip. Both tasks' play variants switch it on -- `play`
  exposes no `--env.*` overrides, so the cfg is the only place it can be set --
  and ``MC_MJLAB_PRINT_RESIDUAL=<n>`` overrides the interval (0 to silence) for
  a run already going, like ``MC_MJLAB_WORKER_LOG_DIR`` does for worker logs."""

  console_output: Literal["none", "single", "all"] = "none"
  """mc_rtc terminal output: "none" silences every controller, "single" lets
  only env 0's controller print (it gets a dedicated worker process), "all"
  suppresses nothing. Workers are silenced by an fd redirect at startup (no
  per-step cost) and their error replies still carry the captured mc_rtc
  output. In-process hosting falls back to per-call fd guards; a threaded
  in-process pool honors "single" only at construction/reset."""

  recovery_detector_path: str | None = None
  """Calibration JSON for recovery-conditioned authority; ``None`` gives full
  authority for compatibility and detector calibration runs."""

  walking_reference_velocity_scale: tuple[float, float, float] | None = None
  """Maximum recovery-gated ``(vx, vy, yaw_rate)`` delta sent through the
  walking controller's datastore; ``None`` preserves the existing action space."""

  walking_reference_velocity_slew_rate: tuple[float, float, float] = (
    2.0,
    1.5,
    3.0,
  )
  """Maximum physical command change per second while authority is nonzero."""

  datastore_scalar_commands: tuple[tuple[str, str], ...] = ()
  """Probe-only paired scalar getter/setters; values are baseline-relative."""


class McRtcResidualActionBase(BaseAction):
  """mc_rtc residual action base: steps controllers via a pool, adds RL residual."""

  cfg: McRtcResidualActionCfg

  output_channels: tuple[str, ...] = ()
  """Controller output channels consumed, in output-block order (must match what
  the host writes to ``out_np``). Set by the subclass."""

  residual_unit: str = ""
  """Unit the residual is expressed in, for the printout. Set by the subclass."""

  def __init__(self, cfg: McRtcResidualActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    self.cfg = cfg
    self._mc_rtc_robot_name = cfg.mc_rtc_robot_name
    self._num_targets = len(self._target_names)

    self._setup_residual(cfg)
    self._setup_walking_reference(cfg)
    self._setup_datastore_scalar_commands(cfg)
    self._setup_residual_printing(cfg)

    self._steps_since_run = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    # Controller transport: workers (or the in-process host), pipes and the
    # shared I/O blocks. The spawn is non-blocking so controller construction
    # overlaps the metadata wait.
    self._pool = ControllerPool(
      cfg.mc_rtc_config_path,
      self.num_envs,
      self._target_names,
      num_workers=cfg.num_workers,
      use_worker_processes=cfg.use_worker_processes,
      console_output=cfg.console_output,
    )
    metadata = self._pool.await_ready()
    self._check_initial_pose_agreement(metadata)

    # Sim <-> mc_rtc input wiring (model introspection, IoLayout, per-step fill).
    self._io = ControllerIoBinding(
      self._env,
      self._entity,
      self._target_names,
      self._target_ids,
      metadata,
      cfg.use_controller_reset,
      self.output_channels,
      cfg.controller_vectors,
      self._datastore_vector_commands,
      self._datastore_scalar_commands,
    )
    # refJointOrder is only known now, so the gain override is applied here.
    if cfg.pd_gains_path is not None:
      apply_reference_pd_gains(
        self._entity, metadata.ref_joint_order, self._target_names, cfg.pd_gains_path
      )

    self._pool.configure(self._io.layout)
    # Aliases of the pool's shared blocks; filled/read directly on the hot path.
    self._in_np = self._pool.in_np
    self._out_np = self._pool.out_np

    self._alloc_interpolation_buffers()
    self._recovery_authority = (
      RecoveryAuthority(env, self, cfg.recovery_detector_path)
      if cfg.recovery_detector_path is not None
      else None
    )
    if self._recovery_authority is not None:
      self._last_gate.zero_()

  # ---- Construction helpers. ----

  # Tolerances for the initial-pose agreement check. Position is generous
  # because a centimetre of disagreement is harmless; heading is tight because
  # it is not -- see the measurements quoted below.
  _POSE_TOL_M = 0.05
  _YAW_TOL_RAD = 0.10

  def _check_initial_pose_agreement(self, metadata: HostMetadata) -> None:
    """One line if the controller's assumed start pose differs from the sim's."""
    if metadata.assumed_base_pose is None:
      return
    ax, ay, az, ayaw = metadata.assumed_base_pose
    init = self._entity.cfg.init_state
    sx, sy, sz = (float(v) for v in init.pos)
    w, x, y, z = (float(v) for v in init.rot)
    syaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    dyaw = abs(math.atan2(math.sin(syaw - ayaw), math.cos(syaw - ayaw)))
    dpos = math.dist((sx, sy, sz), (ax, ay, az))
    if dpos > self._POSE_TOL_M or dyaw > self._YAW_TOL_RAD:
      print(
        f"[mc_rtc] controller config starts at yaw={ayaw:+.3f}, sim resets to "
        f"yaw={syaw:+.3f} ({dpos:.3f} m, {dyaw:.3f} rad apart); reconciled per "
        f"episode by the reset teleport.",
        file=sys.stderr,
        flush=True,
      )

  def close(self) -> None:
    """Stop controller workers and release their shared memory."""
    self._pool.close()

  def _setup_residual(self, cfg: McRtcResidualActionCfg) -> None:
    """Slice scale/offset/clip down to the residual actuator subset."""
    self._residual_ids: torch.Tensor | None = None
    self._last_gate = torch.ones(self.num_envs, device=self.device)
    self._torque_peak = torch.zeros(
      self.num_envs, self._num_targets, device=self.device
    )
    if cfg.residual_actuator_names is not None:
      ids, _ = resolve_matching_names(cfg.residual_actuator_names, self._target_names)
      self._residual_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
      self._action_dim = len(ids)
      self._raw_actions = torch.zeros(
        self.num_envs, self._action_dim, device=self.device
      )
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
    self._setup_hardware_bounds()

  def _setup_walking_reference(self, cfg: McRtcResidualActionCfg) -> None:
    """Append an optional normalized Vector3 walking-reference action."""
    self._residual_action_dim = self._action_dim
    self._residual_raw_actions = self._raw_actions
    self._datastore_vector_commands: tuple[tuple[str, str], ...] = ()
    self._walking_reference_scale = torch.empty(0, device=self.device)
    self._walking_reference_requested = torch.empty(
      self.num_envs, 0, device=self.device
    )
    self._walking_reference_executed = torch.empty_like(
      self._walking_reference_requested
    )
    self._previous_walking_reference_executed = torch.empty_like(
      self._walking_reference_requested
    )
    self._walking_reference_active = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    scale = cfg.walking_reference_velocity_scale
    if scale is None:
      return
    if len(scale) != 3 or any(value <= 0.0 for value in scale):
      raise ValueError(
        "walking reference velocity scales must be three positive values"
      )
    slew = cfg.walking_reference_velocity_slew_rate
    if len(slew) != 3 or any(value <= 0.0 for value in slew):
      raise ValueError("walking reference slew rates must be three positive values")
    self._datastore_vector_commands = (WALKING_REFERENCE_CALLBACKS,)
    self._walking_reference_scale = torch.tensor(scale, device=self.device).unsqueeze(0)
    self._walking_reference_slew = (
      torch.tensor(slew, device=self.device).unsqueeze(0) * self._env.step_dt
    )
    self._walking_reference_requested = torch.zeros(
      self.num_envs, 3, device=self.device
    )
    self._walking_reference_executed = torch.zeros_like(
      self._walking_reference_requested
    )
    self._previous_walking_reference_executed = torch.zeros_like(
      self._walking_reference_requested
    )
    self._action_dim += 3
    self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)

  def _setup_datastore_scalar_commands(self, cfg: McRtcResidualActionCfg) -> None:
    """Allocate probe-controlled scalar datastore deltas without policy actions."""
    self._datastore_scalar_commands = tuple(cfg.datastore_scalar_commands)
    count = len(self._datastore_scalar_commands)
    self._datastore_scalar_active = torch.zeros(
      self.num_envs, count, dtype=torch.bool, device=self.device
    )
    self._datastore_scalar_delta = torch.zeros(self.num_envs, count, device=self.device)

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

  def _alloc_interpolation_buffers(self) -> None:
    """Per-channel ramp endpoints plus the one-period-behind staging buffer."""
    assert self._io.layout.output_channels == self.output_channels, (
      f"shared block carries {self._io.layout.output_channels}, "
      f"but this action consumes {self.output_channels}"
    )
    self._previous_control = {
      c: torch.zeros(self.num_envs, self._num_targets, device=self.device)
      for c in self.output_channels
    }
    self._next_control = {
      c: torch.zeros(self.num_envs, self._num_targets, device=self.device)
      for c in self.output_channels
    }
    self._staged_control = {
      c: torch.zeros(self.num_envs, self._num_targets, device=self.device)
      for c in self.output_channels
    }
    self._has_staged_control = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    # Whole-controller vectors: latched as collected, no ramp (see the cfg).
    self._controller_vectors = {
      v: torch.zeros(self.num_envs, 3, device=self.device)
      for v in self.cfg.controller_vectors
    }
    self._controller_scalars = {
      getter: torch.zeros(self.num_envs, device=self.device)
      for getter, _ in self._datastore_scalar_commands
    }
    self._controller_scalar_baselines = {
      getter: torch.zeros(self.num_envs, device=self.device)
      for getter, _ in self._datastore_scalar_commands
    }
    # Latched per env until reset; read by the `controller_failed` termination
    # term so a QP giving up ends that episode instead of the whole run.
    self.controller_failed = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    # Apart from the QP latch: losing a worker is exogenous, so the task truncates.
    self.controller_worker_failed = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  # ---- Pipeline. ----

  def _collect_controller_output(self) -> None:
    """Await the outstanding async step (if any) and stage its outputs."""
    env_indices = self._pool.collect()
    if env_indices is None:
      return
    new_output = self._io.read_controller_output(self._out_np, env_indices)
    env_indices_t = torch.tensor(env_indices, device=self.device, dtype=torch.long)
    for c in self.output_channels:
      self._staged_control[c][env_indices_t] = new_output[c]
    self._has_staged_control[env_indices_t] = True
    if self._controller_vectors:
      new_vectors = self._io.read_controller_vectors(self._out_np, env_indices)
      for v, values in new_vectors.items():
        self._controller_vectors[v][env_indices_t] = values
    if self._controller_scalars:
      new_scalars = self._io.read_datastore_scalar_commands(self._out_np, env_indices)
      for name, values in new_scalars.items():
        self._controller_scalars[name][env_indices_t] = values
      new_baselines = self._io.read_datastore_scalar_baselines(
        self._out_np, env_indices
      )
      for name, values in new_baselines.items():
        self._controller_scalar_baselines[name][env_indices_t] = values
    # Latch (not assign): the flags must survive until this env is reset, even
    # though the substeps in between keep collecting.
    qp_failed, worker_failed = self._io.read_controller_failed(
      self._out_np, env_indices
    )
    self.controller_failed[env_indices_t] |= qp_failed
    self.controller_worker_failed[env_indices_t] |= worker_failed

  # ---- Introspection (the task's mdp terms read the reference through this). ----

  def controller_reference(self, channel: str) -> torch.Tensor:
    """Latest raw controller output for ``channel``, residual excluded."""
    return self._next_control[channel]

  def controller_vector(self, name: str) -> torch.Tensor:
    """Latest ``(num_envs, 3)`` value of the ``controller_vectors`` entry."""
    try:
      return self._controller_vectors[name]
    except KeyError:
      raise KeyError(
        f"controller vector {name!r} is not collected; add it to the action "
        f"term's `controller_vectors` (have: {sorted(self._controller_vectors)})"
      ) from None

  def controller_scalar(self, getter: str) -> torch.Tensor:
    """Latest scalar datastore getter value collected from each controller."""
    try:
      return self._controller_scalars[getter]
    except KeyError:
      raise KeyError(
        f"controller scalar {getter!r} is not configured; have: "
        f"{sorted(self._controller_scalars)}"
      ) from None

  def controller_scalar_baseline(self, getter: str) -> torch.Tensor:
    """Baseline captured when a scalar datastore command became active."""
    try:
      return self._controller_scalar_baselines[getter]
    except KeyError:
      raise KeyError(
        f"controller scalar {getter!r} is not configured; have: "
        f"{sorted(self._controller_scalar_baselines)}"
      ) from None

  def set_datastore_scalar_delta(
    self, getter: str, active: torch.Tensor, delta: torch.Tensor
  ) -> None:
    """Set one probe-only scalar command as a delta from its live baseline."""
    getters = [pair[0] for pair in self._datastore_scalar_commands]
    try:
      index = getters.index(getter)
    except ValueError:
      raise KeyError(
        f"scalar datastore getter {getter!r} is not configured; have: {getters}"
      ) from None
    expected = (self.num_envs,)
    if tuple(active.shape) != expected or tuple(delta.shape) != expected:
      raise ValueError(
        f"scalar command shapes {tuple(active.shape)} and {tuple(delta.shape)}, "
        f"expected {expected}"
      )
    self._datastore_scalar_active[:, index].copy_(active)
    self._datastore_scalar_delta[:, index].copy_(delta)

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
  def walking_reference_velocity(self) -> torch.Tensor:
    """Executed ``(vx, vy, yaw_rate)`` delta in physical units."""
    return self._walking_reference_executed

  @property
  def walking_reference_normalized(self) -> torch.Tensor:
    """Executed walking-reference delta in normalized action coordinates."""
    if self._walking_reference_scale.numel() == 0:
      return self._walking_reference_executed
    return self._walking_reference_executed / self._walking_reference_scale

  @property
  def previous_walking_reference_normalized(self) -> torch.Tensor:
    """Previous executed walking-reference delta in normalized coordinates."""
    if self._walking_reference_scale.numel() == 0:
      return self._previous_walking_reference_executed
    return self._previous_walking_reference_executed / self._walking_reference_scale

  @property
  def projection_mask(self) -> torch.Tensor:
    """Per-residual-joint mask where feasibility changed the gated request."""
    return self._projection_mask

  @property
  def last_gate(self) -> torch.Tensor:
    """Most recent recovery authority in 0..1."""
    return self._last_gate

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

  # ---- Subclass hooks. ----

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

  # ---- ActionTerm API. ----

  def process_actions(self, actions: torch.Tensor) -> None:
    self._previous_executed_physical.copy_(self._executed_physical)
    self._previous_walking_reference_executed.copy_(self._walking_reference_executed)
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
    if self._walking_reference_scale.numel():
      normalized = actions[:, self._residual_action_dim :].clamp(-1.0, 1.0)
      self._walking_reference_requested.copy_(
        normalized * self._walking_reference_scale
      )
      target = self._walking_reference_requested * self._last_gate.unsqueeze(-1)
      delta = (target - self._walking_reference_executed).clamp(
        -self._walking_reference_slew, self._walking_reference_slew
      )
      self._walking_reference_executed.add_(delta)
      self._walking_reference_executed[self._last_gate == 0.0] = 0.0
      self._walking_reference_active.copy_(
        self._walking_reference_executed.abs().amax(dim=1) > 1.0e-6
      )
    if self._print_every:
      self._print_pending = True

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids=env_ids)

    # A step may be in flight from the last apply_actions; drain it before the
    # I/O binding overwrites the input block or the pool sends reset commands
    # (the workers must be done reading it). Outputs for envs not being reset are
    # staged in `staged_control` and still applied at their next period start.
    self._collect_controller_output()

    if env_ids is None:
      env_ids = slice(None)

    self._torque_peak[env_ids] = 0.0
    self._last_gate[env_ids] = 0.0 if self._recovery_authority is not None else 1.0
    self._executed_physical[env_ids] = 0.0
    self._previous_executed_physical[env_ids] = 0.0
    self._residual_raw_actions[env_ids] = 0.0
    self._walking_reference_requested[env_ids] = 0.0
    self._walking_reference_executed[env_ids] = 0.0
    self._previous_walking_reference_executed[env_ids] = 0.0
    self._walking_reference_active[env_ids] = False
    self._datastore_scalar_active[env_ids] = False
    self._datastore_scalar_delta[env_ids] = 0.0
    self._projection_mask[env_ids] = False
    if self._recovery_authority is not None:
      self._recovery_authority.reset(env_ids)

    if isinstance(env_ids, slice):
      env_indices = list(range(self.num_envs))[env_ids]
    else:
      env_indices = env_ids.tolist()

    self._io.reset_controller_input(self._in_np)
    self._pool.reset_envs(env_indices)
    self._steps_since_run[env_indices] = 0

    # Seed interpolation (subclass-specific rest value per channel) and discard
    # any staged output for the reset envs; they restart from that seed.
    env_indices_t = torch.tensor(env_indices, device=self.device, dtype=torch.long)
    self._seed_interpolation(env_indices_t)
    self._has_staged_control[env_indices_t] = False
    for values in self._controller_vectors.values():
      values[env_indices_t] = 0.0
    for values in self._controller_scalars.values():
      values[env_indices_t] = 0.0
    for values in self._controller_scalar_baselines.values():
      values[env_indices_t] = 0.0
    # The pool has re-initialized these controllers, so clear the latches too.
    self.controller_failed[env_indices_t] = False
    self.controller_worker_failed[env_indices_t] = False
    self._out_np[env_indices, self._io.layout.status_off] = STATUS_OK

  def apply_actions(self) -> None:
    substep_in_period = self._steps_since_run % self.cfg.frameskip
    run_envs = substep_in_period == 0

    run_indices = run_envs.nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(run_indices, int):
      run_indices = [run_indices]

    if run_indices:
      # Collect the previous period's dispatch (it solved while the intervening
      # sim substeps ran) before reusing the shared I/O blocks.
      self._collect_controller_output()

      run_indices_t = torch.tensor(run_indices, device=self.device, dtype=torch.long)
      # Promote freshly collected outputs to `next` (previous<-next, next<-staged)
      # at each env's period start, keeping the ramp continuous one period
      # behind. Envs without a collected output yet (startup, just reset) hold
      # their seeded value.
      fresh = self._has_staged_control[run_indices_t]
      if bool(fresh.any()):
        fresh_indices_t = run_indices_t[fresh]
        for c in self.output_channels:
          self._previous_control[c][fresh_indices_t] = self._next_control[c][
            fresh_indices_t
          ]
          self._next_control[c][fresh_indices_t] = self._staged_control[c][
            fresh_indices_t
          ]
        self._has_staged_control[fresh_indices_t] = False

      # Sample the current state and dispatch this period's solve without
      # blocking; it overlaps the next `frameskip` substeps of sim.
      self._io.fill_controller_input(self._in_np)
      self._io.write_datastore_vector_commands(
        self._in_np,
        self._walking_reference_active,
        self._walking_reference_executed,
      )
      self._io.write_datastore_scalar_commands(
        self._in_np,
        self._datastore_scalar_active,
        self._datastore_scalar_delta,
      )
      self._pool.dispatch_controller_step(run_indices)

    # coef=1 on the last substep gives the full new target, matching mc_mujoco.
    interpolation_coef = (
      (substep_in_period + 1).float() / self.cfg.frameskip
    ).unsqueeze(-1)
    interpolated_control = {
      c: self._previous_control[c]
      + interpolation_coef * (self._next_control[c] - self._previous_control[c])
      for c in self.output_channels
    }

    # Peak-held across the decimation window: what sizes an actuator is the worst
    # instant, and apply_actions runs before sim.step so this trails by a substep.
    torch.maximum(
      self._torque_peak,
      self._entity.data.qfrc_actuator[:, self._target_ids].abs(),
      out=self._torque_peak,
    )

    self._steps_since_run += 1

    residual = self._processed_actions * self._last_gate.unsqueeze(-1)

    # Scatter a restricted residual into the full target width (non-matched
    # joints get 0, i.e. pure mc_rtc tracking).
    if self._residual_ids is not None:
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
