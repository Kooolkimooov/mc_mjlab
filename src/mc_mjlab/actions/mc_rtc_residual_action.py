"""Reusable base for residual action terms backed by per-env mc_rtc controllers.

Owns the control law: per-substep interpolation of the controller outputs, the
one-period-behind dispatch sequencing that overlaps the workers' solve with the
GPU sim, and the RL residual plumbing. It leans on two collaborators: the
transport (``mc_rtc_controller_pool.ControllerPool`` -- processes, pipes, shared
blocks) and the sim I/O wiring (``mc_rtc_controller_io_binding.ControllerIoBinding`` -- model
introspection, ``IoLayout`` and per-step input assembly).

Subclasses supply only what differs between residual types: which controller
output channels to consume (``output_channels``, matching what the host writes),
how to seed those channels at reset (``_seed_interpolation``) and how to map the
interpolated outputs plus residual onto the entity's actuator targets
(``_apply_control``). See ``mc_rtc_residual_joint_position_actions`` for the
position/velocity subclass.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.utils.lab_api.string import resolve_matching_names

from mc_mjlab.actions.mc_rtc_controller_io_binding import (
  ControllerIoBinding,
  apply_reference_pd_gains,
)
from mc_mjlab.actions.mc_rtc_controller_pool import ControllerPool

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class McRtcResidualActionCfg(BaseActionCfg):
  """Shared configuration for mc_rtc residual action terms.

  Abstract: subclasses add a ``build`` returning their concrete action term.
  """

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

  use_controller_reset: bool = True
  """Reset via ``MCGlobalController.reset()`` (mc_mujoco parity). Requires the
  locally patched mc_rtc (stock fsm::Controller segfaults on destruction, see
  the GUI/StateBuilder fix). When False, resets re-run ``init()``, which raises
  for plugins that register datastore entries."""


class McRtcResidualActionBase(BaseAction):
  """mc_rtc residual action base: steps controllers via a pool, adds RL residual.

  Concrete subclasses set ``output_channels`` and implement ``_seed_interpolation``
  and ``_apply_control``.
  """

  cfg: McRtcResidualActionCfg

  output_channels: tuple[str, ...] = ()
  """Controller output channels consumed, in output-block order (must match what
  the host writes to ``out_np``). Set by the subclass."""

  def __init__(self, cfg: McRtcResidualActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    self.cfg = cfg
    self._mc_rtc_robot_name = cfg.mc_rtc_robot_name
    self._num_targets = len(self._target_names)

    self._setup_residual(cfg)

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
    )
    metadata = self._pool.await_ready()

    # Sim <-> mc_rtc input wiring (model introspection, IoLayout, per-step fill).
    self._io = ControllerIoBinding(
      self._env,
      self._entity,
      self._target_names,
      self._target_ids,
      metadata,
      cfg.use_controller_reset,
      self.output_channels,
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

  # ---- Construction helpers. ----

  def _setup_residual(self, cfg: McRtcResidualActionCfg) -> None:
    """Slice scale/offset/clip down to the residual actuator subset.

    The base class resolved them against the full target list; when only a
    subset receives the residual, the columns must match.
    """
    self._residual_ids: torch.Tensor | None = None
    if cfg.residual_actuator_names is None:
      return
    ids, _ = resolve_matching_names(cfg.residual_actuator_names, self._target_names)
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

  def _alloc_interpolation_buffers(self) -> None:
    """Per-channel ramp endpoints plus the one-period-behind staging buffer.

    Each channel is sized to the full controlled-joint set (not the residual
    subset) and laid out in the output block in ``output_channels`` order -- the
    same order the host writes.
    """
    out_width = self._io.layout.out_width
    assert out_width == len(self.output_channels) * self._num_targets, (
      f"output block width {out_width} does not match "
      f"{len(self.output_channels)} channel(s) x {self._num_targets} targets"
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

  # ---- Pipeline. ----

  def _collect_controller_output(self) -> None:
    """Await the outstanding async step (if any) and stage its outputs.

    Results move into the ``staged_control`` buffers; each env promotes them to
    ``next`` at its own next period start, so the ramp stays period-aligned.
    A no-op when nothing is in flight.
    """
    env_indices = self._pool.collect()
    if env_indices is None:
      return
    new_output = self._io.read_controller_output(self._out_np, env_indices)
    env_indices_t = torch.tensor(env_indices, device=self.device, dtype=torch.long)
    for c in self.output_channels:
      self._staged_control[c][env_indices_t] = new_output[c]
    self._has_staged_control[env_indices_t] = True

  # ---- Subclass hooks. ----

  @abc.abstractmethod
  def _seed_interpolation(self, env_ids: torch.Tensor) -> None:
    """Seed the interpolation endpoints for the given (reset) envs.

    Called from ``reset`` after the controller has been reset. Set
    ``_previous_control[c]``/``_next_control[c]`` for each channel at ``env_ids`` to a sensible rest
    value so the first control period does not ramp from zero.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def _apply_control(
    self, interpolated_control: dict[str, torch.Tensor], residual: torch.Tensor
  ) -> None:
    """Write actuator targets from the interpolated controller outputs.

    ``interpolated_control`` maps each channel in ``output_channels`` to its
    per-substep interpolated value (num_envs x num_targets). ``residual`` is the
    processed RL residual, scattered to full target width.
    """
    raise NotImplementedError

  # ---- ActionTerm API. ----

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids=env_ids)

    # A step may be in flight from the last apply_actions; drain it before the
    # I/O binding overwrites the input block or the pool sends reset commands
    # (the workers must be done reading it). Outputs for envs not being reset are
    # staged in `staged_control` and still applied at their next period start.
    self._collect_controller_output()

    if env_ids is None:
      env_ids = slice(None)

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

    self._steps_since_run += 1

    # Scatter a restricted residual into the full target width (non-matched
    # joints get 0, i.e. pure mc_rtc tracking).
    residual = self._processed_actions
    if self._residual_ids is not None:
      self._residual_full.zero_()
      self._residual_full[:, self._residual_ids] = residual
      residual = self._residual_full

    self._apply_control(interpolated_control, residual)
