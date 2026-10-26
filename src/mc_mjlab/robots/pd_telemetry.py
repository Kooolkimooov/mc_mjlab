"""Optional exact pre-clamp PD telemetry for diagnostic environments."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.actuator.actuator import ActuatorCfg, ActuatorCmd, TransmissionType
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg
from mjlab.utils.lab_api.string import resolve_matching_names
from mjlab.utils.spec import create_motor_actuator

if TYPE_CHECKING:
  import mujoco
  import mujoco_warp as mjwarp
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnvCfg

#: Channels of one stacked sample, in the order of its leading dimension.
FIELDS = (
  "position_target_rad",
  "velocity_target_rad_s",
  "position_rad",
  "velocity_rad_s",
  "p_nm",
  "d_nm",
  "feedforward_nm",
  "raw_nm",
  "clipped_nm",
  "time_s",
  "kp",
  "kd",
  "limit_nm",
)
_P, _D, _RAW, _CLIPPED, _LIMIT = (
  FIELDS.index(name) for name in ("p_nm", "d_nm", "raw_nm", "clipped_nm", "limit_nm")
)
_DELAY_FIELDS = (
  "delay_min_lag",
  "delay_max_lag",
  "delay_hold_prob",
  "delay_update_period",
  "delay_per_env_phase",
)


class PdDemand:
  """Fold integrated PD samples into policy-window extrema and saturation runs."""

  def __init__(self, template: torch.Tensor, dt: float) -> None:
    if dt <= 0:
      raise ValueError("positive physics timestep required")
    self.dt = dt
    # Durations are substep counts: float32 seconds would drift over an episode.
    self.steps = torch.zeros_like(template)
    self.run = torch.zeros_like(template)
    self.peak = torch.zeros_like(template)
    self.p_max = torch.zeros_like(template)
    self.d_max = torch.zeros_like(template)
    self.saturated = torch.zeros_like(template)
    self.longest = torch.zeros_like(template)
    self.samples = torch.zeros_like(template)
    self.at_peak = template.new_zeros(len(FIELDS), *template.shape)
    self.last = torch.zeros_like(self.at_peak)
    self.terminal: dict[str, torch.Tensor] = {}
    self.rate_rows: torch.Tensor | None = None
    self.rate: list[torch.Tensor] = []

  def evaluate(
    self, cmd: ActuatorCmd, kp: torch.Tensor, kd: torch.Tensor, limit: torch.Tensor
  ) -> torch.Tensor:
    """Stack one sample in ``pd_torque``'s summation order, clipped last."""
    p = kp * (cmd.position_target - cmd.pos)
    d = kd * (cmd.velocity_target - cmd.vel)
    raw = p + d
    raw += cmd.effort_target
    clipped = torch.clamp(raw, -limit, limit)
    values = (
      cmd.position_target,
      cmd.velocity_target,
      cmd.pos,
      cmd.vel,
      p,
      d,
      cmd.effort_target,
      raw,
      clipped,
      self.steps * self.dt,
      kp,
      kd,
      limit,
    )
    return torch.stack(torch.broadcast_tensors(*values))

  def record(self, sample: torch.Tensor) -> None:
    """Fold one sample the simulator integrated; reset-time writes never arrive here."""
    raw = sample[_RAW].abs()
    first = (raw > self.peak) | (self.samples == 0)
    self.at_peak = torch.where(first, sample, self.at_peak)
    self.peak = torch.maximum(self.peak, raw)
    self.p_max = torch.maximum(self.p_max, sample[_P].abs())
    self.d_max = torch.maximum(self.d_max, sample[_D].abs())
    saturated = raw > sample[_LIMIT]
    self.saturated += saturated
    self.run = torch.where(saturated, self.run + 1, 0.0)
    self.longest = torch.maximum(self.longest, self.run)
    self.steps += 1
    self.samples += 1
    self.last = sample
    if self.rate_rows is not None:
      self.rate.append(sample[:, self.rate_rows])

  def begin_interval(self, ids: torch.Tensor | slice = slice(None)) -> None:
    """Open a window; a saturation run continuing across its boundary keeps counting."""
    for value in (
      self.peak,
      self.p_max,
      self.d_max,
      self.saturated,
      self.longest,
      self.samples,
    ):
      value[ids] = 0
    self.at_peak[:, ids] = 0

  def reset(self, ids: torch.Tensor | slice = slice(None)) -> None:
    """Keep the selected rows' terminal window, then clear only those rows."""
    for key, value in self.snapshot().items():
      self.terminal.setdefault(key, torch.zeros_like(value))[ids] = value[ids]
    self.begin_interval(ids)
    self.steps[ids] = 0
    self.run[ids] = 0
    self.last[:, ids] = 0

  def snapshot(self) -> dict[str, torch.Tensor]:
    """Clone the current window so later substeps cannot mutate it."""
    values = {
      "peak_abs_nm": self.peak,
      "independent_p_peak_nm": self.p_max,
      "independent_d_peak_nm": self.d_max,
      "saturated_s": self.saturated * self.dt,
      "longest_continuous_s": self.longest * self.dt,
      "samples": self.samples,
    }
    for index, name in enumerate(FIELDS):
      values[f"at_total_peak/{name}"] = self.at_peak[index]
      values[f"last/{name}"] = self.last[index]
    return {key: value.clone() for key, value in values.items()}

  def drain_rate(self) -> torch.Tensor | None:
    """``(substeps, fields, rows, joints)`` for the traced rows since the last drain."""
    if not self.rate:
      return None
    trace = torch.stack(self.rate)
    self.rate.clear()
    return trace


@dataclass(kw_only=True)
class InstrumentedPdActuatorCfg(IdealPdActuatorCfg):
  """One diagnostic actuator standing in for ideal PD parts; never used in training."""

  parts: tuple[IdealPdActuatorCfg, ...]
  physics_dt: float
  policy_substeps: int
  # Per-target values come from `parts`; the scalars are unused.
  stiffness: float = math.nan
  damping: float = math.nan

  def build(
    self, entity: Entity, target_ids: list[int], target_names: list[str]
  ) -> InstrumentedPdActuator:
    return InstrumentedPdActuator(self, entity, target_ids, target_names)


class InstrumentedPdActuator(IdealPdActuator[InstrumentedPdActuatorCfg]):
  """Apply the ideal PD law and keep the delayed, pre-clamp demand behind it."""

  def __init__(
    self,
    cfg: InstrumentedPdActuatorCfg,
    entity: Entity,
    target_ids: list[int],
    target_names: list[str],
  ) -> None:
    super().__init__(cfg, entity, target_ids, target_names)
    self._parts, self._creation_order = _assign_parts(cfg.parts, target_names)
    self._pending: torch.Tensor | None = None
    self._substep = 0
    self.telemetry: PdDemand | None = None

  def edit_spec(self, spec: mujoco.MjSpec, target_names: list[str]) -> None:
    if len(target_names) != len(self._parts):
      raise NotImplementedError("PD telemetry does not support sort_actuators")

    # Motor ids follow the parts' order; ctrl columns follow the targets'.
    motors = {}
    for index in self._creation_order:
      part = self._parts[index]
      motors[index] = create_motor_actuator(
        spec,
        target_names[index],
        effort_limit=part.effort_limit,
        armature=part.armature,
        frictionloss=part.frictionloss,
        viscous_damping=part.viscous_damping,
        transmission_type=part.transmission_type,
      )
    self._mjs_actuators.extend(motors[index] for index in range(len(target_names)))

  def initialize(
    self,
    mj_model: mujoco.MjModel,
    model: mjwarp.Model,
    data: mjwarp.Data,
    device: str,
  ) -> None:
    super().initialize(mj_model, model, data, device)

    def per_target(attribute: str) -> torch.Tensor:
      values = [getattr(part, attribute) for part in self._parts]
      row = torch.tensor(values, dtype=torch.float, device=device)
      return row.expand(data.nworld, -1).clone()

    self.stiffness = per_target("stiffness")
    self.damping = per_target("damping")
    self.force_limit = per_target("effort_limit")
    self.default_stiffness = self.stiffness.clone()
    self.default_damping = self.damping.clone()
    self.default_force_limit = self.force_limit.clone()
    self.telemetry = PdDemand(self.stiffness, self.cfg.physics_dt)

  def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
    assert self.telemetry is not None and self.stiffness is not None
    assert self.damping is not None and self.force_limit is not None
    self._pending = self.telemetry.evaluate(
      cmd, self.stiffness, self.damping, self.force_limit
    )
    return self._pending[_CLIPPED]

  def update(self, dt: float) -> None:
    # Runs only after sim.step(), so a reset's extra write_data_to_sim never counts.
    del dt
    if self.telemetry is None or self._pending is None:
      return

    if self._substep == 0:
      self.telemetry.begin_interval()
    self.telemetry.record(self._pending)
    self._pending = None
    self._substep = (self._substep + 1) % self.cfg.policy_substeps

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids)
    if self.telemetry is not None:
      self.telemetry.reset(slice(None) if env_ids is None else env_ids)


def instrument_pd(cfg: ManagerBasedRlEnvCfg, entity: str = "robot") -> None:
  """Replace an entity's exact ideal PD configs with one telemetry actuator."""
  articulation = cfg.scene.entities[entity].articulation
  if articulation is None or not articulation.actuators:
    raise ValueError(f"{entity} has no actuators to instrument")
  articulation.actuators = (
    instrumented_pd_cfg(
      articulation.actuators, cfg.sim.mujoco.timestep, cfg.decimation
    ),
  )


def instrumented_pd_cfg(
  parts: Sequence[ActuatorCfg], physics_dt: float, policy_substeps: int
) -> InstrumentedPdActuatorCfg:
  """The telemetry actuator equivalent to exact joint ``IdealPdActuatorCfg`` parts."""
  pd = tuple(
    part
    for part in parts
    if type(part) is IdealPdActuatorCfg
    and part.transmission_type == TransmissionType.JOINT
  )
  if len(pd) != len(parts):
    raise TypeError("PD telemetry requires exact joint IdealPdActuatorCfg parts")
  delays = {tuple(getattr(part, name) for name in _DELAY_FIELDS) for part in pd}
  if len(delays) != 1:
    raise ValueError("PD telemetry requires one shared actuator delay configuration")

  return InstrumentedPdActuatorCfg(
    target_names_expr=tuple(e for part in pd for e in part.target_names_expr),
    parts=pd,
    physics_dt=physics_dt,
    policy_substeps=policy_substeps,
    **{name: getattr(pd[0], name) for name in _DELAY_FIELDS},
  )


def _assign_parts(
  parts: tuple[IdealPdActuatorCfg, ...], target_names: list[str]
) -> tuple[list[IdealPdActuatorCfg], list[int]]:
  """Each target's part, and target indices in the order the parts create motors."""
  owners: dict[int, IdealPdActuatorCfg] = {}
  order: list[int] = []
  for part in parts:
    indices, _ = resolve_matching_names(part.target_names_expr, target_names)
    for index in indices:
      if index in owners:
        raise ValueError(f"{target_names[index]} is claimed by two PD configs")
      owners[index] = part
    order.extend(indices)
  return [owners[index] for index in range(len(target_names))], order
