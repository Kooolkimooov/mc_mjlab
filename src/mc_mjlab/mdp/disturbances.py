"""Event terms that disturb the robot, and the record every recovery term reads."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp import events
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from mc_mjlab.mdp.sensors import residual_term

if TYPE_CHECKING:
  from collections.abc import Iterable

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.manager_base import ManagerTermBaseCfg


def randomize_current_pd_gains(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  scale_range: tuple[float, float],
  asset_cfg: SceneEntityCfg | None = None,
  action_name: str = "mc_rtc_residual",
) -> None:
  """Scale the active reference PD gains independently per environment and joint."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  else:
    env_ids = env_ids.to(env.device)

  term = residual_term(env, action_name)
  kp = getattr(term, "_kp", None)
  kd = getattr(term, "_kd", None)
  if kp is not None and kd is not None:
    scale = torch.empty(len(env_ids), kp.shape[1], device=env.device).uniform_(
      *scale_range
    )
    kp[env_ids] *= scale
    kd[env_ids] *= scale
    return

  asset = env.scene[(asset_cfg or SceneEntityCfg("robot")).name]
  for actuator in asset.actuators:
    stiffness = getattr(actuator, "stiffness", None)
    damping = getattr(actuator, "damping", None)
    if stiffness is None or damping is None:
      continue
    scale = torch.empty(
      len(env_ids), len(actuator.target_names), device=env.device
    ).uniform_(*scale_range)
    stiffness[env_ids] *= scale
    damping[env_ids] *= scale


class recorded_disturbance:
  """Shared per-environment record for disturbance-aware terms."""

  #: Monotone counter, so this reads as "no push yet" for any run length.
  NEVER = -(1 << 30)

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    del cfg
    self.last_push_step = torch.full(
      (env.num_envs,), self.NEVER, dtype=torch.long, device=env.device
    )
    self.last_push_vel = torch.zeros((env.num_envs, 3), device=env.device)
    self.enabled = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

  def disable(self, env_ids: torch.Tensor) -> None:
    """Suppress scheduled pushes for selected calibration environments."""
    self.enabled[env_ids] = False


class push_and_record(recorded_disturbance):
  """``push_by_setting_velocity``, plus a record of when it last fired."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg | None = None,
    warmup_s: float = 0.0,
    planar_speed: float | None = None,
  ) -> None:
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    ids = ids[self.enabled[ids]]
    if ids.numel() == 0:
      return

    if warmup_s > 0.0:
      # Suppress, do not reschedule: `EventManager` re-samples the countdown
      # whenever this fires. docs/difficulty.md#warmup_s
      ids = ids[env.episode_length_buf[ids] * env.step_dt >= warmup_s]
      if ids.numel() == 0:
        return

    asset = env.scene[(asset_cfg or SceneEntityCfg("robot")).name]
    # Mirrors `events.push_by_setting_velocity`, sampling here so the delta can be
    # recorded: `root_link_vel_w` comes from `cvel`, which MuJoCo does not
    # recompute until the next forward, so a before/after difference reads zero.
    vel_w = asset.data.root_link_vel_w[ids]
    if planar_speed is None:
      delta = events._sample_se3_range(velocity_range, vel_w.shape, str(env.device))
    else:
      angle = 2.0 * torch.pi * torch.rand(len(ids), device=env.device)
      delta = torch.zeros_like(vel_w)
      delta[:, 0] = planar_speed * torch.cos(angle)
      delta[:, 1] = planar_speed * torch.sin(angle)

    asset.write_root_link_velocity_to_sim(vel_w + delta, env_ids=ids)
    self.last_push_vel[ids] = quat_apply_inverse(
      asset.data.root_link_quat_w[ids], delta[:, :3]
    )
    self.last_push_step[ids] = env.common_step_counter


class finite_impulse_curriculum(recorded_disturbance):
  """Apply finite, mass-scaled torso impulses with a global-step curriculum."""

  constructor_parameters = frozenset({"interval_range_s", "warmup_s", "asset_cfg"})

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self._env = env
    self.asset = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
    self.interval_range_s = cfg.params["interval_range_s"]
    self.warmup_s = cfg.params["warmup_s"]
    # `play` takes no `--env.*` overrides, and a finite wrench draws nothing, so
    # this is the only way to see a push. docs/difficulty.md#MC_MJLAB_PUSH_DEBUG
    self._debug_scale = float(os.environ.get("MC_MJLAB_PUSH_DEBUG", "0.0") or 0.0)
    if self._debug_scale > 0.0:
      self.warmup_s = min(self.warmup_s, 1.0)

    self.force = torch.zeros(env.num_envs, 1, 3, device=env.device)
    self.torque = torch.zeros_like(self.force)
    self.remaining = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.next_push_step = torch.zeros_like(self.remaining)

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Clear active wrenches and schedule the first post-warmup impulse."""
    env = self._env
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    self._write_zeros(ids)
    self.last_push_step[ids] = self.NEVER
    self.last_push_vel[ids] = 0.0

    warmup = round(self.warmup_s / env.step_dt)
    span = max(
      1, round((self.interval_range_s[1] - self.interval_range_s[0]) / env.step_dt)
    )
    self.next_push_step[ids] = warmup + torch.randint(
      0, span + 1, (len(ids),), device=env.device
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    duration_range_s: tuple[float, float],
    height_range_m: tuple[float, float],
    stages: tuple[tuple[int, tuple[float, float]], ...],
    enabled: bool = True,
    **configuration: object,
  ) -> None:
    """Expire the current wrench and trigger any due curriculum impulse."""
    del env_ids
    unknown = configuration.keys() - self.constructor_parameters
    if unknown:
      raise TypeError(f"unknown impulse parameters: {sorted(unknown)}")

    active = self.remaining > 0
    self.remaining[active] -= 1
    expired = active & (self.remaining == 0)
    if bool(expired.any()):
      self._write_zeros(expired.nonzero(as_tuple=False).flatten())

    if not enabled:
      return

    due = self._due(env)
    ids = due.nonzero(as_tuple=False).flatten()
    if ids.numel() == 0:
      return

    self._trigger(env, ids, duration_range_s, height_range_m, stages)

    low, high = self.interval_range_s
    low_steps = round(low / env.step_dt)
    high_steps = round(high / env.step_dt)
    self.next_push_step[ids] = env.episode_length_buf[ids] + torch.randint(
      low_steps, high_steps + 1, (len(ids),), device=env.device
    )

  def _due(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return environments whose next scheduled impulse has arrived."""
    return self.enabled & (env.episode_length_buf >= self.next_push_step)

  def _trigger(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    duration_range_s: tuple[float, float],
    height_range_m: tuple[float, float],
    stages: tuple[tuple[int, tuple[float, float]], ...],
  ) -> None:
    """Sample and install one force-equivalent planar velocity change."""
    velocity_range = stages[0][1]
    for step, candidate in stages:
      if env.common_step_counter >= step:
        velocity_range = candidate

    count = len(env_ids)
    angle = 2.0 * torch.pi * torch.rand(count, device=env.device)
    speed = velocity_range[0] + (velocity_range[1] - velocity_range[0]) * torch.rand(
      count, device=env.device
    )
    if self._debug_scale > 0.0:
      speed = speed * self._debug_scale
      for row, env_id in enumerate(env_ids.tolist()):
        print(
          f"[push] env {env_id} {float(speed[row]):.3f} m/s at "
          f"{math.degrees(float(angle[row])):.0f} deg",
          flush=True,
        )

    delta_b = torch.zeros(count, 3, device=env.device)
    delta_b[:, 0] = speed * torch.cos(angle)
    delta_b[:, 1] = speed * torch.sin(angle)
    quat = self.asset.data.root_link_quat_w[env_ids]
    delta_w = quat_apply(quat, delta_b)

    min_steps = math.ceil(duration_range_s[0] / env.step_dt)
    max_steps = math.floor(duration_range_s[1] / env.step_dt)
    duration_steps = torch.randint(
      min_steps, max_steps + 1, (count,), device=env.device
    )
    duration = duration_steps * env.step_dt

    body_ids = self.asset.indexing.body_ids
    mass = env.sim.model.body_mass[env_ids][:, body_ids].sum(dim=1)
    force = mass.unsqueeze(-1) * delta_w / duration.unsqueeze(-1)
    height = height_range_m[0] + (height_range_m[1] - height_range_m[0]) * torch.rand(
      count, device=env.device
    )
    offset_b = torch.zeros_like(force)
    offset_b[:, 2] = height
    torque = torch.cross(quat_apply(quat, offset_b), force, dim=1)

    self.force[env_ids, 0] = force
    self.torque[env_ids, 0] = torque
    self.remaining[env_ids] = duration_steps
    self.asset.write_external_wrench_to_sim(
      self.force[env_ids], self.torque[env_ids], env_ids=env_ids, body_ids=[0]
    )
    self.last_push_vel[env_ids] = delta_b
    self.last_push_step[env_ids] = env.common_step_counter

  def _write_zeros(self, env_ids: torch.Tensor) -> None:
    """Remove external wrenches for selected environments."""
    if env_ids.numel() == 0:
      return

    zeros = torch.zeros(len(env_ids), 1, 3, device=env_ids.device)
    self.asset.write_external_wrench_to_sim(zeros, zeros, env_ids=env_ids, body_ids=[0])
    self.force[env_ids] = 0.0
    self.torque[env_ids] = 0.0
    self.remaining[env_ids] = 0


class stratified_finite_impulse_curriculum(finite_impulse_curriculum):
  """Draw each reset cohort from a stationary standing-plus-band mixture."""

  constructor_parameters = finite_impulse_curriculum.constructor_parameters | {
    "bands",
    "band_weights",
  }

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.bands = tuple(tuple(band) for band in cfg.params["bands"])
    weights = self.validated_weights(cfg.params["band_weights"])
    union = (min(low for low, _ in self.bands), max(high for _, high in self.bands))
    stages = cfg.params["stages"]

    # `stages` is inert here but reaches the manifest, so keep it honest.
    if len(stages) != 1 or tuple(stages[0][1]) != union:
      raise ValueError(f"stages must record the single band union {union}")

    self.band_weights = weights
    self._band_weights = torch.tensor(weights, device=env.device)
    self.sampled_band = torch.full(
      (env.num_envs,), -1, dtype=torch.long, device=env.device
    )

  def validated_weights(self, weights: Iterable[float]) -> tuple[float, ...]:
    """Return one reset-time mixture, rejecting a malformed one."""
    values = tuple(float(value) for value in weights)
    if len(values) != len(self.bands) + 1:
      raise ValueError("band weights need standing plus every impulse band")
    if abs(sum(values) - 1.0) > 1e-9 or any(value < 0.0 for value in values):
      raise ValueError("band weights must be nonnegative and sum to one")

    return values

  def set_band_weights(self, weights: Iterable[float]) -> None:
    """Replace the mixture drawn by cohorts resetting from now on."""
    values = self.validated_weights(weights)
    self.band_weights = values
    self._band_weights = torch.tensor(values, device=self._env.device)

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Schedule an impulse and draw the reset cohort's magnitude band."""
    super().reset(env_ids)
    ids = (
      torch.arange(self._env.num_envs, device=self._env.device)
      if env_ids is None
      else env_ids
    )
    if ids.numel() == 0:
      return

    self.sampled_band[ids] = (
      torch.multinomial(self._band_weights, len(ids), replacement=True) - 1
    )

  def _due(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Exclude the standing cohort from scheduled disturbances."""
    return super()._due(env) & (self.sampled_band >= 0)

  def _trigger(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    duration_range_s: tuple[float, float],
    height_range_m: tuple[float, float],
    stages: tuple[tuple[int, tuple[float, float]], ...],
  ) -> None:
    """Draw each due environment inside the band chosen at its last reset."""
    del stages
    for band, values in enumerate(self.bands):
      ids = env_ids[self.sampled_band[env_ids] == band]
      if ids.numel():
        super()._trigger(env, ids, duration_range_s, height_range_m, ((0, values),))


class achievement_finite_impulse_curriculum(finite_impulse_curriculum):
  """Apply checkpointed difficulty with standing and prior-stage rehearsal."""

  constructor_parameters = finite_impulse_curriculum.constructor_parameters | {
    "rehearsal_weights",
    "initial_stage",
  }

  is_achievement_curriculum = True

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.rehearsal_weights = cfg.params["rehearsal_weights"]
    self.current_stage = int(cfg.params.get("initial_stage", 0))
    self.sampled_stage = torch.full(
      (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
    self.set_stage(self.current_stage)

  def set_stage(self, stage: int) -> None:
    """Select the mixture used by environments at their next reset."""
    if not 0 <= stage < len(self.rehearsal_weights):
      raise ValueError(f"invalid achievement stage {stage}")
    weights = self.rehearsal_weights[stage]
    if len(weights) != len(self.rehearsal_weights) + 1:
      raise ValueError("rehearsal weights need standing plus every physical stage")
    if abs(sum(weights) - 1.0) > 1e-9 or any(value < 0.0 for value in weights):
      raise ValueError("rehearsal weights must be nonnegative and sum to one")
    if any(weights[stage + 2 :]):
      raise ValueError("rehearsal mixture cannot sample a future stage")

    self.current_stage = stage

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Schedule an impulse and sample the reset cohort's rehearsal level."""
    super().reset(env_ids)
    ids = (
      torch.arange(self._env.num_envs, device=self._env.device)
      if env_ids is None
      else env_ids
    )
    if ids.numel() == 0:
      return

    weights = torch.tensor(
      self.rehearsal_weights[self.current_stage], device=self._env.device
    )
    self.sampled_stage[ids] = torch.multinomial(weights, len(ids), replacement=True) - 1

  def curriculum_state(self) -> dict[str, float | int]:
    """Expose current target, standing share, and earlier-stage rehearsal."""
    weights = self.rehearsal_weights[self.current_stage]
    return {
      "stage": self.current_stage,
      "standing_share": weights[0],
      "earlier_stage_share": sum(weights[1 : self.current_stage + 1]),
      "target_stage_share": weights[self.current_stage + 1],
    }

  def _due(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Exclude the standing cohort from scheduled disturbances."""
    return super()._due(env) & (self.sampled_stage >= 0)

  def _trigger(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    duration_range_s: tuple[float, float],
    height_range_m: tuple[float, float],
    stages: tuple[tuple[int, tuple[float, float]], ...],
  ) -> None:
    """Sample each due environment from its reset-time rehearsal stage."""
    for stage in range(self.current_stage + 1):
      ids = env_ids[self.sampled_stage[env_ids] == stage]
      if ids.numel():
        super()._trigger(
          env,
          ids,
          duration_range_s,
          height_range_m,
          ((0, stages[stage][1]),),
        )


#: Age reported for an env that has not been pushed inside its current episode.
NEVER_AGE = 1 << 30


def record_disturbance(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  equivalent_velocity_b: torch.Tensor,
  term_name: str = "push_robot",
) -> None:
  """Record a deterministic external disturbance for recovery terms."""
  term = push_term(env, term_name)
  term.last_push_vel[env_ids] = equivalent_velocity_b
  term.last_push_step[env_ids] = env.common_step_counter


def push_term(env: ManagerBasedRlEnv, term_name: str) -> recorded_disturbance:
  """The recorded disturbance behind ``term_name``, or a ``TypeError``."""
  term = env.event_manager.get_term_cfg(term_name).func
  if not isinstance(term, recorded_disturbance):
    raise TypeError(
      f"event term {term_name!r} must record disturbances for a "
      f"disturbance-gated reward to know when it fired, got {type(term).__name__}"
    )
  return term


def age_since_push(env: ManagerBasedRlEnv, term: recorded_disturbance) -> torch.Tensor:
  """See :func:`steps_since_push`; this is that, with the term already resolved."""
  # Python int on the left: `torch.as_tensor` here would be an H2D copy per step.
  age = env.common_step_counter - term.last_push_step
  return age.masked_fill(age >= env.episode_length_buf, NEVER_AGE)
