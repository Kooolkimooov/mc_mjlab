"""Aggregate qualification diagnostics by behavior regime and disturbance direction."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from mc_mjlab import mdp
from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase
from mc_mjlab.robots import robot_module as mc_rtc

STRATUM_LABELS = (
  ("startup", "none", "none"),
  ("sustained", "none", "none"),
  ("recovery", "sagittal", "forward"),
  ("recovery", "sagittal", "backward"),
  ("recovery", "lateral", "left"),
  ("recovery", "lateral", "right"),
)
SCALAR_METRICS = (
  "com_velocity_error",
  "dcm_error",
  "zmp_error",
  "zmp_grounded",
  "foot_slip",
  "gate_mean",
  "detector_score",
  "max_effort_ratio",
)
JOINT_SUM_FIELDS = (
  "requested_normalized_square",
  "executed_normalized_square",
  "executed_physical_square",
  "projection",
  "near_bound",
  "active",
  "effort_ratio",
)


@dataclass
class StratumRecord:
  """One episode's aggregate within one behavioral stratum."""

  checkpoint: str
  scenario: str
  seed: int
  env_id: int
  pair: int
  arm: str
  regime: str
  axis: str
  direction: str
  steps: int
  duration_s: float
  terminations: dict[str, int]
  metrics: dict[str, float]
  joints: dict[str, dict[str, float]]


def classify_strata(
  episode_steps: torch.Tensor,
  push_age: torch.Tensor,
  push_velocity_b: torch.Tensor,
  warmup_steps: int,
  recovery_steps: int,
) -> torch.Tensor:
  """Classify each environment as startup, sustained, or directional recovery."""
  strata = torch.where(
    episode_steps <= warmup_steps,
    torch.zeros_like(episode_steps),
    torch.ones_like(episode_steps),
  )
  recovery = (push_age >= 1) & (push_age <= recovery_steps)
  sagittal = push_velocity_b[:, 0].abs() >= push_velocity_b[:, 1].abs()
  directions = torch.where(
    sagittal,
    torch.where(push_velocity_b[:, 0] >= 0.0, 2, 3),
    torch.where(push_velocity_b[:, 1] >= 0.0, 4, 5),
  )
  strata[recovery] = directions[recovery]
  return strata


def _safe_ratio(numerator: float, denominator: float) -> float:
  """Divide conditional sums without manufacturing a zero-sample value."""
  return numerator / denominator if denominator > 0.0 else float("nan")


class StratifiedDiagnostics:
  """Accumulate GPU-side regime and per-joint diagnostics until episode end."""

  def __init__(
    self,
    env: Any,
    action: McRtcResidualActionBase,
    warmup_s: float,
    recovery_s: float,
  ) -> None:
    self.env = env
    self.action = action
    self.step_dt = float(env.step_dt)
    self.warmup_steps = round(warmup_s / self.step_dt)
    self.recovery_steps = round(recovery_s / self.step_dt)
    names = env.metrics_manager.active_terms
    missing = set(SCALAR_METRICS) - set(names)
    if missing:
      raise KeyError(f"stratified qualification needs metrics {sorted(missing)}")
    self._metric_indices = torch.tensor(
      [names.index(name) for name in SCALAR_METRICS],
      dtype=torch.long,
      device=env.device,
    )
    shape = (env.num_envs, len(STRATUM_LABELS))
    joint_shape = (*shape, len(action.residual_names))
    self._counts = torch.zeros(shape, dtype=torch.long, device=env.device)
    self._scalar_sums = torch.zeros(*shape, len(SCALAR_METRICS), device=env.device)
    self._scalar_max = torch.zeros_like(self._scalar_sums)
    self._joint_sums = torch.zeros(
      *joint_shape, len(JOINT_SUM_FIELDS), device=env.device
    )
    self._joint_effort_max = torch.zeros(joint_shape, device=env.device)
    limits = mc_rtc.get_effort_limits(action.cfg.mc_rtc_robot_name)
    self._effort_limits = torch.tensor(
      [limits[name] for name in action.residual_names], device=env.device
    )
    self._last_stratum = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

  def _residual_effort_ratio(self) -> torch.Tensor:
    """Actuator effort over the RobotModule limit, per residual joint."""
    term = self.action
    effort = self.env.scene[term.cfg.entity_name].data.qfrc_actuator[:, term.target_ids]
    if term.residual_ids is not None:
      effort = effort[:, term.residual_ids]
    return effort.abs() / self._effort_limits

  def capture(self, active: torch.Tensor) -> None:
    """Accumulate the just-computed step for all still-qualified environments."""
    env_ids = active.nonzero(as_tuple=False).flatten()
    if env_ids.numel() == 0:
      return
    strata = classify_strata(
      self.env.episode_length_buf,
      mdp.observations.steps_since_push(self.env),
      mdp.observations.last_push_velocity(self.env),
      self.warmup_steps,
      self.recovery_steps,
    )
    columns = strata[env_ids]
    self._last_stratum[env_ids] = columns
    self._counts[env_ids, columns] += 1
    values = self.env.metrics_manager._step_values[:, self._metric_indices]
    self._scalar_sums[env_ids, columns] += values[env_ids]
    self._scalar_max[env_ids, columns] = torch.maximum(
      self._scalar_max[env_ids, columns], values[env_ids]
    )
    requested = self.action.requested_normalized_action
    executed_normalized = self.action.executed_normalized_action
    executed_physical = self.action.executed_physical_action
    effort = self._residual_effort_ratio()
    joint_values = torch.stack(
      (
        requested.square(),
        executed_normalized.square(),
        executed_physical.square(),
        self.action.projection_mask.float(),
        (requested.abs() >= 0.99).float(),
        (executed_physical.abs() > 1.0e-9).float(),
        effort,
      ),
      dim=-1,
    )
    self._joint_sums[env_ids, columns] += joint_values[env_ids]
    self._joint_effort_max[env_ids, columns] = torch.maximum(
      self._joint_effort_max[env_ids, columns], effort[env_ids]
    )

  def finish(
    self,
    checkpoint: str,
    scenario: str,
    seed: int,
    env_ids: torch.Tensor,
    pairs: Sequence[int],
    policy_mask: torch.Tensor,
    terminations: Mapping[str, torch.Tensor],
  ) -> list[StratumRecord]:
    """Materialize and clear completed environments' per-stratum aggregates."""
    records: list[StratumRecord] = []
    for env_id in env_ids.tolist():
      counts = self._counts[env_id].cpu().tolist()
      scalar_sums = self._scalar_sums[env_id].cpu()
      scalar_max = self._scalar_max[env_id].cpu()
      joint_sums = self._joint_sums[env_id].cpu()
      joint_effort_max = self._joint_effort_max[env_id].cpu()
      last_stratum = int(self._last_stratum[env_id].item())
      scale = self.action.residual_scale[env_id].detach().cpu().tolist()
      for stratum, count in enumerate(counts):
        if count == 0:
          continue
        regime, axis, direction = STRATUM_LABELS[stratum]
        sums = {
          name: float(scalar_sums[stratum, index].item())
          for index, name in enumerate(SCALAR_METRICS)
        }
        maxima = {
          name: float(scalar_max[stratum, index].item())
          for index, name in enumerate(SCALAR_METRICS)
        }
        grounded = sums["zmp_grounded"]
        metrics = {
          "grounded_fraction": grounded / count,
          "dcm_error_grounded": _safe_ratio(sums["dcm_error"], grounded),
          "zmp_error_grounded": _safe_ratio(sums["zmp_error"], grounded),
          "com_velocity_error": sums["com_velocity_error"] / count,
          "foot_slip": sums["foot_slip"] / count,
          "gate_duty": sums["gate_mean"] / count,
          "detector_score": sums["detector_score"] / count,
          "max_effort_ratio": maxima["max_effort_ratio"],
        }
        joints = {}
        for joint_index, joint_name in enumerate(self.action.residual_names):
          values = joint_sums[stratum, joint_index]
          fields = {
            name: float(values[index].item())
            for index, name in enumerate(JOINT_SUM_FIELDS)
          }
          joints[joint_name] = {
            "authority_scale": float(scale[joint_index]),
            "requested_normalized_rms": math.sqrt(
              fields["requested_normalized_square"] / count
            ),
            "executed_normalized_rms": math.sqrt(
              fields["executed_normalized_square"] / count
            ),
            "executed_physical_rms": math.sqrt(
              fields["executed_physical_square"] / count
            ),
            "projection_fraction": fields["projection"] / count,
            "near_bound_fraction": fields["near_bound"] / count,
            "active_fraction": fields["active"] / count,
            "effort_ratio_mean": fields["effort_ratio"] / count,
            "effort_ratio_max": float(joint_effort_max[stratum, joint_index].item()),
          }
        records.append(
          StratumRecord(
            checkpoint=checkpoint,
            scenario=scenario,
            seed=seed,
            env_id=env_id,
            pair=pairs[env_id],
            arm="policy" if bool(policy_mask[env_id]) else "baseline",
            regime=regime,
            axis=axis,
            direction=direction,
            steps=count,
            duration_s=count * self.step_dt,
            terminations={
              name: int(values[env_id].item()) if stratum == last_stratum else 0
              for name, values in terminations.items()
            },
            metrics=metrics,
            joints=joints,
          )
        )
      self._clear(env_id)
    return records

  def _clear(self, env_id: int) -> None:
    """Clear every accumulator row belonging to one completed environment."""
    self._counts[env_id] = 0
    self._scalar_sums[env_id] = 0.0
    self._scalar_max[env_id] = 0.0
    self._joint_sums[env_id] = 0.0
    self._joint_effort_max[env_id] = 0.0
    self._last_stratum[env_id] = 0
