"""Repeatable disturbances for paired qualification episodes."""

from __future__ import annotations

import math
import random

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply

from mc_mjlab import mdp
from mc_mjlab.tasks.residual_balance.curriculum_stages import (
  achievement_stage,
)


class PairedDisturbances:
  """Apply a deterministic schedule shared by both arms of each episode pair."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    scenario: str,
    seed: int,
    achievement_stage_index: int | None = None,
  ) -> None:
    self.env = env
    self.scenario = scenario
    self.seed = seed
    self.achievement_stage = (
      achievement_stage(achievement_stage_index)
      if achievement_stage_index is not None
      else None
    )
    self.asset = env.scene["robot"]
    self.remaining = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.force = torch.zeros(env.num_envs, 1, 3, device=env.device)
    self.torque = torch.zeros_like(self.force)

  def before_step(
    self, active: torch.Tensor, pairs: list[int], episode_steps: torch.Tensor
  ) -> None:
    """Expire wrenches and trigger disturbances at fixed episode ages."""
    expired = (self.remaining == 0) & (self.force.square().sum(dim=(1, 2)) > 0.0)
    if bool(expired.any()):
      self._write_zeros(expired.nonzero(as_tuple=False).flatten())
    if self.scenario != "nominal":
      first = round(10.0 / self.env.step_dt)
      interval = round(6.0 / self.env.step_dt)
      due = (
        active & (episode_steps >= first) & ((episode_steps - first) % interval == 0)
      )
      ids = due.nonzero(as_tuple=False).flatten()
      if ids.numel():
        occurrences = ((episode_steps[ids] - first) // interval).tolist()
        self._trigger(ids, pairs, occurrences)

  def after_step(self) -> None:
    """Advance the finite-wrench duration by one policy step."""
    self.remaining.clamp_min_(0)
    self.remaining -= (self.remaining > 0).long()

  def clear(self, env_ids: torch.Tensor) -> None:
    """Clear any finite wrench before resetting an environment."""
    if env_ids.numel():
      self._write_zeros(env_ids)

  def _trigger(
    self, env_ids: torch.Tensor, pairs: list[int], occurrences: list[int]
  ) -> None:
    delta_b = torch.zeros(len(env_ids), 3, device=self.env.device)
    durations = torch.zeros(len(env_ids), device=self.env.device)
    heights = torch.zeros(len(env_ids), device=self.env.device)
    for row, (env_id, occurrence) in enumerate(
      zip(env_ids.tolist(), occurrences, strict=True)
    ):
      rng = random.Random(
        self.seed * 1_000_003
        + env_id * 10_007
        + pairs[env_id] * 101
        + occurrence * 17
        + sum(map(ord, self.scenario))
      )
      angle = rng.uniform(-math.pi, math.pi)
      if self.achievement_stage is None:
        dv = rng.uniform(0.50, 0.60) if self.scenario == "robust" else 0.40
      elif self.scenario == "robust":
        dv = rng.uniform(*self.achievement_stage.robust_velocity_range)
      else:
        dv = self.achievement_stage.qualification_velocity
      delta_b[row, :2] = torch.tensor(
        [dv * math.cos(angle), dv * math.sin(angle)], device=self.env.device
      )
      durations[row] = rng.uniform(0.08, 0.20)
      heights[row] = rng.uniform(0.0, 0.25)
    if self.scenario == "current_kick":
      self._velocity_kick(env_ids, delta_b)
    else:
      self._finite_impulse(env_ids, delta_b, durations, heights)
    mdp.disturbances.record_disturbance(self.env, env_ids, delta_b)

  def _velocity_kick(self, env_ids: torch.Tensor, delta_b: torch.Tensor) -> None:
    quat = self.asset.data.root_link_quat_w[env_ids]
    delta_w = quat_apply(quat, delta_b)
    velocity = self.asset.data.root_link_vel_w[env_ids].clone()
    velocity[:, :3] += delta_w
    self.asset.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

  def _finite_impulse(
    self,
    env_ids: torch.Tensor,
    delta_b: torch.Tensor,
    durations: torch.Tensor,
    heights: torch.Tensor,
  ) -> None:
    quat = self.asset.data.root_link_quat_w[env_ids]
    delta_w = quat_apply(quat, delta_b)
    body_ids = self.asset.indexing.body_ids
    mass = self.env.sim.model.body_mass[env_ids][:, body_ids].sum(dim=1)
    force = mass.unsqueeze(-1) * delta_w / durations.unsqueeze(-1)
    offset_b = torch.zeros_like(force)
    offset_b[:, 2] = heights
    offset_w = quat_apply(quat, offset_b)
    torque = torch.cross(offset_w, force, dim=1)
    self.force[env_ids, 0] = force
    self.torque[env_ids, 0] = torque
    self.remaining[env_ids] = torch.ceil(durations / self.env.step_dt).long()
    self.asset.write_external_wrench_to_sim(
      self.force[env_ids], self.torque[env_ids], env_ids=env_ids, body_ids=[0]
    )

  def _write_zeros(self, env_ids: torch.Tensor) -> None:
    zeros = torch.zeros(len(env_ids), 1, 3, device=self.env.device)
    self.asset.write_external_wrench_to_sim(zeros, zeros, env_ids=env_ids, body_ids=[0])
    self.force[env_ids] = 0.0
    self.torque[env_ids] = 0.0
    self.remaining[env_ids] = 0
