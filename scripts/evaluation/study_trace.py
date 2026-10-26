"""Terminal-safe study records and raw phase exposure independent of action width."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from evaluation.pd_demand import PdDemandSummary
from evaluation.rollout import metric_snapshot
from evaluation.study_noise import KeyedNoise
from mc_mjlab.tasks.locomanip.mdp import accessors, observations


class StudyTrace:
  """Capture selected diagnostic traces and outcomes before any manager reset."""

  def __init__(
    self,
    env: Any,
    rows: list[dict],
    path: Path | None,
    pd: PdDemandSummary | None = None,
  ) -> None:
    self.env, self.rows, self.pd = env, rows, pd
    self.finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.valid = torch.ones_like(self.finished)
    self.start = env.scene["robot"].data.root_link_pos_w[:, 0].clone()
    self.lowest = torch.full((env.num_envs,), float("inf"), device=env.device)
    self.tilt = torch.zeros_like(self.lowest)
    self.exposure = [defaultdict(float) for _ in rows]
    self.results: list[dict | None] = [None] * len(rows)
    self.stream = path.open("w") if path else None

  def begin(self, ids: torch.Tensor) -> None:
    """Start target episodes without clearing surviving rows' accumulators."""
    self.start[ids] = self.env.scene["robot"].data.root_link_pos_w[ids, 0]
    self.valid[ids] = True
    self.lowest[ids] = float("inf")
    self.tilt[ids] = 0
    for i in ids.tolist():
      self.exposure[i].clear()
    if self.pd is not None:
      self.pd.begin(ids)

  def update(self, active: torch.Tensor) -> dict:
    """Retain raw phases, freshness, motion and residual delivery at the last substep."""
    env = self.env
    term = accessors.residual_action(env)
    robot = env.scene["robot"].data
    tilt = torch.acos((-robot.projected_gravity_b[:, 2]).clamp(-1, 1))
    values = {
      "time_s": env.episode_length_buf * env.step_dt,
      "root_position_m": robot.root_link_pos_w - env.scene.env_origins,
      "root_velocity_m_s": robot.root_link_lin_vel_w,
      "tilt_rad": tilt,
      "phase_left": term.datastore_scalar_output(accessors.LEFT_PHASE),
      "phase_right": term.datastore_scalar_output(accessors.RIGHT_PHASE),
      "datastore_fresh": term._datastore_output_fresh,
      "controller_q": term.controller_reference("q"),
      "joint_position_rad": robot.joint_pos,
      "joint_velocity_rad_s": robot.joint_vel,
      "joint_position_target_rad": robot.joint_pos_target,
      "foot_force_kn": observations.foot_normal_forces(env),
      "requested_action": term.requested_normalized_action,
      "executed_residual_rad": term.executed_physical_action,
      "cart_error_m": accessors.object_position_error(env),
      "hand_force_n": accessors.hand_contact_forces(env),
    }
    for value in values.values():
      self.valid[active] &= (
        torch.isfinite(value).reshape(env.num_envs, -1).all(dim=1)[active]
      )
    self.lowest[active] = torch.minimum(
      self.lowest[active], robot.root_link_pos_w[active, 2]
    )
    self.tilt[active] = torch.maximum(self.tilt[active], tilt[active])
    frame = {key: value.cpu().tolist() for key, value in values.items()}
    for name, noise in env.observation_manager._group_obs_class_instances.get(
      "actor", {}
    ).items():
      if isinstance(noise, KeyedNoise) and noise.last_noise is not None:
        frame[f"observation_noise/{name}"] = noise.last_noise.cpu().tolist()
    rewards = env.reward_manager._step_reward.cpu().tolist()
    for column, name in enumerate(env.reward_manager.active_terms):
      frame[f"reward/{name}"] = [row[column] for row in rewards]
    names = env.termination_manager.active_terms
    flags = (
      torch.stack([env.termination_manager.get_term(n) for n in names], dim=1)
      .cpu()
      .tolist()
    )
    frame["terminations"] = [dict(zip(names, row, strict=True)) for row in flags]
    if self.pd is not None:
      frame.update(self.pd.update(active))
    for i in active.nonzero().flatten().tolist():
      key = f"{frame['phase_left'][i]:g}/{frame['phase_right'][i]:g}/fresh={frame['datastore_fresh'][i]}"
      self.exposure[i][key] += env.step_dt
      if self.stream:
        self.stream.write(
          json.dumps({**self.rows[i], **{k: v[i] for k, v in frame.items()}}) + "\n"
        )
    return frame

  def finish(
    self,
    ids: torch.Tensor,
    frame: dict,
    observed: bool = True,
    warmup_failed: bool = False,
  ) -> None:
    """Keep timeouts as outcomes and failed warm-ups as missing target measurements."""
    env = self.env
    metrics = metric_snapshot(env, ids)
    for offset, i in enumerate(ids.tolist()):
      flags = frame["terminations"][i]
      # Timing is diagnostic; a policy step contains several residual-fed solves.
      start = bool(flags.get("controller_failed")) and frame["time_s"][i] <= (
        1.5 * env.step_dt
      )
      failed = any(
        flags.get(n, False) for n in (*accessors.FALL_TERMINATIONS, "controller_failed")
      )
      worker = bool(flags.get("controller_worker_failed", False))
      valid = bool(self.valid[i]) and observed and not warmup_failed
      complete = (
        bool(flags.get("task_complete", False)) and not failed and not worker and valid
      )
      self.results[i] = {
        **self.rows[i],
        "observed": valid,
        "warmup_failed": warmup_failed,
        "worker_failure": worker,
        "controller_start_failure": start,
        "combined_failure": failed,
        "complete": complete and not start,
        "timeout": bool(flags.get("time_out", False)) and not complete,
        "terminations": flags,
        "time_s": frame["time_s"][i],
        "actual_mass_kg": float(accessors.object_mass_kg(env)[i]),
        "encoder_bias_rad": env.scene["robot"].data.encoder_bias[i].cpu().tolist(),
        "robot_displacement_m": float(
          env.scene["robot"].data.root_link_pos_w[i, 0] - self.start[i]
        ),
        "min_height_m": float(self.lowest[i]) if valid else None,
        "max_tilt_rad": float(self.tilt[i]) if valid else None,
        "phase_exposure_s": dict(self.exposure[i]),
        "terminal_phase": [frame["phase_left"][i], frame["phase_right"][i]],
        "metrics": {name: values[offset] for name, values in metrics.items()},
        "weighted_reward_totals": {
          name: float(value[i])
          for name, value in env.reward_manager._episode_sums.items()
        },
      }
      if self.pd is not None:
        self.results[i]["pd_demand"] = self.pd.result(i)
    self.finished[ids] = True

  def close(self) -> None:
    if self.stream:
      self.stream.close()
