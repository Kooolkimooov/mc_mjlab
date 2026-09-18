"""Matched constant-action conditions and terminal-safe locomanip measurements."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from evaluation.rollout import reset_done
from mc_mjlab.tasks.locomanip.mdp import accessors, observations

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

CHANNELS = ("left_fx", "left_fy", "left_fz", "right_fx", "right_fy", "right_fz")


def conditions(task: str, levels: list[float], channel: str) -> list[dict[str, Any]]:
  """Include repeated zero controls and independent signed force excitations."""
  if task == "position":
    return [{"channel": "all", "level": level} for level in [0.0, *levels]]

  rows = [{"channel": "zero", "level": 0.0} for _ in range(3)]
  channels = CHANNELS if channel == "all" else (channel,)
  for name in channels:
    for magnitude in sorted({abs(level) for level in levels if level != 0}):
      rows.extend({"channel": name, "level": sign * magnitude} for sign in (1, -1))
  return rows


def action_matrix(env: ManagerBasedRlEnv, rows: list[dict[str, Any]]) -> torch.Tensor:
  """Build normalized requests without exciting unrelated hand components."""
  actions = torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )
  for index, row in enumerate(rows):
    if row["channel"] == "all":
      actions[index] = row["level"]
    elif row["channel"] != "zero":
      actions[index, CHANNELS.index(row["channel"])] = row["level"]
  return actions


class AuthorityTrace:
  """Retain each condition's first episode and its matched controller response."""

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self.env = env
    self.term = accessors.residual_action(env)
    self.start = env.scene[accessors.OBJECT_ENTITY].data.root_link_pos_w[:, 0].clone()
    self.lowest = env.scene["robot"].data.root_link_pos_w[:, 2].clone()
    self.finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.held = torch.zeros(env.num_envs, device=env.device)
    self.response_sum = torch.zeros_like(self.held)
    self.response_steps = torch.zeros_like(self.held)
    self.executed_sum = torch.zeros_like(self.held)
    self.executed_steps = torch.zeros_like(self.held)
    self.final: list[dict[str, Any] | None] = [None] * env.num_envs

  def update(self, terminated: torch.Tensor, truncated: torch.Tensor) -> None:
    """Measure before recycling failed rows so a fallen controller cannot wedge."""
    env, term = self.env, self.term
    live = ~self.finished
    holding = accessors.holding(env).bool()
    self.held += (live & holding) * env.step_dt
    height = env.scene["robot"].data.root_link_pos_w[:, 2]
    self.lowest.copy_(
      torch.where(live, torch.minimum(self.lowest, height), self.lowest)
    )

    matched = live & holding & ~self.finished[0] & holding[0]
    reference = term.controller_reference("q")
    difference = (reference - reference[:1]).square().mean(dim=1)
    self.response_sum += torch.where(matched, difference, 0.0)
    self.response_steps += matched
    self.executed_sum += torch.where(
      live, term.executed_physical_action.square().mean(dim=1), 0.0
    )
    self.executed_steps += live

    done = terminated | truncated
    for index in (done & live).nonzero().flatten().tolist():
      self.final[index] = self._snapshot(
        index, bool(terminated[index]), bool(truncated[index])
      )
    self.finished |= done
    if done.any():
      reset_done(env, done.nonzero().flatten())

  def results(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare every response against the repeated-zero numerical floor."""
    results = [
      {**row, **(self.final[index] or self._snapshot(index, False, False))}
      for index, row in enumerate(rows)
    ]
    noise = max(r["target_response_rms_rad"] for r in results if r["level"] == 0)
    for row in results:
      row["zero_response_floor_rad"] = noise
      row["measurable_authority"] = (
        row["level"] != 0
        and row["matched_steps"] > 0
        and row["target_response_rms_rad"] > noise + 1e-6
      )
    return results

  def _snapshot(self, index: int, terminated: bool, truncated: bool) -> dict[str, Any]:
    """Capture physical outcomes before a row's reset replaces them."""
    env, term = self.env, self.term
    contacts = accessors.hand_contact_forces(env).reshape(env.num_envs, 2, 3)
    return {
      "cart_travel_m": float(
        env.scene[accessors.OBJECT_ENTITY].data.root_link_pos_w[index, 0]
        - self.start[index]
      ),
      "object_position_error_m": float(
        accessors.object_position_error(env)[index].norm()
      ),
      "object_yaw_error_rad": float(accessors.object_yaw_error(env)[index].abs()),
      "held_s": float(self.held[index]),
      "min_base_z_m": float(self.lowest[index]),
      "phases": observations.manipulation_phase(env)[index].int().tolist(),
      "hand_contact_force_n": contacts[index].norm(dim=-1).tolist(),
      "executed_residual_rms": float(
        (self.executed_sum[index] / self.executed_steps[index].clamp(min=1)).sqrt()
      ),
      "target_response_rms_rad": float(
        (self.response_sum[index] / self.response_steps[index].clamp(min=1)).sqrt()
      ),
      "matched_steps": int(self.response_steps[index]),
      "terminated": terminated,
      "truncated": truncated,
      "controller_failed": bool(term.controller_failed[index]),
      "worker_failed": bool(term.controller_worker_failed[index]),
    }
