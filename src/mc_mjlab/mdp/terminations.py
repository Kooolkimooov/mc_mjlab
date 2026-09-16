"""Terminations that end an episode on a fall or on the controller giving up."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp import terminations as mjlab_terminations
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mc_mjlab.mdp.sensors import residual_term

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def controller_failed(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Terminate envs whose mc_rtc controller gave up."""
  return residual_term(env, action_name).controller_failed


def controller_worker_failed(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """End envs whose controller *process* died -- as a truncation, not a failure."""
  # Correlated across a worker's envs and unrelated to the action, so it must be
  # configured `time_out=True`. docs/coupling.md#worker-failure-is-a-truncation
  return residual_term(env, action_name).controller_worker_failed


def collapsed(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  limit_angle: float,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Root below ``minimum_height`` while still upright: a crouch, not a topple."""
  # Exclusive with `bad_orientation` on purpose, so the two counters partition the
  # balance failures; their union is what it always was.
  cfg = asset_cfg or SceneEntityCfg("robot")
  low = mjlab_terminations.root_height_below_minimum(env, minimum_height, cfg)
  return low & ~mjlab_terminations.bad_orientation(env, limit_angle, cfg)
