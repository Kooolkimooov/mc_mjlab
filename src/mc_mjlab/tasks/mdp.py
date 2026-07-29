"""MDP terms specific to residual control on top of an mc_rtc controller.

mjlab's own terms cover generic robot state; these exist because a residual
task needs to talk about the controller itself: how far the policy departs
from it (``action_l2``), what it is asking for versus what it got
(``controller_position_error``, ``controller_reference_velocity``), whether it
is still generating a gait (``controller_reference_motion``) and whether it has
given up (``controller_failed``). What they compare against is the raw
controller output with the residual excluded, read off the action term.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _residual_term(env: ManagerBasedRlEnv, action_name: str) -> McRtcResidualActionBase:
  term = env.action_manager.get_term(action_name)
  if not isinstance(term, McRtcResidualActionBase):
    raise TypeError(
      f"action term {action_name!r} is expected to be an mc_rtc residual "
      f"action, got {type(term).__name__}"
    )
  return term


def controller_failed(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Terminate envs whose mc_rtc controller gave up."""
  return _residual_term(env, action_name).controller_failed


def action_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared magnitude of the residual action."""
  return torch.sum(torch.square(env.action_manager.action), dim=1)


def _restrict(term: McRtcResidualActionBase, values: torch.Tensor) -> torch.Tensor:
  """Keep only the columns carrying the residual (see ``residual_ids``)."""
  ids = term.residual_ids
  return values if ids is None else values[:, ids]


def controller_position_error(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """What the controller asked of the residual joints, minus what it got."""
  term = _residual_term(env, action_name)
  asset = env.scene[term.cfg.entity_name]
  error = term.controller_reference("q") - asset.data.joint_pos[:, term.target_ids]
  return _restrict(term, error)


def controller_reference_velocity(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The controller's joint-velocity reference: where its gait is headed."""
  term = _residual_term(env, action_name)
  return _restrict(term, term.controller_reference("alpha"))


def root_height_l2(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared root-height error against the controller's stance height."""
  asset = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return torch.square(height - target_height)
