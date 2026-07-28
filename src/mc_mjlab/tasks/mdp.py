"""MDP terms specific to residual control on top of an mc_rtc controller.

mjlab's own terms cover the rest; these two exist because a residual task
needs to talk about the residual itself (how far the policy departs from the
controller) and about holding the controller's nominal stance height.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def controller_failed(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Terminate envs whose mc_rtc controller gave up."""
  term = env.action_manager.get_term(action_name)
  if not isinstance(term, McRtcResidualActionBase):
    raise TypeError(
      f"termination term 'controller_failed' expects action term "
      f"'{action_name}' to be an mc_rtc residual action, got {type(term).__name__}"
    )
  return term.controller_failed


def action_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared magnitude of the residual action."""
  return torch.sum(torch.square(env.action_manager.action), dim=1)


def root_height_l2(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared root-height error against the controller's stance height."""
  asset = env.scene[asset_cfg.name]
  height = asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
  return torch.square(height - target_height)
