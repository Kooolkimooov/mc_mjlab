"""MDP terms specific to residual control on top of an mc_rtc controller.

mjlab's own terms cover generic robot state; these exist because a residual
task needs to talk about the controller itself: how far the policy departs
from it (``action_l2``), what it is asking for versus what it got
(``controller_position_error``, ``controller_reference_velocity``), whether it
is still generating a gait (``controller_reference_motion``) and whether it has
given up (``controller_failed``). What they compare against is the raw
controller output with the residual excluded, read off the action term.

A note on what is *not* here, because it looks like it should be. A term
comparing commanded joint positions against measured ones does not measure
"is the controller's plan being executed" under this coupling: the joints are
position-controlled with stiff PD, so the measured angle follows the commanded
one to within 0.04 rad even while the robot topples (measured), and since the
command is reference-plus-residual such a term reduces to a second penalty on
the residual. Whole-body failure shows up in the base -- attitude, height,
travel -- which is what the task's terminations and ``base_progress_tanh``
read instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


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


def base_progress_tanh(
  env: ManagerBasedRlEnv, speed: float = 0.1, asset_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
  """Reward the robot actually travelling forward, saturating at ``speed``."""
  asset = env.scene[(asset_cfg or SceneEntityCfg("robot")).name]
  forward = asset.data.root_link_lin_vel_b[:, 0].clamp(min=0.0)
  return torch.tanh(forward / speed)


def controller_reference_motion(
  env: ManagerBasedRlEnv, scale: float = 1.0, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Reward the controller still generating a gait, bounded by tanh."""
  alpha = controller_reference_velocity(env, action_name)
  return torch.tanh(torch.linalg.vector_norm(alpha, dim=1) / scale)
