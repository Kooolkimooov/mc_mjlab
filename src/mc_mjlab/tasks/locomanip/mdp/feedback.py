"""Observations and diagnostics of the hand-force feedback actually dispatched."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mc_mjlab.tasks.locomanip.mdp.accessors import residual_action

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def executed_feedback(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Read six normalized force corrections from the most recent dispatch."""
  return residual_action(env).executed_normalized_action


def feedback_active(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Read whether the most recent dispatch authorized hand-force feedback."""
  return residual_action(env).last_gate


def feedback_force_rms(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Read the RMS dispatched force correction over all six components in newtons."""
  return residual_action(env).executed_physical_action.square().mean(dim=1).sqrt()


def feedback_saturation(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Read the fraction of dispatched components at their configured force bound."""
  return (executed_feedback(env).abs() >= 0.999).float().mean(dim=1)
