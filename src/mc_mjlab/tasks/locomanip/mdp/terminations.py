"""Terminations only the Locomanip FSM can decide."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mc_mjlab.tasks.locomanip.mdp import accessors

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def task_complete(
  env: ManagerBasedRlEnv, action_name: str = accessors.ACTION_NAME
) -> torch.Tensor:
  """True once the controller raises its own completion flag."""
  term = accessors.residual_action(env, action_name)
  # `reset` zeroes the collected scalars, so this cannot latch across episodes.
  return term.datastore_scalar_output(accessors.COMPLETE) > 0.0
