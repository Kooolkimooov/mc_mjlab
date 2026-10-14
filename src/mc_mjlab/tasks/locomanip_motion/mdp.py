"""Observations and diagnostics for explicit controller motion authority."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mc_mjlab.actions.locomanip_motion_action import (
  MOTION_LIMIT_GETTER,
  MOTION_SCALE_GETTER,
  MOTION_STATE_GETTER,
)
from mc_mjlab.tasks.locomanip.mdp import accessors

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from mc_mjlab.actions.locomanip_motion_action import LocomanipMotionAction


def motion_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Report dispatched/applied scale, transition state and the resume limit."""
  term = cast("LocomanipMotionAction", accessors.residual_action(env))
  return torch.stack(
    (
      term.motion_scale_dispatched,
      term.datastore_scalar_output(MOTION_SCALE_GETTER),
      term.datastore_scalar_output(MOTION_STATE_GETTER) / 2.0,
      term.datastore_scalar_output(MOTION_LIMIT_GETTER),
    ),
    dim=1,
  )


def motion_scale(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Controller-reported scale, including the deceleration transition."""
  return accessors.residual_action(env).datastore_scalar_output(MOTION_SCALE_GETTER)


def motion_holding(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Whether the controller has completed braking and is holding the cart."""
  state = accessors.residual_action(env).datastore_scalar_output(MOTION_STATE_GETTER)
  return (state == 2.0).to(torch.float32)
