"""Observations and diagnostics for explicit controller motion authority."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mc_mjlab.actions.locomanip_motion_action import (
  MOTION_COUNT_GETTER,
  MOTION_DELAY_GETTER,
  MOTION_FILTER_GETTER,
  MOTION_SCALE_GETTER,
  MOTION_STATE_GETTER,
  MOTION_TARGET_GETTER,
)
from mc_mjlab.tasks.locomanip.mdp import accessors

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from mc_mjlab.actions.locomanip_motion_action import LocomanipMotionAction


def motion_observation(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Expose motion execution, filtered intent, accepted target and request delay."""
  term = cast("LocomanipMotionAction", accessors.residual_action(env))
  return torch.stack(
    (
      term.motion_scale_dispatched,
      term.datastore_scalar_output(MOTION_SCALE_GETTER),
      term.datastore_scalar_output(MOTION_STATE_GETTER) / 2.0,
      term.datastore_scalar_output(MOTION_TARGET_GETTER),
      term.datastore_scalar_output(MOTION_FILTER_GETTER),
      (term.datastore_scalar_output(MOTION_DELAY_GETTER) / 3.0).clamp(0.0, 1.0),
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


def braking(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Charge actual applied slowdown during bilateral Hold."""
  return (1.0 - motion_scale(env).clamp(0.0, 1.0)) * accessors.holding(env)


def braking_start(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Return an event rate so the timestep-scaled reward charges once per start."""
  term = cast("LocomanipMotionAction", accessors.residual_action(env))
  return term.motion_braking_events / env.step_dt


def braking_count(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Report the controller's exact episode count of braking starts."""
  return accessors.residual_action(env).datastore_scalar_output(MOTION_COUNT_GETTER)


def holding(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Charge sustained holding separately from travelling at reduced speed."""
  return motion_holding(env) * accessors.holding(env)


def motion_decision_state(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Expose the pre-action decision mask, clock and latched motion request."""
  return cast(
    "LocomanipMotionAction", accessors.residual_action(env)
  ).motion_decision_state


def motion_decisions(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Count selections of motion requests rather than repeated held actions."""
  return cast("LocomanipMotionAction", accessors.residual_action(env)).motion_decisions
