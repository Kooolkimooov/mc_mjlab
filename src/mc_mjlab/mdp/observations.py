"""Observations of the controller's own references, requests and disturbances."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mc_mjlab.actions.walking_reference_action import WALKING_REF_VEL_GETTER
from mc_mjlab.bridge.controller_datastore import CONTROL_COM_VEL
from mc_mjlab.mdp.disturbances import NEVER_AGE, _age_since_push, _push_term
from mc_mjlab.mdp.sensors import (
  _residual_term,
  _restrict,
  _walking_term,
  planned_zmp_offset,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def executed_action(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The physical residual actually delivered to the actuator target."""
  return _residual_term(env, action_name).executed_physical_action


def walking_reference_velocity(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Executed walking-reference delta in normalized action coordinates."""
  return _walking_term(env, action_name).walking_reference_normalized


def requested_normalized_action(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The bounded policy request before physical scaling."""
  return _residual_term(env, action_name).requested_normalized_action


def requested_physical_action(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The physical residual requested before gating and feasibility projection."""
  return _residual_term(env, action_name).requested_physical_action


def recovery_dcm_error_vector(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Signed deployable DCM error used by the recovery detector."""
  authority = _residual_term(env, action_name).recovery_authority
  if authority is None:
    return torch.zeros(env.num_envs, 2, device=env.device)
  return authority.dcm_error


def controller_position_error(
  env: ManagerBasedRlEnv,
  action_name: str = "mc_rtc_residual",
  biased: bool = True,
) -> torch.Tensor:
  """Controller reference minus encoder-visible or privileged joint position."""
  term = _residual_term(env, action_name)
  asset = env.scene[term.cfg.entity_name]
  position = asset.data.joint_pos_biased if biased else asset.data.joint_pos
  error = term.controller_reference("q") - position[:, term.target_ids]
  return _restrict(term, error)


def controller_reference_velocity(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The controller's joint-velocity reference: where its gait is headed."""
  term = _residual_term(env, action_name)
  return _restrict(term, term.controller_reference("alpha"))


def controller_reference_position(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The controller's joint-position reference: the vector the residual is added to."""
  term = _residual_term(env, action_name)
  return _restrict(term, term.controller_reference("q"))


def controller_planned_zmp_offset(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The controller's planned CoM-to-ZMP offset: where it means to push."""
  return planned_zmp_offset(env, action_name)


def controller_planned_com_velocity(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The CoM velocity the controller's plan calls for."""
  return _residual_term(env, action_name).datastore_vector_output(CONTROL_COM_VEL)


def controller_walking_reference_velocity(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Current ``(vx, vy, yaw_rate)`` command reported by the walking controller."""
  return _residual_term(env, action_name).datastore_vector_output(
    WALKING_REF_VEL_GETTER
  )


def steps_since_push(
  env: ManagerBasedRlEnv, term_name: str = "push_robot"
) -> torch.Tensor:
  """Policy steps since each env was last pushed *within its current episode*."""
  return _age_since_push(env, _push_term(env, term_name))


def push_recency(
  env: ManagerBasedRlEnv, term_name: str = "push_robot", tau_s: float = 2.0
) -> torch.Tensor:
  """1 at the instant of a push, decaying to 0; 0 for an env not pushed this episode."""
  # Bounded on purpose: `steps_since_push` reports NEVER_AGE (2^30), which would
  # wreck the observation normalizer if fed in raw.
  age = _age_since_push(env, _push_term(env, term_name)).clamp(min=0)
  return torch.exp(-age.float() * env.step_dt / tau_s).unsqueeze(-1)


def last_push_velocity(
  env: ManagerBasedRlEnv, term_name: str = "push_robot"
) -> torch.Tensor:
  """The velocity delta of the last push, zeroed once it predates this episode."""
  term = _push_term(env, term_name)
  within = (_age_since_push(env, term) < NEVER_AGE).unsqueeze(-1)
  return term.last_push_vel * within
