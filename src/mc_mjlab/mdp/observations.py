"""Observations of the controller's references and requests, joint load and pushes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mc_mjlab.bridge.controller_datastore import CONTROL_COM_VEL
from mc_mjlab.mdp.disturbances import NEVER_AGE, age_since_push, push_term
from mc_mjlab.mdp.sensors import (
  planned_zmp_offset,
  residual_columns,
  residual_term,
)
from mc_mjlab.robots import robot_module as mc_rtc

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.manager_base import ManagerTermBaseCfg


def executed_action(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The physical residual actually delivered to the actuator target."""
  return residual_term(env, action_name).executed_physical_action


def requested_normalized_action(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The bounded policy request before physical scaling."""
  return residual_term(env, action_name).requested_normalized_action


def requested_physical_action(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The physical residual requested before gating and feasibility projection."""
  return residual_term(env, action_name).requested_physical_action


def recovery_dcm_error_vector(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Signed deployable DCM error used by the recovery detector."""
  authority = residual_term(env, action_name).recovery_authority
  if authority is None:
    return torch.zeros(env.num_envs, 2, device=env.device)
  return authority.dcm_error


def controller_position_error(
  env: ManagerBasedRlEnv,
  action_name: str = "mc_rtc_residual",
  biased: bool = True,
  residual_only: bool = True,
) -> torch.Tensor:
  """Controller reference minus encoder-visible or privileged joint position."""
  term = residual_term(env, action_name)
  asset = env.scene[term.cfg.entity_name]
  position = asset.data.joint_pos_biased if biased else asset.data.joint_pos
  error = term.controller_reference("q") - position[:, term.target_ids]
  return residual_columns(term, error) if residual_only else error


def controller_reference_velocity(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The controller's joint-velocity reference: where its gait is headed."""
  term = residual_term(env, action_name)
  return residual_columns(term, term.controller_reference("alpha"))


def controller_reference_position(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The controller's joint-position reference: the vector the residual is added to."""
  term = residual_term(env, action_name)
  return residual_columns(term, term.controller_reference("q"))


def controller_planned_zmp_offset(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The controller's planned CoM-to-ZMP offset: where it means to push."""
  return planned_zmp_offset(env, action_name)


def controller_planned_com_velocity(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The CoM velocity the controller's plan calls for."""
  return residual_term(env, action_name).datastore_vector_output(CONTROL_COM_VEL)


class joint_torque:
  """Actuator torque of every limited target joint, as a fraction of its limit."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    term = residual_term(env, cfg.params.get("action_name", "mc_rtc_residual"))
    limits = mc_rtc.get_effort_limits(term.cfg.mc_rtc_robot_name)

    # Not the residual columns: the load is measured whether or not the policy
    # acts on a joint, and the feedback task's residual has none.
    names = [name for name in term.target_names if name in limits]
    self._cols = torch.tensor(
      [term.target_names.index(name) for name in names],
      device=env.device,
      dtype=torch.long,
    )
    self._limits = torch.tensor([limits[name] for name in names], device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Stateless; the observation manager calls this on every class term."""
    del env_ids

  def __call__(
    self, env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
  ) -> torch.Tensor:
    term = residual_term(env, action_name)
    effort = env.scene[term.cfg.entity_name].data.qfrc_actuator[:, term.target_ids]
    return effort[:, self._cols] / self._limits


def steps_since_push(
  env: ManagerBasedRlEnv, term_name: str = "push_robot"
) -> torch.Tensor:
  """Policy steps since each env was last pushed *within its current episode*."""
  return age_since_push(env, push_term(env, term_name))


def push_recency(
  env: ManagerBasedRlEnv, term_name: str = "push_robot", tau_s: float = 2.0
) -> torch.Tensor:
  """1 at the instant of a push, decaying to 0; 0 for an env not pushed this episode."""
  # Bounded on purpose: `steps_since_push` reports NEVER_AGE (2^30), which would
  # wreck the observation normalizer if fed in raw.
  age = age_since_push(env, push_term(env, term_name)).clamp(min=0)
  return torch.exp(-age.float() * env.step_dt / tau_s).unsqueeze(-1)


def last_push_velocity(
  env: ManagerBasedRlEnv, term_name: str = "push_robot"
) -> torch.Tensor:
  """The velocity delta of the last push, zeroed once it predates this episode."""
  term = push_term(env, term_name)
  within = (age_since_push(env, term) < NEVER_AGE).unsqueeze(-1)
  return term.last_push_vel * within
