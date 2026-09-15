"""Length-independent diagnostics: the answer the reward sums cannot give."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mc_mjlab.bridge.controller_datastore import CONTROL_COM_VEL
from mc_mjlab.mdp.disturbances import _age_since_push, _push_term
from mc_mjlab.mdp.sensors import _residual_term, _zmp_sensors
from mc_mjlab.robots import robot_module as mc_rtc

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.manager_base import ManagerTermBaseCfg


def projection_fraction(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Fraction of residual joints changed by feasibility projection this step."""
  return _residual_term(env, action_name).projection_mask.float().mean(dim=1)


def near_bound_fraction(
  env: ManagerBasedRlEnv,
  action_name: str = "mc_rtc_residual",
  threshold: float = 0.99,
) -> torch.Tensor:
  """Fraction of normalized policy requests within ``1-threshold`` of a bound."""
  action = _residual_term(env, action_name).requested_normalized_action
  return (action.abs() >= threshold).float().mean(dim=1)


def gate_mean(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Recovery-conditioned residual authority in 0..1."""
  return _residual_term(env, action_name).last_gate


def detector_score(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Calibrated transparent recovery score before temporal filtering."""
  authority = _residual_term(env, action_name).recovery_authority
  if authority is None:
    return torch.zeros(env.num_envs, device=env.device)
  return authority.score


def inactive_residual_violation(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Peak executed residual where authority is exactly zero."""
  term = _residual_term(env, action_name)
  inactive = term.last_gate == 0.0
  peak = term.executed_physical_action.abs().amax(dim=1)
  return peak * inactive


class zmp_error:
  """Distance from the measured centre of pressure to the planned one, in metres."""

  # Read as `zmp_error / zmp_grounded`; alone it falls when the feet lift.

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = _zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    action_name: str = "mc_rtc_residual",
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg  # Resolved at init.
    error, normal_force = self._sensors.offset_error(
      env, action_name, min_normal_force, plane_height
    )
    return error * (normal_force >= min_normal_force)


class zmp_grounded:
  """Share of steps whose feet carry enough load for a centre of pressure."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = _zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    action_name: str = "mc_rtc_residual",
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg  # Resolved at init.
    _, normal_force = self._sensors.offset_error(
      env, action_name, min_normal_force, plane_height
    )
    return (normal_force >= min_normal_force).float()


class com_velocity_error:
  """Distance from the controller's commanded CoM velocity, m/s."""

  # The one term negative in every comparison: keep it as the canary for a policy
  # fighting the plan. docs/reward-shaping.md#com_velocity_error

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    self._root_body_id = env.scene[cfg.params["asset_cfg"].name].indexing.root_body_id

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    action_name: str = "mc_rtc_residual",
  ) -> torch.Tensor:
    del asset_cfg  # Resolved at init.
    term = _residual_term(env, action_name)
    error = env.sim.data.subtree_linvel[
      :, self._root_body_id
    ] - term.datastore_vector_output(CONTROL_COM_VEL)
    return torch.linalg.vector_norm(error, dim=1)


class dcm_error:
  """Distance from the divergent component of motion to the centre of pressure."""

  # Read as `dcm_error / zmp_grounded`, for the same reason `zmp_error` is.

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = _zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    action_name: str = "mc_rtc_residual",
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg  # Resolved at init.
    error, normal_force = self._sensors.dcm_offset(
      env, action_name, min_normal_force, plane_height
    )
    return error * (normal_force >= min_normal_force)


class max_effort_ratio:
  """Maximum residual-joint actuator effort divided by its hardware limit."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    term = _residual_term(env, cfg.params.get("action_name", "mc_rtc_residual"))
    ids = term.residual_ids
    cols = list(range(len(term.target_names))) if ids is None else ids.tolist()
    limits = mc_rtc.get_effort_limits(term.cfg.mc_rtc_robot_name)
    self._cols = torch.tensor(cols, device=env.device, dtype=torch.long)
    self._limits = torch.tensor(
      [limits[term.target_names[i]] for i in cols], device=env.device
    )

  def __call__(
    self, env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
  ) -> torch.Tensor:
    term = _residual_term(env, action_name)
    effort = env.scene[term.cfg.entity_name].data.qfrc_actuator[:, term.target_ids]
    return (effort[:, self._cols].abs() / self._limits).amax(dim=1)


def impulse_speed(
  env: ManagerBasedRlEnv, term_name: str = "push_robot"
) -> torch.Tensor:
  """Equivalent delta-velocity magnitude of each environment's last impulse."""
  return torch.linalg.vector_norm(_push_term(env, term_name).last_push_vel, dim=1)


class recovery_dcm_error:
  """Command-relative DCM error during a recorded recovery window."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = _zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )
    self._push = _push_term(env, cfg.params.get("push_term_name", "push_robot"))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    window_s: float,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    push_term_name: str = "push_robot",
    action_name: str = "mc_rtc_residual",
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg, push_term_name
    error, normal_force = self._sensors.dcm_offset(
      env, action_name, min_normal_force, plane_height
    )
    age = _age_since_push(env, self._push)
    active = (age >= 1) & (age <= round(window_s / env.step_dt))
    return error * active * (normal_force >= min_normal_force)


class recovery_authority_coverage:
  """Recovery-window steps that carried authority; read over ``recovery_active``."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = _zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )
    self._push = _push_term(env, cfg.params.get("push_term_name", "push_robot"))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    window_s: float,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    push_term_name: str = "push_robot",
    action_name: str = "mc_rtc_residual",
    min_normal_force: float = 20.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg, push_term_name
    normal_force = self._sensors.normal_forces(env).sum(dim=1)
    age = _age_since_push(env, self._push)
    active = (age >= 1) & (age <= round(window_s / env.step_dt))
    active = active & (normal_force >= min_normal_force)
    return active & (_residual_term(env, action_name).last_gate > 0.0)


class recovery_active:
  """Grounded indicator for the recorded post-disturbance recovery window."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = _zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )
    self._push = _push_term(env, cfg.params.get("push_term_name", "push_robot"))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    window_s: float,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    push_term_name: str = "push_robot",
    min_normal_force: float = 20.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg, push_term_name
    normal_force = self._sensors.normal_forces(env).sum(dim=1)
    age = _age_since_push(env, self._push)
    active = (age >= 1) & (age <= round(window_s / env.step_dt))
    return active * (normal_force >= min_normal_force)
