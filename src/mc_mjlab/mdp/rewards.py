"""Rewards that score the residual against what the mc_rtc controller planned."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mc_mjlab.mdp.disturbances import age_since_push, push_term
from mc_mjlab.mdp.sensors import (
  residual_term,
  scalar_sensor_range,
  zmp_sensors,
)
from mc_mjlab.robots import robot_module as mc_rtc

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg


def action_l2(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Squared magnitude of the normalized residual actually delivered."""
  action = residual_term(env, action_name).executed_normalized_action
  return torch.sum(torch.square(action), dim=1)


def requested_action_l2(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Squared normalized residual request while recovery authority is nonzero."""
  term = residual_term(env, action_name)
  cost = torch.sum(torch.square(term.requested_normalized_action), dim=1)
  return cost * (term.last_gate > 0.0)


def requested_action_rate_l2(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Squared normalized request change across consecutive authorized steps."""
  term = residual_term(env, action_name)
  delta = term.requested_normalized_action - term.previous_requested_normalized_action
  # Burst onset has no delivered predecessor; differencing against zero there
  # charges the onset step the magnitude cost twice. docs/reward-shaping.md
  paired = (term.last_gate > 0.0) & (term.previous_gate > 0.0)
  return torch.sum(torch.square(delta), dim=1) * paired


class dcm_stability:
  """Reward the robot not diverging, which is what mc_rtc's plan cannot buy itself."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
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
    return torch.exp(-torch.square(error / std)) * (normal_force >= min_normal_force)


class angular_momentum_l2:
  """Penalise centroidal angular momentum, which the stabilizer QP does not regulate."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    del cfg
    self._adr = scalar_sensor_range(env.sim.mj_model, "root_angmom", 3, env.device)

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    momentum = env.sim.data.sensordata[:, self._adr]
    return torch.sum(torch.square(momentum), dim=1)


class foot_slip:
  """Penalise a loaded sole sliding, a contact violation the QP's model cannot see."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )
    self._adr = torch.cat(
      [
        scalar_sensor_range(env.sim.mj_model, name, 3, env.device)
        for name in cfg.params["velocimeter_names"]
      ]
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    velocimeter_names: tuple[str, ...],
    min_normal_force: float = 20.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg, velocimeter_names  # Resolved at init.
    num_feet = self._adr.numel() // 3
    velocity = env.sim.data.sensordata[:, self._adr].reshape(-1, num_feet, 3)
    loaded = self._sensors.normal_forces(env) >= min_normal_force
    tangential = torch.sum(torch.square(velocity[:, :, :2]), dim=2)

    return torch.sum(tangential * loaded, dim=1)


class torque_margin:
  """Penalise peak joint torque past the robot's own hardware limit."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    term = residual_term(env, cfg.params.get("action_name", "mc_rtc_residual"))
    ids = term.residual_ids
    cols = list(range(len(term.target_names))) if ids is None else ids.tolist()
    limits = mc_rtc.get_effort_limits(term.cfg.mc_rtc_robot_name)

    missing = [term.target_names[i] for i in cols if term.target_names[i] not in limits]
    if missing:
      raise KeyError(
        f"the mc_rtc RobotModule reports no torque limit for {missing}; "
        f"`torque_margin` cannot bound a joint it has no limit for."
      )

    self._cols = torch.tensor(cols, device=env.device, dtype=torch.long)
    self._limits = torch.tensor(
      [limits[term.target_names[i]] for i in cols], device=env.device
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    soft_ratio: float = 1.0,
    action_name: str = "mc_rtc_residual",
    warmup_steps: int = 25,
  ) -> torch.Tensor:
    term = residual_term(env, action_name)
    # Fold in this step's own read: apply_actions trails sim.step by one substep,
    # so the accumulator alone misses the last of the decimation window.
    peak = torch.maximum(
      term.consume_torque_peak(),
      env.scene[term.cfg.entity_name].data.qfrc_actuator[:, term.target_ids].abs(),
    )[:, self._cols]
    over = torch.relu(peak / self._limits - soft_ratio)

    # The reset teleport drives a substep transient of 22x the limit that no policy
    # -rate sample sees and the residual did not cause. docs/reward-shaping.md
    settled = env.episode_length_buf >= warmup_steps
    return torch.sum(torch.log1p(over), dim=1) * settled


class recovery_dcm:
  """``dcm_stability``, paid only in the window after a push."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )
    # Resolved here: `get_term_cfg` walks every mode's name list.
    self._push = push_term(env, cfg.params.get("push_term_name", "push_robot"))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    window_s: float,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    push_term_name: str = "push_robot",
    action_name: str = "mc_rtc_residual",
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg, push_term_name  # Resolved at init.
    error, normal_force = self._sensors.dcm_offset(
      env, action_name, min_normal_force, plane_height
    )

    age = age_since_push(env, self._push)
    gate = (age >= 1) & (age <= round(window_s / env.step_dt))

    return (
      torch.exp(-torch.square(error / std)) * gate * (normal_force >= min_normal_force)
    )
