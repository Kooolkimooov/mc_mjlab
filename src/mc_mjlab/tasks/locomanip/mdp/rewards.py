"""Balance and safety rewards only the Locomanip task scores."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mc_mjlab.mdp.sensors import contact_zmp_sensors
from mc_mjlab.tasks.locomanip.mdp import accessors
from mc_mjlab.tasks.locomanip.mdp.observations import (
  root_height,
  root_vertical_velocity,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.manager_base import ManagerTermBaseCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg


class contact_dcm_stability:
  """Score commanded CoM motion against the net contact wrench."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    self._contacts, self._ground = contact_zmp_sensors(
      env,
      cfg.params["sensor_names"],
      cfg.params.get("contact_sensor_names"),
      cfg.params["asset_cfg"].name,
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    std: float,
    contact_sensor_names: tuple[str, ...] | None = None,
    action_name: str = accessors.ACTION_NAME,
    min_normal_force: float = 20.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg, contact_sensor_names  # Resolved at init.
    error, _ = self._contacts.dcm_offset(env, action_name, min_normal_force)
    _, normal_force = self._ground.measured_offset(env, min_normal_force)
    # Unloaded feet have no centre of pressure, so they score nothing rather than
    # a free maximum. docs/locomanip.md#locomanip-safety-objective
    return torch.exp(-error.square() / std**2) * (normal_force >= min_normal_force)


def collapse_risk(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  warning_margin: float,
  downward_speed: float,
) -> torch.Tensor:
  """Penalize a low root earlier when it is still descending."""
  shortfall = minimum_height + warning_margin - root_height(env).squeeze(-1)
  proximity = (shortfall / warning_margin).clamp(min=0.0, max=1.0)
  descent = (-root_vertical_velocity(env).squeeze(-1) / downward_speed).clamp(
    min=0.0, max=1.0
  )
  return proximity.square() * (1.0 + descent)


def rapid_tilt(env: ManagerBasedRlEnv, free_rate: float) -> torch.Tensor:
  """Charge only roll and pitch rates above the normal motion band."""
  velocity = env.scene[accessors.ROBOT_ENTITY].data.root_link_ang_vel_b[:, :2]
  rate = torch.linalg.vector_norm(velocity, dim=-1)
  return (rate - free_rate).clamp(min=0.0).square()
