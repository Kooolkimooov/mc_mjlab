"""The task objective: keep the object on the trajectory the controller planned."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mc_mjlab.mdp.sensors import zmp_sensors
from mc_mjlab.tasks.locomanip.mdp import accessors

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.manager_base import ManagerTermBaseCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def object_position_tracking(
  env: ManagerBasedRlEnv, std: float, action_name: str = accessors.ACTION_NAME
) -> torch.Tensor:
  """Gaussian kernel on the object's distance from its commanded position."""
  error = accessors.object_position_error(env, action_name)
  return torch.exp(-error.square().sum(dim=-1) / std**2)


def object_yaw_tracking(
  env: ManagerBasedRlEnv, std: float, action_name: str = accessors.ACTION_NAME
) -> torch.Tensor:
  """Gaussian kernel on the object's angle from its commanded yaw."""
  error = accessors.object_yaw_error(env, action_name)
  return torch.exp(-error.square() / std**2)


class zmp_tracking:
  """Gaussian kernel on the distance from the controller's planned ZMP."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    std: float,
    action_name: str = accessors.ACTION_NAME,
    min_normal_force: float = 20.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg  # Resolved at init.
    error, normal_force = self._sensors.offset_error(env, action_name, min_normal_force)
    # Unloaded feet have no centre of pressure, so they score nothing rather than
    # a free maximum. docs/locomanip.md#zmp_tracking_std
    return torch.exp(-error.square() / std**2) * (normal_force >= min_normal_force)
