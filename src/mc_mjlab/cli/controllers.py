"""Locate external controller inputs configured on mc_rtc action terms."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from mc_mjlab.bridge.config import get_controller_name, get_main_robot_name
from mc_mjlab.utils.controller_install import (
  controller_config_paths,
  user_controller_config_paths,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg

  from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionCfg


def controller_records(env: ManagerBasedRlEnvCfg) -> dict[str, Any]:
  """Describe each mc_rtc action's source files without merging their contents."""
  if not any(hasattr(action, "mc_rtc_config_path") for action in env.actions.values()):
    return {}

  from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionCfg

  return {
    name: _controller_record(action, env.sim.mujoco.timestep)
    for name, action in env.actions.items()
    if isinstance(action, McRtcResidualActionCfg)
  }


def _controller_record(
  action: McRtcResidualActionCfg, physics_dt: float
) -> dict[str, Any]:
  """Record one controller's configured robot and discovered file locations."""
  project = Path(action.mc_rtc_config_path).expanduser()
  record: dict[str, Any] = {
    "controller_name": None,
    "robot_name": action.mc_rtc_robot_name,
    "main_robot": None,
    "required_controller": action.required_controller,
    "controller_period_s": action.frameskip * physics_dt,
    "files": [
      _file_record("project_mc_rtc", project),
      _file_record("user_mc_rtc", Path.home() / ".config/mc_rtc/mc_rtc.yaml"),
    ],
    "installed_controller_configs": [],
    "user_controller_configs": [],
  }
  if action.pd_gains_path is not None:
    record["files"].append(_file_record("pd_gains", Path(action.pd_gains_path)))

  for key, reader in (
    ("controller_name", get_controller_name),
    ("main_robot", get_main_robot_name),
  ):
    try:
      record[key] = reader(project)
    except (OSError, ValueError) as error:
      record.setdefault("config_errors", []).append(str(error))

  controller = record["controller_name"]
  if controller is None:
    return record

  record["installed_controller_configs"] = [
    _file_record("installed_controller", path)
    for path in controller_config_paths(controller)
  ]
  record["user_controller_configs"] = [
    _file_record("user_controller", path)
    for path in user_controller_config_paths(controller)
  ]
  return record


def _file_record(role: str, path: Path) -> dict[str, Any]:
  """Report a candidate path and its current file presence."""
  path = path.expanduser().absolute()
  return {"role": role, "path": str(path), "present": path.is_file()}
