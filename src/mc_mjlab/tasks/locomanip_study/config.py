"""Typed action variants for the safe-completion study, all at original authority."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any

from mc_mjlab.actions.grouped_residual_action import (
  GroupedResidualJointPositionActionCfg,
)
from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionActionCfg,
)
from mc_mjlab.tasks.locomanip.locomanip_residual_env_cfg import (
  locomanip_residual_env_cfg,
)
from mc_mjlab.tasks.locomanip.mdp.accessors import ACTION_NAME
from mc_mjlab.tasks.locomanip.profiles import HRP5P

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg

#: HRP5P joint groups the phase probe excites, as actuator-name regexes.
STUDY_GROUPS = {"arms": r"[LR](S[CPRY]|EP)", "waist": r"W[PRY]"}


@dataclass(frozen=True)
class StudyActionSpec:
  """Which residual joints a variant drives and whether outputs are shared per group."""

  suffix: str
  groups: tuple[str, ...] = ()
  coordinated: bool = False


CONTROL = StudyActionSpec("control")
ARMS_WAIST = StudyActionSpec("arms-waist", groups=("arms", "waist"))
COORDINATED = StudyActionSpec("coordinated", groups=("arms", "waist"), coordinated=True)
SPECS = (CONTROL, ARMS_WAIST, COORDINATED)


def study_cfg(spec: StudyActionSpec, play: bool = False) -> ManagerBasedRlEnvCfg:
  """The original-scale control with only its residual action narrowed."""
  cfg = locomanip_residual_env_cfg(HRP5P, play=play, apply_scale_gains=False)
  if not spec.groups:
    return cfg

  action = cfg.actions[ACTION_NAME]
  assert type(action) is McRtcResidualJointPositionActionCfg
  joints = tuple(STUDY_GROUPS[g] for g in spec.groups)
  if not spec.coordinated:
    cfg.actions[ACTION_NAME] = replace(action, residual_actuator_names=joints)
    return cfg

  values: dict[str, Any] = {f.name: getattr(action, f.name) for f in fields(action)}
  values["residual_actuator_names"] = joints
  cfg.actions[ACTION_NAME] = GroupedResidualJointPositionActionCfg(
    **values, residual_groups=tuple((pattern,) for pattern in joints)
  )
  return cfg
