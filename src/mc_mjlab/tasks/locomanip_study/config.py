"""Typed variants of the safe-completion study control, all at original authority."""

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
#: Discrete actor channels that stay instantaneous under observation history.
INSTANTANEOUS_TERMS = frozenset({"manipulation_phase", "task_complete"})


@dataclass(frozen=True)
class StudySpec:
  """One isolated change to the control: residual joints, grouping or history."""

  suffix: str
  groups: tuple[str, ...] = ()
  coordinated: bool = False
  history_frames: int = 0


CONTROL = StudySpec("control")
ARMS_WAIST = StudySpec("arms-waist", groups=("arms", "waist"))
COORDINATED = StudySpec("coordinated", groups=("arms", "waist"), coordinated=True)
HISTORY_10 = StudySpec("history-10", history_frames=10)
HISTORY_20 = StudySpec("history-20", history_frames=20)
SPECS = (CONTROL, ARMS_WAIST, COORDINATED, HISTORY_10, HISTORY_20)


def study_cfg(spec: StudySpec, play: bool = False) -> ManagerBasedRlEnvCfg:
  """The original-scale control with only the spec's change applied."""
  cfg = locomanip_residual_env_cfg(HRP5P, play=play, apply_scale_gains=False)
  if spec.history_frames:
    _apply_history(cfg, spec.history_frames)
  if spec.groups:
    _narrow_residual(cfg, spec)
  return cfg


def _apply_history(cfg: ManagerBasedRlEnvCfg, frames: int) -> None:
  """Give every continuous actor channel, and its critic copy, the same history."""
  actor = cfg.observations["actor"].terms
  shared = [name for name in actor if name not in INSTANTANEOUS_TERMS]
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    for name in shared:
      terms[name] = replace(terms[name], history_length=frames)


def _narrow_residual(cfg: ManagerBasedRlEnvCfg, spec: StudySpec) -> None:
  action = cfg.actions[ACTION_NAME]
  assert type(action) is McRtcResidualJointPositionActionCfg
  joints = tuple(STUDY_GROUPS[g] for g in spec.groups)
  if not spec.coordinated:
    cfg.actions[ACTION_NAME] = replace(action, residual_actuator_names=joints)
    return

  values: dict[str, Any] = {f.name: getattr(action, f.name) for f in fields(action)}
  values["residual_actuator_names"] = joints
  cfg.actions[ACTION_NAME] = GroupedResidualJointPositionActionCfg(
    **values, residual_groups=tuple((pattern,) for pattern in joints)
  )
