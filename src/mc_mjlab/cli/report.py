"""Collect configuration summaries independently of environment construction."""

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any

from mc_mjlab.cli.controllers import controller_records
from mc_mjlab.cli.serialization import qualified_name, serialize

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg
  from mjlab.rl import RslRlBaseRunnerCfg

_MANAGERS = (
  "rewards",
  "terminations",
  "commands",
  "events",
  "curriculum",
  "metrics",
  "recorders",
)
_TERM_FIELDS = (
  "weight",
  "time_out",
  "mode",
  "interval_range_s",
  "resampling_time_range",
  "reduce",
)
_ACTION_FIELDS = {"entity_name", "actuator_names", "scale", "offset", "clip"}
_ACTION_PREFIXES = (
  "residual_",
  "feedback_",
  "root_",
  "joint_velocity_",
  "wrench_",
  "gate_",
  "recovery_",
  "walking_",
  "motion_",
  "hold_",
  "minimum_motion_",
)


def collect_report(
  task_id: str,
  env: ManagerBasedRlEnvCfg,
  agent: RslRlBaseRunnerCfg,
  runner_cls: type,
  *,
  play: bool = False,
  full: bool = False,
) -> dict[str, Any]:
  """Describe copied configurations before any runtime resolution occurs."""
  physics_dt = env.sim.mujoco.timestep
  policy_dt = physics_dt * env.decimation
  if not math.isfinite(policy_dt) or policy_dt <= 0:
    raise ValueError(
      "physics timestep and decimation must give a positive policy period"
    )

  report: dict[str, Any] = {
    "inspection": "configuration",
    "task_id": task_id,
    "mode": "play" if play else "training",
    "detail": "full" if full else "summary",
    "runner": qualified_name(runner_cls),
    "scene": {
      "num_envs": env.scene.num_envs,
      "entities": list(env.scene.entities),
      "sensors": [sensor.name for sensor in env.scene.sensors],
    },
    "timing": {
      "physics_dt_s": physics_dt,
      "decimation": env.decimation,
      "policy_dt_s": policy_dt,
      "policy_frequency_hz": 1 / policy_dt,
      "episode_duration_s": env.episode_length_s,
      "episode_step_limit": math.ceil(env.episode_length_s / policy_dt),
    },
    "actions": {name: _action_summary(cfg) for name, cfg in env.actions.items()},
    "observations": {
      name: None
      if group is None
      else {
        "terms": list(group.terms),
        "disabled_terms": [name for name, term in group.terms.items() if term is None],
        "concatenate_terms": group.concatenate_terms,
        "history_length": group.history_length,
        "enable_corruption": group.enable_corruption,
      }
      for name, group in env.observations.items()
    },
  }
  report.update({name: _manager_summary(getattr(env, name)) for name in _MANAGERS})
  report["training"] = _training_summary(agent, env.scene.num_envs)

  controllers = controller_records(env)
  if controllers:
    report["controllers"] = controllers
    for name, controller in controllers.items():
      action = report["actions"][name]
      if action is not None:
        action["controller_period_s"] = controller["controller_period_s"]
  if full:
    report["configuration"] = {"env": serialize(env), "agent": serialize(agent)}
  return report


def _action_summary(cfg: Any) -> dict[str, Any] | None:
  """Show configured action controls without resolving actuator expressions."""
  if cfg is None:
    return None
  record = {"type": qualified_name(type(cfg))}
  record.update(
    {
      field.name: serialize(getattr(cfg, field.name))
      for field in dataclasses.fields(cfg)
      if field.name in _ACTION_FIELDS or field.name.startswith(_ACTION_PREFIXES)
    }
  )
  return record


def _manager_summary(terms: dict[str, Any]) -> dict[str, Any]:
  """Retain configured terms, including disabled and zero-weight entries."""
  return {name: _term_summary(cfg) for name, cfg in terms.items()}


def _term_summary(cfg: Any) -> dict[str, Any] | None:
  """Describe one manager term's identity and summary settings."""
  if cfg is None:
    return None
  record: dict[str, Any] = {"type": qualified_name(type(cfg))}
  if hasattr(cfg, "func"):
    record["func"] = (
      qualified_name(cfg.func) if callable(cfg.func) else serialize(cfg.func)
    )
  record.update(
    {name: serialize(getattr(cfg, name)) for name in _TERM_FIELDS if hasattr(cfg, name)}
  )
  return record


def _training_summary(agent: RslRlBaseRunnerCfg, num_envs: int) -> dict[str, Any]:
  """Collect model architecture, PPO settings and configured training budgets."""
  fields = (
    "actor",
    "critic",
    "algorithm",
    "obs_groups",
    "num_steps_per_env",
    "max_iterations",
    "seed",
    "experiment_name",
    "logger",
    "clip_actions",
  )
  record = {
    name: serialize(getattr(agent, name)) for name in fields if hasattr(agent, name)
  }
  record["rollout_transitions"] = num_envs * agent.num_steps_per_env
  return record
