"""Build and validate canonical effective-training checkpoint contracts."""

from __future__ import annotations

import dataclasses
import enum
import functools
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = 1
_INJECTED_PARAMETERS = {"self", "env", "env_ids"}

_OPERATIONAL_ENV_KEYS = {
  "console_output",
  "num_envs",
  "num_workers",
  "print_residual_every",
  "viewer",
}

_OPERATIONAL_RUNNER_KEYS = {
  "experiment_name",
  "load_checkpoint",
  "load_run",
  "logger",
  "max_iterations",
  "multi_gpu",
  "resume",
  "run_name",
  "save_interval",
  "upload_model",
  "wandb_project",
  "wandb_tags",
}

_RUNTIME_MODULES = (
  "mjlab.envs.manager_based_rl_env",
  "mjlab.managers.action_manager",
  "mjlab.managers.curriculum_manager",
  "mjlab.managers.event_manager",
  "mjlab.managers.metrics_manager",
  "mjlab.managers.observation_manager",
  "mjlab.managers.reward_manager",
  "mjlab.managers.termination_manager",
  "mjlab.rl.runner",
  "rsl_rl.runners.on_policy_runner",
)
_RUNTIME_PACKAGES = ("mc-mjlab", "mjlab", "mujoco", "rsl-rl-lib", "torch")

# A contract names every term and model by import path, so the source-tree move
# would have stranded every checkpoint written before it. Verified path-only:
# the moved modules were not edited. docs/evaluation.md#_RENAMED_MODULES
_RENAMED_MODULES = {
  "mc_mjlab.controller_datastore": "mc_mjlab.bridge.controller_datastore",
  "mc_mjlab.recovery_authority": "mc_mjlab.residuals.recovery_authority",
  "mc_mjlab.residual_mpc": "mc_mjlab.residuals.mpc_math",
  "mc_mjlab.residual_printer": "mc_mjlab.residuals.printer",
  "mc_mjlab.residual_safety": "mc_mjlab.residuals.safety",
  "mc_mjlab.robots.additional_sensors_configuration": "mc_mjlab.robots.sensors",
  "mc_mjlab.robots.collision_configuration": "mc_mjlab.robots.collisions",
  "mc_mjlab.robots.mc_rtc_robot_configuration": "mc_mjlab.robots.robot_module",
  "mc_mjlab.robots.pd_actuator_configuration": "mc_mjlab.robots.actuators",
  "mc_mjlab.robots.robots_registry": "mc_mjlab.robots.registry",
  "mc_mjlab.sim_controller_bridge": "mc_mjlab.bridge.sim_controller_bridge",
  "mc_mjlab.tasks.residual_balance.effective_training_manifest": (
    "mc_mjlab.rl.effective_training_manifest"
  ),
  "mc_mjlab.tasks.rollout_adaptive_ppo": "mc_mjlab.rl.rollout_adaptive_ppo",
  "mc_mjlab.tasks.squashed_gaussian": "mc_mjlab.rl.squashed_gaussian",
  "mc_mjlab.tasks.zero_init_actor": "mc_mjlab.rl.zero_init_actor",
  "utils.mc_rtc_config": "mc_mjlab.bridge.config",
  "utils.pd_gains": "mc_mjlab.robots.pd_gains",
  "utils.shared_memory": "mc_mjlab.bridge.shared_memory",
  "utils.task_naming": "mc_mjlab.tasks.naming",
}

_RENAMED_CALLABLES = {
  "mc_mjlab.tasks.residual_mpc.mdp:root_position": "mc_mjlab.tasks.residual_mpc.mdp.observations:root_position",
  "mc_mjlab.tasks.residual_mpc.mdp:root_quaternion": "mc_mjlab.tasks.residual_mpc.mdp.observations:root_quaternion",
  "mc_mjlab.tasks.residual_mpc.mdp:joint_position": "mc_mjlab.tasks.residual_mpc.mdp.observations:joint_position",
  "mc_mjlab.tasks.residual_mpc.mdp:joint_velocity": "mc_mjlab.tasks.residual_mpc.mdp.observations:joint_velocity",
  "mc_mjlab.tasks.residual_mpc.mdp:body_linear_velocity": "mc_mjlab.tasks.residual_mpc.mdp.observations:body_linear_velocity",
  "mc_mjlab.tasks.residual_mpc.mdp:body_angular_velocity": "mc_mjlab.tasks.residual_mpc.mdp.observations:body_angular_velocity",
  "mc_mjlab.tasks.residual_mpc.mdp:controller_contact_phases": "mc_mjlab.tasks.residual_mpc.mdp.observations:controller_contact_phases",
  "mc_mjlab.tasks.residual_mpc.mdp:controller_qp_objective": "mc_mjlab.tasks.residual_mpc.mdp.observations:controller_qp_objective",
  "mc_mjlab.tasks.residual_mpc.mdp:linear_velocity_tracking": "mc_mjlab.tasks.residual_mpc.mdp.rewards:linear_velocity_tracking",
  "mc_mjlab.tasks.residual_mpc.mdp:angular_velocity_tracking": "mc_mjlab.tasks.residual_mpc.mdp.rewards:angular_velocity_tracking",
  "mc_mjlab.tasks.residual_mpc.mdp:first_action_rate": "mc_mjlab.tasks.residual_mpc.mdp.rewards:first_action_rate",
  "mc_mjlab.tasks.residual_mpc.mdp:second_action_rate": "mc_mjlab.tasks.residual_mpc.mdp.rewards:second_action_rate",
  "mc_mjlab.tasks.residual_mpc.mdp:torque_l2": "mc_mjlab.tasks.residual_mpc.mdp.rewards:torque_l2",
  "mc_mjlab.tasks.residual_mpc.mdp:orientation_reward": "mc_mjlab.tasks.residual_mpc.mdp.rewards:orientation_reward",
  "mc_mjlab.tasks.residual_mpc.mdp:height_reward": "mc_mjlab.tasks.residual_mpc.mdp.rewards:height_reward",
  "mc_mjlab.tasks.residual_mpc.mdp:joint_regularization": "mc_mjlab.tasks.residual_mpc.mdp.rewards:joint_regularization",
  "mc_mjlab.tasks.residual_mpc.mdp:self_collision": "mc_mjlab.tasks.residual_mpc.mdp.rewards:self_collision",
  "mc_mjlab.tasks.residual_mpc.mdp:excessive_base_speed": "mc_mjlab.tasks.residual_mpc.mdp.terminations:excessive_base_speed",
  "mc_mjlab.tasks.residual_mpc.mdp:excessive_angular_speed": "mc_mjlab.tasks.residual_mpc.mdp.terminations:excessive_angular_speed",
  "mc_mjlab.tasks.residual_mpc.mdp:height_outside": "mc_mjlab.tasks.residual_mpc.mdp.terminations:height_outside",
  "mc_mjlab.tasks.residual_mpc.mdp:refresh_action_scaling": "mc_mjlab.tasks.residual_mpc.mdp.disturbances:refresh_action_scaling",
  "mc_mjlab.tasks.residual_mpc.mdp:initial_velocity_kick": "mc_mjlab.tasks.residual_mpc.mdp.disturbances:initial_velocity_kick",
  "mc_mjlab.tasks.residual_mpc.mdp:survival_kick_curriculum": "mc_mjlab.tasks.residual_mpc.mdp.curricula:survival_kick_curriculum",
  "mc_mjlab.tasks.residual_mpc.mdp:forward_speed": "mc_mjlab.tasks.residual_mpc.mdp.metrics:forward_speed",
  "mc_mjlab.tasks.residual_mpc.mdp:commanded_speed": "mc_mjlab.tasks.residual_mpc.mdp.metrics:commanded_speed",
  "mc_mjlab.tasks.residual_mpc.mdp:maximum_effort_ratio": "mc_mjlab.tasks.residual_mpc.mdp.metrics:active_effort_ratio",
  "mc_mjlab.mdp.metrics:max_effort_ratio": "mc_mjlab.mdp.metrics:nominal_effort_ratio",
  "mc_mjlab.tasks.residual_mpc.mdp:controller_failed": "mc_mjlab.mdp.terminations:controller_failed",
  "mc_mjlab.tasks.residual_mpc.mdp:controller_worker_failed": "mc_mjlab.mdp.terminations:controller_worker_failed",
  "mc_mjlab.tasks.residual_mpc.mdp:projection_fraction": "mc_mjlab.mdp.metrics:projection_fraction",
  "mc_mjlab.tasks.mdp:max_effort_ratio": "mc_mjlab.mdp.metrics:nominal_effort_ratio",
}

#: Split across `mc_mjlab.mdp`, so its terms are looked up, not mapped by hand.
_SPLIT_MDP_MODULE = "mc_mjlab.tasks.mdp"
_MDP_SUBMODULES = (
  "curricula",
  "disturbances",
  "metrics",
  "observations",
  "rewards",
  "sensors",
  "terminations",
)


def _qualified_name(value: Any) -> str:
  """Return a stable import-style name for a callable or type."""
  target = value if inspect.isfunction(value) or inspect.isclass(value) else type(value)
  return f"{target.__module__}:{target.__qualname__}"


@functools.lru_cache(maxsize=None)
def _file_sha256(path: str) -> str | None:
  """Hash a source file when it exists."""
  source = Path(path)
  if not source.is_file():
    return None
  return hashlib.sha256(source.read_bytes()).hexdigest()


def _callable_record(value: Any) -> dict[str, Any]:
  """Record callable identity together with its defining source file."""
  target = value if inspect.isfunction(value) or inspect.isclass(value) else type(value)
  try:
    source_path = inspect.getsourcefile(target)
  except TypeError:
    source_path = None
  return {
    "name": _qualified_name(value),
    "source_sha256": _file_sha256(source_path) if source_path else None,
  }


def canonicalize(value: Any) -> Any:
  """Convert configuration values to deterministic JSON-compatible data."""
  if value is None or isinstance(value, (str, int, bool)):
    return value
  if isinstance(value, float):
    return value if math.isfinite(value) else str(value)
  if isinstance(value, enum.Enum):
    return canonicalize(value.value)
  if isinstance(value, Path):
    return str(value.expanduser().resolve())
  if isinstance(value, torch.Tensor):
    return canonicalize(value.detach().cpu().tolist())

  if dataclasses.is_dataclass(value) and not inspect.isclass(value):
    return {
      "__type__": _qualified_name(value),
      **{
        field.name: canonicalize(getattr(value, field.name))
        for field in dataclasses.fields(value)
      },
    }

  if isinstance(value, Mapping):
    return {
      str(key): canonicalize(item)
      for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
    }
  if isinstance(value, (list, tuple)):
    return [canonicalize(item) for item in value]
  if isinstance(value, (set, frozenset)):
    items = [canonicalize(item) for item in value]
    return sorted(items, key=_json_text)
  if isinstance(value, slice):
    return {
      "start": canonicalize(value.start),
      "stop": canonicalize(value.stop),
      "step": canonicalize(value.step),
    }
  if isinstance(value, torch.dtype | torch.device):
    return str(value)

  if callable(value):
    return {"__callable__": _callable_record(value)}

  if hasattr(value, "tolist") and callable(value.tolist):
    try:
      return canonicalize(value.tolist())
    except (TypeError, ValueError, RuntimeError):
      pass
  if hasattr(value, "item") and callable(value.item):
    try:
      return canonicalize(value.item())
    except (TypeError, ValueError, RuntimeError):
      pass

  return {"__type__": _qualified_name(value), "value": str(value)}


def _json_text(value: Any) -> str:
  """Serialize canonical data for hashing and ordering."""
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
  """Hash canonical data with an unambiguous JSON encoding."""
  return hashlib.sha256(_json_text(value).encode()).hexdigest()


def _effective_parameters(func: Any, params: Mapping[str, Any]) -> dict[str, Any]:
  """Resolve explicit term parameters and callable defaults into one contract."""
  target = func
  if not inspect.isfunction(func) and not inspect.isclass(func):
    target = type(func).__call__
  try:
    parameters = inspect.signature(target).parameters
  except (TypeError, ValueError):
    parameters = {}
  effective: dict[str, Any] = {}
  for name, parameter in parameters.items():
    if name in _INJECTED_PARAMETERS:
      continue
    if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
      continue
    if name in params:
      effective[name] = {"source": "explicit", "value": canonicalize(params[name])}
    elif parameter.default is not inspect.Parameter.empty:
      effective[name] = {
        "source": "default",
        "value": canonicalize(parameter.default),
      }
    else:
      effective[name] = {"source": "required"}
  for name in sorted(set(params) - set(effective)):
    effective[name] = {"source": "explicit", "value": canonicalize(params[name])}
  return effective


def _term_contract(cfg: Any) -> dict[str, Any]:
  """Describe a manager term after manager-side resolution and instantiation."""
  contract = {"config": canonicalize(cfg)}
  func = getattr(cfg, "func", None)
  if func is not None:
    params = getattr(cfg, "params", {})
    contract["callable"] = _callable_record(func)
    contract["effective_parameters"] = _effective_parameters(func, params)
  return contract


def _flat_active_terms(active_terms: Any) -> list[str]:
  """Flatten ordinary and mode-grouped active-term collections."""
  if isinstance(active_terms, Mapping):
    return [name for names in active_terms.values() for name in names]
  return list(active_terms)


def _manager_contract(manager: Any) -> dict[str, Any]:
  """Capture one manager's live resolved term configuration."""
  active_terms = getattr(manager, "active_terms", [])
  names = _flat_active_terms(active_terms)
  getter = getattr(manager, "get_term_cfg", None)
  terms = {}
  if callable(getter):
    for name in names:
      terms[name] = _term_contract(getter(name))
  return {
    "active_terms": canonicalize(active_terms),
    "terms": terms,
  }


def _observation_contract(manager: Any) -> dict[str, Any]:
  """Capture observation ordering, dimensions, group settings, and terms."""
  groups = {}
  for group_name, names in manager.active_terms.items():
    group_cfg = manager.cfg[group_name]
    group_settings = canonicalize(group_cfg)
    if isinstance(group_settings, dict):
      group_settings.pop("terms", None)
    groups[group_name] = {
      "settings": group_settings,
      "terms": {
        name: _term_contract(manager.get_term_cfg(group_name, name)) for name in names
      },
    }
  return {
    "active_terms": canonicalize(manager.active_terms),
    "group_dimensions": canonicalize(manager.group_obs_dim),
    "term_dimensions": canonicalize(manager.group_obs_term_dim),
    "groups": groups,
  }


def _action_contract(manager: Any) -> dict[str, Any]:
  """Capture action ordering, dimensions, and term configuration."""
  terms = {}
  for name in manager.active_terms:
    term = manager.get_term(name)
    terms[name] = {
      "config": canonicalize(manager.cfg[name]),
      "config_source": _callable_record(type(manager.cfg[name])),
      "implementation": _callable_record(type(term)),
    }
  return {
    "active_terms": canonicalize(manager.active_terms),
    "term_dimensions": canonicalize(manager.action_term_dim),
    "total_dimension": manager.total_action_dim,
    "terms": terms,
  }


def _runtime_contract() -> dict[str, Any]:
  """Record dependency versions and core source-file identities."""
  packages = {}
  for package in _RUNTIME_PACKAGES:
    try:
      packages[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
      packages[package] = None
  modules = {}
  for module_name in _RUNTIME_MODULES:
    module = importlib.import_module(module_name)
    source_path = inspect.getsourcefile(module)
    modules[module_name] = _file_sha256(source_path) if source_path else None
  return {
    "packages": packages,
    "python": platform.python_version(),
    "source_sha256": modules,
  }


def _configured_class_sources(value: Any, prefix: str = "") -> dict[str, Any]:
  """Resolve source identities for import-style configured class names."""
  records = {}
  if isinstance(value, Mapping):
    for key, item in value.items():
      path = f"{prefix}.{key}" if prefix else str(key)
      if key == "class_name" and isinstance(item, str):
        record: dict[str, Any] = {"name": item, "source_sha256": None}
        if ":" in item:
          module_name, attribute = item.split(":", maxsplit=1)
          target = importlib.import_module(module_name)
          for part in attribute.split("."):
            target = getattr(target, part)
          record = _callable_record(target)
        records[path] = record
      records.update(_configured_class_sources(item, path))
  elif isinstance(value, (list, tuple)):
    for index, item in enumerate(value):
      records.update(_configured_class_sources(item, f"{prefix}[{index}]"))
  return records


def _drop_keys(value: Any, excluded: set[str]) -> Any:
  """Recursively remove operational fields from a canonical config."""
  if isinstance(value, dict):
    return {
      key: _drop_keys(item, excluded)
      for key, item in value.items()
      if key not in excluded
    }
  if isinstance(value, list):
    return [_drop_keys(item, excluded) for item in value]
  return value


def _effective_managers(env: Any) -> dict[str, Any]:
  """Capture all manager contracts that can affect rollout semantics."""
  contracts = {
    "actions": _action_contract(env.action_manager),
    "observations": _observation_contract(env.observation_manager),
  }
  for name in (
    "command",
    "curriculum",
    "event",
    "metrics",
    "reward",
    "termination",
  ):
    manager = getattr(env, f"{name}_manager", None)
    if manager is not None:
      contracts[name] = _manager_contract(manager)
  return contracts


def _policy_observations(
  observations: dict[str, Any], train_cfg: Mapping[str, Any]
) -> dict[str, Any]:
  """Select structural actor observations while excluding evaluation corruption."""
  actor_groups = train_cfg.get("obs_groups", {}).get("actor", ("actor",))
  groups = {}
  for group_name in actor_groups:
    group = observations["groups"][group_name]
    settings = group["settings"]
    if isinstance(settings, dict):
      settings = {
        key: value
        for key, value in settings.items()
        if key not in {"enable_corruption", "nan_policy", "nan_check_per_term"}
      }

    terms = {}
    for name, term in group["terms"].items():
      term = dict(term)
      config = term.get("config")
      if isinstance(config, dict):
        term["config"] = {
          key: value
          for key, value in config.items()
          if key
          not in {
            "delay_hold_prob",
            "delay_max_lag",
            "delay_min_lag",
            "delay_per_env",
            "delay_per_env_phase",
            "delay_update_period",
            "noise",
          }
        }
      terms[name] = term

    groups[group_name] = {"settings": settings, "terms": terms}

  return {
    "active_terms": {
      group_name: observations["active_terms"][group_name]
      for group_name in actor_groups
    },
    "group_dimensions": {
      group_name: observations["group_dimensions"][group_name]
      for group_name in actor_groups
    },
    "term_dimensions": {
      group_name: observations["term_dimensions"][group_name]
      for group_name in actor_groups
    },
    "groups": groups,
  }


def build_effective_training_manifest(env: Any, train_cfg: Mapping[str, Any]) -> dict:
  """Build audit, training-resume, and actor-interface checkpoint contracts."""
  env = env.unwrapped
  runner_config = canonicalize(train_cfg)
  env_config = canonicalize(env.cfg)
  managers = _effective_managers(env)
  runtime = _runtime_contract()
  runtime["configured_class_sources"] = _configured_class_sources(train_cfg)

  record = {
    "environment": env_config,
    "managers": managers,
    "runner": runner_config,
    "runtime": runtime,
  }

  training = {
    "environment": _drop_keys(env_config, _OPERATIONAL_ENV_KEYS),
    "managers": _drop_keys(managers, _OPERATIONAL_ENV_KEYS),
    "runner": _drop_keys(runner_config, _OPERATIONAL_RUNNER_KEYS),
    "runtime": runtime,
  }

  policy_interface = {
    "actions": _drop_keys(managers["actions"], _OPERATIONAL_ENV_KEYS),
    "actor": runner_config.get("actor"),
    "actor_sources": _configured_class_sources({"actor": train_cfg.get("actor")}),
    "observations": _policy_observations(managers["observations"], train_cfg),
  }

  # Source bytes stay in `record` for audit and out of both enforced contracts:
  # an unrelated edit to a defining file must not strand a checkpoint.
  # docs/leo-mjlab-review.md, docs/evaluation.md#source_drift
  training = _strip_source_hashes(training)
  policy_interface = _strip_source_hashes(policy_interface)

  return {
    "schema_version": SCHEMA_VERSION,
    "record": record,
    "record_sha256": _digest(record),
    "training_contract": training,
    "training_sha256": _digest(training),
    "policy_interface": policy_interface,
    "policy_interface_sha256": _digest(policy_interface),
  }


def _strip_source_hashes(value: Any) -> Any:
  """Drop file digests so an enforced contract is semantic, not byte identity."""
  if isinstance(value, dict):
    return {
      key: _strip_source_hashes(item)
      for key, item in value.items()
      if key != "source_sha256"
    }
  if isinstance(value, list):
    return [_strip_source_hashes(item) for item in value]
  return value


def rename_legacy_modules(value: Any) -> Any:
  """Rewrite import paths a checkpoint recorded before the source tree moved."""
  if isinstance(value, str):
    return _renamed_qualified_name(value)
  if isinstance(value, dict):
    return {key: rename_legacy_modules(item) for key, item in value.items()}
  if isinstance(value, list):
    return [rename_legacy_modules(item) for item in value]
  return value


def _renamed_qualified_name(name: str) -> str:
  """Map one ``module:qualname`` (or bare module) onto its current spelling."""
  if name in _RENAMED_CALLABLES:
    return _RENAMED_CALLABLES[name]
  module, separator, attribute = name.partition(":")
  if module == _SPLIT_MDP_MODULE and attribute:
    return _split_mdp_name(attribute) or name
  renamed = _RENAMED_MODULES.get(module)
  qualified = f"{renamed}{separator}{attribute}" if renamed else name
  return _RENAMED_CALLABLES.get(qualified, qualified)


def _split_mdp_name(attribute: str) -> str | None:
  """Find which ``mc_mjlab.mdp`` submodule a once-single term now lives in."""
  package = importlib.import_module("mc_mjlab.mdp")
  name = attribute.split(".")[0]
  for submodule in _MDP_SUBMODULES:
    target = getattr(getattr(package, submodule), name, None)
    if target is not None and target.__module__ == f"mc_mjlab.mdp.{submodule}":
      return f"{target.__module__}:{attribute}"
  return None


def source_drift(saved: Mapping[str, Any], active: Mapping[str, Any]) -> list[str]:
  """Return the audit-record paths whose defining source file changed."""
  # Only the digests: the record also holds evaluator-side differences such as
  # worker count and the qualifier's own disturbance term, which are expected.
  return [
    path
    for path in _differences(
      rename_legacy_modules(saved.get("record")), active.get("record")
    )
    if path.endswith("source_sha256")
  ]


def _differences(saved: Any, active: Any, prefix: str = "") -> list[str]:
  """Return leaf paths that differ between two canonical structures."""
  if type(saved) is not type(active):
    return [prefix or "<root>"]
  if isinstance(saved, dict):
    paths = []
    for key in sorted(set(saved) | set(active)):
      path = f"{prefix}.{key}" if prefix else key
      if key not in saved or key not in active:
        paths.append(path)
      else:
        paths.extend(_differences(saved[key], active[key], path))
    return paths
  if isinstance(saved, list):
    paths = []
    for index in range(max(len(saved), len(active))):
      path = f"{prefix}[{index}]"
      if index >= len(saved) or index >= len(active):
        paths.append(path)
      else:
        paths.extend(_differences(saved[index], active[index], path))
    return paths
  return [] if saved == active else [prefix or "<root>"]


def _added_keys(saved: Any, active: Any, prefix: str = "") -> list[str]:
  """Return paths the active contract gained, which a checkpoint could not record."""
  if isinstance(saved, dict) and isinstance(active, dict):
    paths = []
    for key in sorted(active):
      path = f"{prefix}.{key}" if prefix else key
      if key not in saved:
        paths.append(path)
      else:
        paths.extend(_added_keys(saved[key], active[key], path))
    return paths
  return []


def _drop_paths(value: Any, paths: set[str], prefix: str = "") -> Any:
  """Rebuild a contract without the named leaves."""
  if not isinstance(value, dict):
    return value
  return {
    key: _drop_paths(item, paths, f"{prefix}.{key}" if prefix else key)
    for key, item in value.items()
    if (f"{prefix}.{key}" if prefix else key) not in paths
  }


def _normalize_impulse_defaults(value: Any) -> Any:
  """Drop only legacy, unused defaults from the three surviving impulse terms."""
  if isinstance(value, list):
    return [_normalize_impulse_defaults(item) for item in value]
  if not isinstance(value, dict):
    return value
  result = {key: _normalize_impulse_defaults(item) for key, item in value.items()}
  name = result.get("callable", {}).get("name")
  if name not in {
    "mc_mjlab.mdp.disturbances:finite_impulse_curriculum",
    "mc_mjlab.mdp.disturbances:stratified_finite_impulse_curriculum",
    "mc_mjlab.mdp.disturbances:achievement_finite_impulse_curriculum",
  }:
    return result
  parameters = result.get("effective_parameters", {})
  for key, default in {
    "asset_cfg": None,
    "rehearsal_weights": None,
    "initial_stage": 0,
    "bands": None,
    "band_weights": None,
  }.items():
    if parameters.get(key) == {"source": "default", "value": default}:
      parameters.pop(key)
  return result


def validate_effective_training_manifest(
  saved: dict, active: dict, *, full_resume: bool
) -> None:
  """Reject incompatible actor loads and semantically changed full resumes."""
  if saved.get("schema_version") != active.get("schema_version"):
    raise RuntimeError("Checkpoint effective-training manifest schema is unsupported.")

  saved = rename_legacy_modules(saved)
  contract = "training" if full_resume else "policy_interface"
  payload_key = f"{contract}_contract" if full_resume else contract
  # Re-digest both sides: a checkpoint written before source hashes became
  # audit-only still carries them. docs/evaluation.md#source_drift
  saved_payload = _normalize_impulse_defaults(
    _strip_source_hashes(saved.get(payload_key))
  )
  active_payload = _normalize_impulse_defaults(
    _strip_source_hashes(active.get(payload_key))
  )

  # A field added after a checkpoint was written cannot appear in its manifest,
  # so an addition alone must not strand it. Removals and changed values still
  # fail, and observation ordering and dimensions are compared separately.
  # docs/evaluation.md#source_drift
  added = set(_added_keys(saved_payload, active_payload))
  if added:
    saved_payload = _drop_paths(saved_payload, added)
    active_payload = _drop_paths(active_payload, added)

  if _digest(saved_payload) == _digest(active_payload):
    if added:
      print(
        f"[mc_mjlab] checkpoint predates {len(added)} contract "
        f"{'field' if len(added) == 1 else 'fields'}: {', '.join(sorted(added)[:4])}",
        flush=True,
      )
    _warn_source_drift(saved, active)
    return

  paths = _differences(saved_payload, active_payload)[:12]
  detail = ", ".join(paths) if paths else "digest only"
  raise RuntimeError(
    f"Checkpoint {contract.replace('_', ' ')} differs from the active run: {detail}."
  )


def _warn_source_drift(saved: Mapping[str, Any], active: Mapping[str, Any]) -> None:
  """Report defining files that changed while the contract still matches."""
  drift = source_drift(saved, active)
  if not drift:
    return
  print(
    f"[mc_mjlab] checkpoint contract matches but {len(drift)} audit-record "
    f"entries changed, first: {', '.join(drift[:4])}",
    flush=True,
  )


def materialize_effective_training_manifest(manifest: dict, log_dir: Path) -> None:
  """Write the effective manifest beside the ordinary YAML parameter dump."""
  path = log_dir / "effective_training_manifest.json"
  path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def curriculum_runtime_snapshot(env: Any) -> dict[str, Any]:
  """Capture mutable curriculum targets and the step that selected them."""
  env = env.unwrapped
  return {
    "common_step_counter": int(env.common_step_counter),
    "curriculum_state": canonicalize(env.curriculum_manager._curriculum_state),
    "rewards": _manager_contract(env.reward_manager),
    "terminations": _manager_contract(env.termination_manager),
  }


def synchronize_resumed_curriculum(env: Any, infos: Mapping[str, Any]) -> dict:
  """Reapply curricula after the checkpoint's global counter has been restored."""
  env = env.unwrapped
  saved_state = infos.get("env_state")
  if not saved_state or "common_step_counter" not in saved_state:
    return curriculum_runtime_snapshot(env)

  expected = int(saved_state["common_step_counter"])
  actual = int(env.common_step_counter)
  if actual != expected:
    raise RuntimeError(
      f"Checkpoint step restoration failed: expected {expected}, found {actual}."
    )

  env.curriculum_manager.compute()
  return curriculum_runtime_snapshot(env)
