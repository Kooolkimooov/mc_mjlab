"""Deterministic effective training manifest contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from mc_mjlab.rl.effective_training_manifest import (
  build_effective_training_manifest,
  source_drift,
  synchronize_resumed_curriculum,
  validate_effective_training_manifest,
)


def _manifest_probe(_env: Any, gain: float = 2.0) -> float:
  """Provide a callable default for effective-manifest assertions."""
  return gain


@dataclass
class _ActionCfg:
  """Provide action fields used by manifest assertions."""

  entity_name: str = "robot"
  scale: float = 0.1
  num_workers: int = 2


class _TermManager:
  """Expose the ordinary manager contract to manifest assertions."""

  def __init__(self, terms: dict[str, RewardTermCfg] | None = None) -> None:
    self.cfg = terms or {}
    self.active_terms = list(self.cfg)

  def get_term_cfg(self, name: str) -> RewardTermCfg:
    """Return one configured fake term."""
    return self.cfg[name]


class _CurriculumManager(_TermManager):
  """Apply a deterministic fake stage from the restored global counter."""

  def __init__(self, env: Any) -> None:
    super().__init__()
    self.env = env
    self._curriculum_state = {"torque_margin": 0}
    self.compute_calls = 0

  def compute(self) -> None:
    """Apply the fake late stage at 48,000 steps."""
    self.compute_calls += 1
    stage = int(self.env.common_step_counter >= 48_000)
    self.env.reward_manager.cfg["probe"].weight = -0.5 if stage else -0.05
    self._curriculum_state["torque_margin"] = stage


class _ObservationManager:
  """Expose actor observation ordering and dimensions to the manifest."""

  def __init__(self) -> None:
    term = ObservationTermCfg(func=_manifest_probe)
    self.cfg = {"actor": ObservationGroupCfg(terms={"probe": term})}
    self.active_terms = {"actor": ["probe"]}
    self.group_obs_dim = {"actor": (1,)}
    self.group_obs_term_dim = {"actor": [(1,)]}

  def get_term_cfg(self, group: str, name: str) -> RewardTermCfg:
    """Return one configured fake observation term."""
    return self.cfg[group].terms[name]


class _ActionManager:
  """Expose action ordering and dimensions to the manifest."""

  def __init__(self) -> None:
    self.cfg = {"residual": _ActionCfg()}
    self.active_terms = ["residual"]
    self.action_term_dim = [1]
    self.total_action_dim = 1

  def get_term(self, _name: str) -> Any:
    """Return a fake built action term."""
    return self


class _ManifestEnv:
  """Provide a lightweight manager-based environment contract."""

  def __init__(self) -> None:
    self.unwrapped = self
    self.cfg = {"scene": {"num_envs": 8}, "viewer": {"width": 640}}
    self.common_step_counter = 48_000
    self.action_manager = _ActionManager()
    self.observation_manager = _ObservationManager()
    self.reward_manager = _TermManager(
      {"probe": RewardTermCfg(func=_manifest_probe, weight=-0.05)}
    )
    self.termination_manager = _TermManager()
    self.command_manager = _TermManager()
    self.event_manager = _TermManager()
    self.metrics_manager = _TermManager()
    self.curriculum_manager = _CurriculumManager(self)


def test_source_hash_is_audit_only() -> None:
  """Check an unrelated source edit cannot strand a checkpoint."""
  saved = {
    "schema_version": 1,
    "record": {"terms": {"a": {"callable": {"name": "m:f", "source_sha256": "old"}}}},
    "training_contract": {"terms": {"a": {"callable": {"name": "m:f"}}}},
    "training_sha256": "same",
    "policy_interface": {"terms": {"a": {"callable": {"name": "m:f"}}}},
    "policy_interface_sha256": "same",
  }
  active = json.loads(json.dumps(saved))
  active["record"]["terms"]["a"]["callable"]["source_sha256"] = "new"

  # Neither enforced contract may carry a file digest.
  for key in ("training_contract", "policy_interface"):
    assert "source_sha256" not in json.dumps(saved[key])

  # Source drift is reported, never fatal, on either contract.
  for full in (True, False):
    validate_effective_training_manifest(saved, active, full_resume=full)

  # A checkpoint written before the change still carries hashes; both sides are
  # re-digested, so a legacy manifest still loads.
  legacy = json.loads(json.dumps(saved))
  for key in ("training_contract", "policy_interface"):
    legacy[key]["terms"]["a"]["callable"]["source_sha256"] = "old"
  legacy["training_sha256"] = "legacy"
  legacy["policy_interface_sha256"] = "legacy"
  for full in (True, False):
    validate_effective_training_manifest(legacy, active, full_resume=full)
  assert source_drift(saved, active) == ["terms.a.callable.source_sha256"]
  assert source_drift(saved, saved) == []
  # Evaluator-side record differences are not source drift.
  evaluator = json.loads(json.dumps(saved))
  evaluator["record"]["terms"]["a"]["callable"]["name"] = "m:other"
  evaluator["record"]["num_workers"] = 8
  assert source_drift(saved, evaluator) == []

  # A field added after the checkpoint was written is not an interface change.
  grown = json.loads(json.dumps(saved))
  grown["policy_interface"]["terms"]["a"]["new_field"] = []
  grown["training_contract"]["terms"]["a"]["new_field"] = []
  for full in (True, False):
    validate_effective_training_manifest(saved, grown, full_resume=full)
  # A field the active contract lost is still fatal.
  shrunk = json.loads(json.dumps(saved))
  del shrunk["policy_interface"]["terms"]["a"]["callable"]
  try:
    validate_effective_training_manifest(saved, shrunk, full_resume=False)
  except RuntimeError:
    pass
  else:
    raise AssertionError("a removed interface field was accepted")

  # A genuine interface change still raises.
  renamed = json.loads(json.dumps(saved))
  renamed["policy_interface"]["terms"]["a"]["callable"]["name"] = "m:other"
  renamed["policy_interface_sha256"] = "different"
  try:
    validate_effective_training_manifest(saved, renamed, full_resume=False)
  except RuntimeError:
    pass
  else:
    raise AssertionError("a renamed observation callable was accepted")


def test_renamed_modules_do_not_strand_checkpoints() -> None:
  """Check the source-tree move alone cannot invalidate an older checkpoint."""
  contract = {
    "terms": {"a": {"callable": {"name": "mc_mjlab.mdp.rewards:requested_action_l2"}}},
    "actor": {"class_name": "mc_mjlab.rl.zero_init_actor:ZeroInitMLPModel"},
    "entity": {"__type__": "mc_mjlab.robots.registry:RobotSpec"},
  }
  active = {
    "schema_version": 1,
    "record": {},
    "training_contract": contract,
    "policy_interface": contract,
  }
  saved = json.loads(
    json.dumps(active)
    .replace(
      "mc_mjlab.mdp.rewards:requested_action_l2",
      "mc_mjlab.tasks.mdp:requested_action_l2",
    )
    .replace("mc_mjlab.rl.zero_init_actor", "mc_mjlab.tasks.zero_init_actor")
    .replace("mc_mjlab.robots.registry", "mc_mjlab.robots.robots_registry")
  )
  assert saved != active
  for full in (True, False):
    validate_effective_training_manifest(saved, active, full_resume=full)

  # Only the recorded path is rewritten: a term that is no longer there fails.
  gone = json.loads(json.dumps(saved))
  gone["policy_interface"]["terms"]["a"]["callable"]["name"] = "mc_mjlab.tasks.mdp:gone"
  try:
    validate_effective_training_manifest(gone, active, full_resume=False)
  except RuntimeError:
    pass
  else:
    raise AssertionError("a term missing from the split mdp package was accepted")


def test_effective_training_manifest() -> None:
  """Check default capture, deterministic hashes, and evaluation compatibility."""
  env = _ManifestEnv()
  train_cfg = {
    "actor": {"hidden_dims": [8, 4]},
    "obs_groups": {"actor": ("actor",)},
    "resume": False,
  }
  manifest = build_effective_training_manifest(env, train_cfg)
  repeated = build_effective_training_manifest(env, train_cfg)
  assert manifest == repeated
  probe = manifest["record"]["managers"]["observations"]["groups"]["actor"]
  effective_gain = probe["terms"]["probe"]["effective_parameters"]["gain"]
  assert effective_gain == {"source": "default", "value": 2.0}

  # The payload decides, not the stored digest: a stale or forged digest can no
  # longer mask a real change, nor invent one.
  evaluation = json.loads(json.dumps(manifest))
  evaluation["training_sha256"] = "different-training-runtime"
  for full in (False, True):
    validate_effective_training_manifest(manifest, evaluation, full_resume=full)
  evaluation["training_contract"]["runtime"] = {"changed": True}
  validate_effective_training_manifest(manifest, evaluation, full_resume=False)
  try:
    validate_effective_training_manifest(manifest, evaluation, full_resume=True)
  except RuntimeError:
    pass
  else:
    raise AssertionError("full resume accepted a changed training contract")


def test_resume_curriculum_synchronization() -> None:
  """Check restored counters immediately select the matching curriculum stage."""
  env = _ManifestEnv()
  snapshot = synchronize_resumed_curriculum(
    env, {"env_state": {"common_step_counter": 48_000}}
  )
  assert env.curriculum_manager.compute_calls == 1
  assert env.reward_manager.cfg["probe"].weight == -0.5
  assert snapshot["curriculum_state"]["torque_margin"] == 1
  env.common_step_counter = 0
  try:
    synchronize_resumed_curriculum(env, {"env_state": {"common_step_counter": 48_000}})
  except RuntimeError:
    pass
  else:
    raise AssertionError("resume accepted an unrestored global counter")


def test_scope_cleanup_callable_migrations_preserve_supported_contracts() -> None:
  """Moved manager terms keep loading while meaningful parameter changes fail."""
  from copy import deepcopy

  from mc_mjlab.rl.effective_training_manifest import canonicalize
  from mc_mjlab.tasks.residual_mpc import mdp

  for old, func in (
    (
      "mc_mjlab.tasks.residual_mpc.mdp:linear_velocity_tracking",
      mdp.rewards.linear_velocity_tracking,
    ),
    (
      "mc_mjlab.tasks.residual_mpc.mdp:initial_velocity_kick",
      mdp.disturbances.initial_velocity_kick,
    ),
    (
      "mc_mjlab.tasks.residual_mpc.mdp:maximum_effort_ratio",
      mdp.metrics.active_effort_ratio,
    ),
  ):
    contract = {"term": canonicalize(func), "weight": 2.0}
    active = {
      "schema_version": 1,
      "training_contract": contract,
      "policy_interface": contract,
    }
    saved = deepcopy(active)
    for key in ("training_contract", "policy_interface"):
      saved[key]["term"]["__callable__"]["name"] = old
    for full_resume in (False, True):
      validate_effective_training_manifest(saved, active, full_resume=full_resume)
      changed = deepcopy(active)
      changed["training_contract" if full_resume else "policy_interface"]["weight"] = (
        3.0
      )
      with pytest.raises(RuntimeError, match="differs"):
        validate_effective_training_manifest(saved, changed, full_resume=full_resume)


def test_impulse_migration_ignores_only_removed_inert_defaults() -> None:
  """Constructor parameters remain enforced when unused call defaults disappear."""
  from copy import deepcopy

  from mc_mjlab.mdp.disturbances import finite_impulse_curriculum
  from mc_mjlab.rl.effective_training_manifest import _effective_parameters

  func = object.__new__(finite_impulse_curriculum)
  params = {
    "stages": ((0, (0.10, 0.25)),),
    "interval_range_s": (5.0, 7.0),
    "warmup_s": 10.0,
    "duration_range_s": (0.08, 0.20),
    "height_range_m": (0.0, 0.25),
  }
  active = {
    "schema_version": 1,
    "training_contract": {
      "term": {
        "callable": {"name": "mc_mjlab.mdp.disturbances:finite_impulse_curriculum"},
        "effective_parameters": _effective_parameters(func, params),
      }
    },
  }
  saved = deepcopy(active)
  defaults = saved["training_contract"]["term"]["effective_parameters"]
  defaults.update(
    {
      name: {"source": "default", "value": value}
      for name, value in {
        "asset_cfg": None,
        "rehearsal_weights": None,
        "initial_stage": 0,
        "bands": None,
        "band_weights": None,
      }.items()
    }
  )
  validate_effective_training_manifest(saved, active, full_resume=True)
  defaults["initial_stage"] = {"source": "explicit", "value": 0}
  with pytest.raises(RuntimeError, match="initial_stage"):
    validate_effective_training_manifest(saved, active, full_resume=True)
  defaults["initial_stage"] = {"source": "default", "value": 1}
  with pytest.raises(RuntimeError, match="initial_stage"):
    validate_effective_training_manifest(saved, active, full_resume=True)
  defaults.pop("initial_stage")
  defaults["warmup_s"]["value"] = 5.0
  with pytest.raises(RuntimeError, match="warmup_s"):
    validate_effective_training_manifest(saved, active, full_resume=True)
