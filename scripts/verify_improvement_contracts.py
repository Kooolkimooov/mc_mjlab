"""Assert bounded-policy and residual-safety contracts without running simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from tensordict import TensorDict

from mc_mjlab.recovery_authority import (
  RecoveryCalibration,
  RecoveryFilter,
  detector_target,
)
from mc_mjlab.residual_safety import project_residual
from mc_mjlab.tasks.mdp import (
  gradual_finite_impulse_curriculum,
  interpolated_impulse_range,
)
from mc_mjlab.tasks.residual_balance.effective_training_manifest import (
  build_effective_training_manifest,
  synchronize_resumed_curriculum,
  validate_effective_training_manifest,
)
from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import (
  _make_env_cfg,
  residual_balance_position_curriculum_env_cfg,
)
from mc_mjlab.tasks.residual_balance.reward_audit import (
  RewardAuditRecorder,
  RewardAuditShapeError,
)
from mc_mjlab.tasks.rollout_adaptive_ppo import RolloutAdaptivePPO
from mc_mjlab.tasks.squashed_gaussian import SquashedGaussianDistribution
from mc_mjlab.tasks.zero_init_actor import (
  ZeroInitMLPModel,
  ZeroInitRNNModel,
  mean_head_magnitude,
)


def _manifest_probe(_env, gain: float = 2.0):
  """Provide a callable default for effective-manifest assertions."""
  return gain


@dataclass
class _TermCfg:
  """Provide the manager-term fields used by manifest assertions."""

  func: Any
  params: dict[str, Any] = field(default_factory=dict)
  weight: float = 1.0


@dataclass
class _ObservationGroupCfg:
  """Provide observation-group fields used by manifest assertions."""

  terms: dict[str, _TermCfg]
  concatenate_terms: bool = True
  enable_corruption: bool = True


@dataclass
class _ActionCfg:
  """Provide action fields used by manifest assertions."""

  entity_name: str = "robot"
  scale: float = 0.1
  num_workers: int = 2


class _TermManager:
  """Expose the ordinary manager contract to manifest assertions."""

  def __init__(self, terms: dict[str, _TermCfg] | None = None) -> None:
    self.cfg = terms or {}
    self.active_terms = list(self.cfg)

  def get_term_cfg(self, name: str) -> _TermCfg:
    """Return one configured fake term."""
    return self.cfg[name]


class _CountingReward:
  """Return fixed reward values while counting manager evaluations."""

  def __init__(self, values: torch.Tensor) -> None:
    self.values = values
    self.calls = 0

  def __call__(self, _env) -> torch.Tensor:
    """Return the configured per-environment values."""
    self.calls += 1
    return self.values


class _AuditRewardManager:
  """Provide the private buffers used by the live reward-audit adapter."""

  def __init__(self, terms: dict[str, _TermCfg]) -> None:
    self.num_envs = 4
    self.device = "cpu"
    self.active_terms = list(terms)
    self._term_cfgs = list(terms.values())
    self._env = object()
    self._scale_by_dt = True
    self._episode_sums = {
      name: torch.zeros(self.num_envs) for name in self.active_terms
    }
    self._reward_buf = torch.zeros(self.num_envs)
    self._step_reward = torch.zeros(self.num_envs, len(terms))

  def compute(self, dt: float) -> torch.Tensor:
    """Match mjlab's weighted, dt-scaled reward accumulation."""
    self._reward_buf.zero_()
    for index, (name, cfg) in enumerate(
      zip(self.active_terms, self._term_cfgs, strict=True)
    ):
      if cfg.weight == 0.0:
        self._step_reward[:, index].zero_()
        continue
      value = cfg.func(self._env, **cfg.params)
      scaled = torch.nan_to_num(value * cfg.weight * dt)
      self._reward_buf += scaled
      self._episode_sums[name] += scaled
      self._step_reward[:, index] = scaled / dt
    return self._reward_buf


class _CurriculumManager(_TermManager):
  """Apply a deterministic fake stage from the restored global counter."""

  def __init__(self, env) -> None:
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
    term = _TermCfg(func=_manifest_probe)
    self.cfg = {"actor": _ObservationGroupCfg(terms={"probe": term})}
    self.active_terms = {"actor": ["probe"]}
    self.group_obs_dim = {"actor": (1,)}
    self.group_obs_term_dim = {"actor": [(1,)]}

  def get_term_cfg(self, group: str, name: str) -> _TermCfg:
    """Return one configured fake observation term."""
    return self.cfg[group].terms[name]


class _ActionManager:
  """Expose action ordering and dimensions to the manifest."""

  def __init__(self) -> None:
    self.cfg = {"residual": _ActionCfg()}
    self.active_terms = ["residual"]
    self.action_term_dim = [1]
    self.total_action_dim = 1

  def get_term(self, _name: str):
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
      {"probe": _TermCfg(func=_manifest_probe, weight=-0.05)}
    )
    self.termination_manager = _TermManager()
    self.command_manager = _TermManager()
    self.event_manager = _TermManager()
    self.metrics_manager = _TermManager()
    self.curriculum_manager = _CurriculumManager(self)


def verify_distribution() -> None:
  """Check bounds, density, latent KL, scalar std, and finite gradients."""
  torch.manual_seed(7)
  distribution = SquashedGaussianDistribution(4, init_std=0.1)
  mean = torch.zeros(4096, 4, requires_grad=True)
  distribution.update(mean)
  action = distribution.sample()
  assert bool((action.abs() < 1.0).all())
  latent = torch.atanh(action)
  expected_log_prob = (
    torch.distributions.Normal(mean, distribution.std).log_prob(latent)
    - torch.log1p(-action.square())
  ).sum(-1)
  assert torch.allclose(
    distribution.log_prob(action), expected_log_prob, atol=2e-5, rtol=2e-5
  )
  old = (torch.zeros(3, 4), torch.full((3, 4), 0.1))
  new = (torch.full((3, 4), 0.2), torch.full((3, 4), 0.15))
  expected_kl = torch.distributions.kl_divergence(
    torch.distributions.Normal(*old), torch.distributions.Normal(*new)
  ).sum(-1)
  assert torch.allclose(distribution.kl_divergence(old, new), expected_kl)
  loss = -(distribution.log_prob(action).mean() + 1e-3 * distribution.entropy.mean())
  loss.backward()
  assert distribution.std_param.shape == (1,)
  assert mean.grad is not None and math.isfinite(float(mean.grad.abs().max()))


def verify_zero_initialization() -> None:
  """Check deterministic output is exactly zero after actor construction."""
  observation = TensorDict({"actor": torch.randn(32, 7)}, batch_size=[32])
  actor = ZeroInitMLPModel(
    observation,
    {"actor": ["actor"]},
    "actor",
    4,
    hidden_dims=(16, 8),
    distribution_cfg={
      "class_name": ("mc_mjlab.tasks.squashed_gaussian:SquashedGaussianDistribution"),
      "init_std": 0.1,
      "std_range": (0.05, 0.30),
    },
  )
  assert mean_head_magnitude(actor, observation) == 0.0
  recurrent = ZeroInitRNNModel(
    observation,
    {"actor": ["actor"]},
    "actor",
    4,
    hidden_dims=(16, 8),
    rnn_type="gru",
    rnn_hidden_dim=16,
    distribution_cfg={
      "class_name": ("mc_mjlab.tasks.squashed_gaussian:SquashedGaussianDistribution"),
      "init_std": 0.1,
      "std_range": (0.05, 0.30),
    },
  )
  assert mean_head_magnitude(recurrent, observation) == 0.0


def verify_projection() -> None:
  """Check target bounds and exact executed-residual accounting."""
  nominal = torch.tensor([[0.9, 1.2, -1.2, 0.0]])
  residual = torch.tensor([[0.2, -0.1, 0.1, -0.4]])
  lower = torch.full((1, 4), -1.0)
  upper = torch.full((1, 4), 1.0)
  target, executed, projected = project_residual(nominal, residual, lower, upper)
  assert bool((target >= lower).all() and (target <= upper).all())
  assert torch.allclose(executed, torch.tensor([[0.1, 0.0, 0.0, -0.4]]))
  assert torch.equal(projected, torch.tensor([[True, True, True, False]]))
  zero_target, zero_executed, _ = project_residual(
    nominal, torch.zeros_like(residual), lower, upper
  )
  assert torch.equal(zero_executed, torch.zeros_like(residual))
  assert torch.equal(zero_target, nominal.clamp(lower, upper))


def verify_recovery_detector() -> None:
  """Check monotonic scoring, sensor onset, bounded duration, and exact cutoff."""
  path = Path(__file__).parents[1] / "etc" / "recovery_detector.json"
  calibration = RecoveryCalibration.from_json(path)
  centers = torch.tensor(calibration.centers)
  scales = torch.tensor(calibration.scales)
  nominal = centers.unsqueeze(0)
  score, target = detector_target(nominal, calibration)
  assert float(score) == 0.0 and float(target) == 0.0
  for index in range(len(centers)):
    disturbed = nominal.clone()
    disturbed[:, index] += scales[index] * (
      calibration.threshold + calibration.activation_span + 0.1
    )
    raised_score, raised_target = detector_target(disturbed, calibration)
    assert float(raised_score) > float(score)
    assert float(raised_target) == 1.0

  recovery_filter = RecoveryFilter(1, "cpu", calibration)
  dt = 0.02
  for _ in range(math.ceil(calibration.rearm_s / dt) + 1):
    authority = recovery_filter.update(score, nominal[:, 1], target, dt)
  assert float(authority) == 0.0
  onset = nominal[:, 1] + calibration.onset_delta + 0.01
  authority = recovery_filter.update(onset, onset, torch.ones(1), dt)
  assert 0.0 < float(authority) < 1.0
  for _ in range(math.ceil(calibration.max_active_s / dt)):
    authority = recovery_filter.update(onset, onset, torch.ones(1), dt)
  assert float(authority) == 0.0


def verify_rollout_schedule() -> None:
  """Check one-event rate decisions retain rsl_rl's thresholds and bounds."""
  update = RolloutAdaptivePPO.next_learning_rate
  assert update(1.0e-3, 0.05, 0.02) == 1.0e-3 / 1.5
  assert update(1.0e-3, 0.005, 0.02) == 1.5e-3
  assert update(1.0e-3, 0.02, 0.02) == 1.0e-3
  assert update(1.0e-5, 0.05, 0.02) == 1.0e-5
  assert update(1.0e-2, 0.005, 0.02) == 1.0e-2


def verify_impulse_curricula() -> None:
  """Check the frozen and interpolated diagnostic schedules."""
  stages = (
    (0, (0.10, 0.25)),
    (48_000, (0.10, 0.25)),
    (80_000, (0.10, 0.40)),
    (112_000, (0.10, 0.50)),
  )
  assert interpolated_impulse_range(48_000, stages) == (0.10, 0.25)
  assert interpolated_impulse_range(64_000, stages) == (0.10, 0.325)
  assert interpolated_impulse_range(112_000, stages) == (0.10, 0.50)
  frozen = residual_balance_position_curriculum_env_cfg("frozen")
  gradual = residual_balance_position_curriculum_env_cfg("gradual")
  assert frozen.events["push_robot"].params["stages"] == ((0, (0.10, 0.25)),)
  assert gradual.events["push_robot"].func is gradual_finite_impulse_curriculum
  assert frozen.curriculum == gradual.curriculum == {}


def verify_environment_variants() -> None:
  """Check deployable observations and staged-randomization configuration."""
  standard = _make_env_cfg("position", num_envs=1, disturbance="none")
  actor = standard.observations["actor"].terms
  critic = standard.observations["critic"].terms
  for name in ("left_foot_lin_vel", "right_foot_lin_vel"):
    assert name not in actor and name in critic
  robust = _make_env_cfg(
    "position", num_envs=1, disturbance="none", randomization_stage=1
  )
  assert {
    "randomize_friction",
    "randomize_pd_gains",
    "randomize_strength",
  } <= robust.events.keys()
  assert all(
    term.delay_max_lag == 1 for term in robust.observations["actor"].terms.values()
  )
  assert all(
    term.delay_max_lag == 0 for term in robust.observations["critic"].terms.values()
  )
  assert robust.scene.entities["robot"].articulation is not None
  assert all(
    actuator.delay_max_lag == 2
    for actuator in robust.scene.entities["robot"].articulation.actuators
  )


def verify_effective_training_manifest() -> None:
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

  evaluation = dict(manifest)
  evaluation["training_sha256"] = "different-training-runtime"
  validate_effective_training_manifest(manifest, evaluation, full_resume=False)
  try:
    validate_effective_training_manifest(manifest, evaluation, full_resume=True)
  except RuntimeError:
    pass
  else:
    raise AssertionError("full resume accepted a changed training contract")


def verify_reward_audit() -> None:
  """Check single evaluation, zero-weight restoration, live weights, and shapes."""
  active = _CountingReward(torch.tensor([1.0, 2.0, 3.0, 4.0]))
  inactive = _CountingReward(torch.tensor([0.0, 0.5, 1.0, 1.5]))
  manager = _AuditRewardManager(
    {
      "active": _TermCfg(func=active, weight=-2.0),
      "inactive": _TermCfg(func=inactive, weight=0.0),
    }
  )
  original_inactive = manager._term_cfgs[1].func
  audit = RewardAuditRecorder(
    manager, {"policy_zero": [0, 1], "checkpoint": [2, 3]}, 0.02
  )
  with audit:
    reward = manager.compute(0.02).clone()
    assert torch.allclose(reward, active.values * -2.0 * 0.02)
    assert inactive.calls == 1
    assert manager._term_cfgs[1].weight == 0.0
    assert bool((manager._episode_sums["inactive"] == 0.0).all())
    assert bool((manager._step_reward[:, 1] == 0.0).all())
    audit.capture_denominator("grounded", torch.tensor([1.0, 1.0, 0.0, 0.0]))
    manager._term_cfgs[0].weight = -4.0
    manager.compute(0.02)
  assert manager._term_cfgs[1].func is original_inactive
  report = audit.report()
  checkpoint = report["arms"]["checkpoint"]
  assert checkpoint["terms"]["active"]["raw"]["mean"] == 3.5
  assert checkpoint["terms"]["inactive"]["effective_weight"]["last"] == 0.0
  assert checkpoint["terms"]["active"]["effective_weight"]["distinct"] == [
    -4.0,
    -2.0,
  ]
  assert checkpoint["conditional_denominators"]["grounded"]["mean"] == 0.0
  assert audit.issues() == []

  bad = _AuditRewardManager(
    {"bad": _TermCfg(func=lambda _env: torch.zeros(4, 1), weight=1.0)}
  )
  try:
    with RewardAuditRecorder(bad, {"all": [0, 1, 2, 3]}, 0.02):
      bad.compute(0.02)
  except RewardAuditShapeError:
    pass
  else:
    raise AssertionError("reward audit accepted a broadcastable (num_envs, 1) term")


def verify_resume_curriculum_synchronization() -> None:
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


def main() -> None:
  """Run every local improvement-contract assertion."""
  verify_distribution()
  verify_zero_initialization()
  verify_projection()
  verify_recovery_detector()
  verify_rollout_schedule()
  verify_impulse_curricula()
  verify_environment_variants()
  verify_effective_training_manifest()
  verify_reward_audit()
  verify_resume_curriculum_synchronization()
  print("improvement contract assertions passed")


if __name__ == "__main__":
  main()
