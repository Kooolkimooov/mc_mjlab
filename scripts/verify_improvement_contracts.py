"""Assert bounded-policy and residual-safety contracts without running simulation."""

from __future__ import annotations

import json
import math
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

# Sibling script, resolved by the interpreter's script directory at runtime.
from qualify_checkpoints import (  # ty: ignore[unresolved-import]
  Episode,
  _cluster_stats,
  _t_critical,
  clusters_for_confidence,
  promotion,
  summarize,
)
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from mc_mjlab.recovery_authority import (
  RecoveryCalibration,
  RecoveryFilter,
  detector_target,
)
from mc_mjlab.residual_safety import project_residual
from mc_mjlab.tasks.mdp import (
  achievement_finite_impulse_curriculum,
  action_l2,
  episode_length_impulse_curriculum,
  finite_impulse_curriculum,
  gradual_finite_impulse_curriculum,
  interpolated_impulse_range,
  requested_action_l2,
  requested_action_rate_l2,
  stratified_finite_impulse_curriculum,
)
from mc_mjlab.tasks.mdp import (
  recovery_authority_coverage as mdp_recovery_authority_coverage,
)
from mc_mjlab.tasks.residual_balance.achievement_curriculum import (
  AchievementCurriculumBridge,
  AchievementState,
  QualificationEvidence,
  apply_qualification,
  read_qualification_evidence,
)
from mc_mjlab.tasks.residual_balance.curriculum_stages import (
  ACHIEVEMENT_STAGES,
  REQUIRED_PASS_REPORTS,
  REQUIRED_QUALIFICATION_SCENARIOS,
  REQUIRED_REGRESSIONS,
  achievement_contract,
  achievement_contract_sha256,
)
from mc_mjlab.tasks.residual_balance.effective_training_manifest import (
  build_effective_training_manifest,
  synchronize_resumed_curriculum,
  validate_effective_training_manifest,
)
from mc_mjlab.tasks.residual_balance.qualification_strata import classify_strata
from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import (
  QUALIFICATION_MATCHED_BANDS,
  QUALIFICATION_MATCHED_MIXTURES,
  QUALIFICATION_MATCHED_WEIGHTS,
  TORQUE_MARGIN_WEIGHT,
  _make_env_cfg,
  residual_balance_position_achievement_curriculum_env_cfg,
  residual_balance_position_curriculum_env_cfg,
  residual_balance_position_env_cfg,
  residual_balance_position_matched_impulse_env_cfg,
)
from mc_mjlab.tasks.residual_balance.residual_balance_ppo_cfg import (
  POLICY_STEPS_PER_ENV,
)
from mc_mjlab.tasks.residual_balance.reward_audit import (
  RewardAuditRecorder,
  RewardAuditShapeError,
)
from mc_mjlab.tasks.residual_balance.training_watchdog import (
  HealthObservation,
  WatchdogThresholds,
  decide_health,
  intervention_for,
  qualification_baseline,
  qualification_regressions,
)
from mc_mjlab.tasks.rollout_adaptive_ppo import (
  RolloutAdaptivePPO,
  normalize_masked_advantages,
)
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


class _AchievementTerm:
  """Expose the runner bridge's marked disturbance protocol."""

  is_achievement_curriculum = True

  def __init__(self) -> None:
    self.current_stage = 0

  def set_stage(self, stage: int) -> None:
    """Record the selected fake stage."""
    self.current_stage = stage


class _StageCurriculumManager:
  """Count bridge-driven manager-state refreshes."""

  def __init__(self) -> None:
    self.compute_calls = 0

  def compute(self) -> None:
    """Record a fake manager refresh."""
    self.compute_calls += 1


class _AchievementEnv:
  """Provide the environment surface used by the runner bridge."""

  def __init__(self) -> None:
    self.unwrapped = self
    self.term = _AchievementTerm()
    self.event_manager = _TermManager({"push_robot": _TermCfg(func=self.term)})
    self.curriculum_manager = _StageCurriculumManager()


class _AchievementRunner:
  """Provide the runner surface used by the curriculum bridge."""

  def __init__(self) -> None:
    self.env = _AchievementEnv()
    self.is_distributed = False


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


class _RequestPricingEnv:
  """Expose only the action-manager lookup the request-pricing rewards perform."""

  def __init__(self, term: Any) -> None:
    self.action_manager = _RequestPricingActions(term)


class _RequestPricingActions:
  """Return one residual term for every action name."""

  def __init__(self, term: Any) -> None:
    self.term = term

  def get_term(self, name: str) -> Any:
    """Return the single fake residual term."""
    del name
    return self.term


def _request_pricing_env(
  request: torch.Tensor,
  previous_request: torch.Tensor,
  gate: torch.Tensor,
  previous_gate: torch.Tensor,
) -> _RequestPricingEnv:
  """Build the buffer surface the request-pricing rewards read."""
  # Imported here: `mc_mjlab.actions.__init__` pulls the subclasses, so a
  # module-level import of the base cycles through a partial package.
  from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase

  # `_residual_term` isinstance-checks, so the double must be the real class.
  term = object.__new__(McRtcResidualActionBase)
  term._residual_raw_actions = request
  term._previous_residual_raw_actions = previous_request
  term._last_gate = gate
  term._previous_gate = previous_gate
  return _RequestPricingEnv(term)


def verify_request_pricing() -> None:
  """Check burst onset is not charged the magnitude cost a second time."""
  request = torch.tensor([[0.3, -0.4], [0.3, -0.4], [0.3, -0.4], [0.3, -0.4]])
  previous = torch.tensor([[0.0, 0.0], [0.1, -0.2], [0.3, -0.4], [0.1, -0.2]])
  gate = torch.tensor([1.0, 1.0, 0.0, 0.4])
  previous_gate = torch.tensor([0.0, 1.0, 1.0, 1.0])
  env = _request_pricing_env(request, previous, gate, previous_gate)
  magnitude = requested_action_l2(env)  # ty: ignore[invalid-argument-type]
  rate = requested_action_rate_l2(env)  # ty: ignore[invalid-argument-type]
  assert torch.allclose(magnitude, torch.tensor([0.25, 0.25, 0.0, 0.25]))
  assert torch.allclose(rate, torch.tensor([0.0, 0.08, 0.0, 0.08]))
  assert float(rate[0]) == 0.0 and float(magnitude[0]) > 0.0


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


def verify_masked_policy_objective() -> None:
  """Check inactive samples have zero surrogate gradient without dilution."""
  advantages = torch.tensor([1.0, 3.0, 100.0, 200.0]).reshape(2, 2, 1)
  mask = torch.tensor([True, True, False, False]).reshape(2, 2, 1)
  normalized = normalize_masked_advantages(advantages, mask)
  assert torch.equal(normalized[~mask], torch.zeros(2))
  assert torch.isclose(normalized[mask].mean(), torch.tensor(0.0), atol=1.0e-6)
  ratio = torch.ones_like(normalized, requires_grad=True)
  (normalized * ratio).mean().backward()
  assert ratio.grad is not None
  assert torch.equal(ratio.grad[~mask], torch.zeros(2))
  expected = (advantages[mask] - advantages[mask].mean()) / advantages[mask].std()
  assert torch.allclose(ratio.grad[mask], expected / mask.sum())
  assert torch.equal(
    normalize_masked_advantages(advantages, torch.zeros_like(mask)),
    torch.zeros_like(advantages),
  )

  cfg = _make_env_cfg("position")
  assert cfg.rewards["residual_magnitude"].func is requested_action_l2
  assert cfg.rewards["residual_rate"].func is requested_action_rate_l2
  assert cfg.metrics["executed_residual_l2"].func is action_l2
  assert cfg.metrics["requested_residual_l2"].func is requested_action_l2

  rollout_obs = TensorDict(
    {"actor": torch.randn(4, 3), "critic": torch.randn(4, 3)}, batch_size=[4]
  )
  obs_groups = {"actor": ["actor"], "critic": ["critic"]}
  distribution_cfg = {
    "class_name": "mc_mjlab.tasks.squashed_gaussian:SquashedGaussianDistribution",
    "init_std": 0.1,
    "std_range": (0.05, 0.30),
  }
  actor = ZeroInitMLPModel(
    rollout_obs,
    obs_groups,
    "actor",
    2,
    hidden_dims=(8,),
    distribution_cfg=distribution_cfg,
  )
  critic = MLPModel(rollout_obs, obs_groups, "critic", 1, hidden_dims=(8,))
  storage = RolloutStorage("rl", 4, 3, rollout_obs, [2])
  algorithm = RolloutAdaptivePPO(
    actor,
    critic,
    storage,
    num_learning_epochs=1,
    num_mini_batches=1,
    schedule="fixed",
  )
  authority = torch.zeros(4)
  algorithm.set_actor_update_mask_source(lambda: authority)
  masks = torch.tensor(
    [
      [True, False, False, True],
      [False, True, False, False],
      [True, False, False, False],
    ]
  )
  for mask_row in masks:
    algorithm.act(rollout_obs)
    authority.copy_(mask_row)
    algorithm.process_env_step(
      rollout_obs, torch.randn(4), torch.zeros(4, dtype=torch.bool), {}
    )
  algorithm.compute_returns(rollout_obs)
  assert torch.equal(storage.advantages[~masks.unsqueeze(-1)], torch.zeros(8))
  losses = algorithm.update()
  assert all(math.isfinite(value) for value in losses.values())
  assert storage.step == 0
  assert math.isclose(algorithm.last_actor_update_fraction, 4 / 12, rel_tol=1.0e-6)


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


def _qualification(
  token: str, stage: int, eligible: bool, iteration: int
) -> QualificationEvidence:
  """Build compact deterministic evidence for curriculum assertions."""
  return QualificationEvidence(
    token=token,
    report_sha256=f"sha-{token}",
    stage=stage,
    checkpoint=f"/run/model_{iteration}.pt",
    checkpoint_iteration=iteration,
    seeds=(42, 43),
    eligible=eligible,
    reasons=() if eligible else ("gate failed",),
  )


class _StratifiedEnv:
  """Provide the fields the stratified impulse term reads."""

  def __init__(self) -> None:
    self.num_envs = 4
    self.device = "cpu"
    self.episode_length_buf = torch.ones(4, dtype=torch.long)


def _recovery_summary(mean: float, sem: float, clusters: float) -> dict:
  """Build the one qualifier summary the recovery gate reads."""
  return {
    "finite_impulse": {
      "recovery_dcm_error": {
        "baseline": 0.131,
        "relative": mean / 0.131,
        "paired": {
          "mean": mean,
          "sem": sem,
          "ci_high": mean + _t_critical(int(clusters)) * sem,
          "clusters": clusters,
        },
      },
      "hazard": {"baseline": 0.09375, "policy": 0.0625},
      "worker_failure": {"baseline": 0.0, "policy": 0.0},
      "max_effort_ratio": {"policy": 0.5},
      "projection_fraction": {"policy": 0.0},
      "near_bound_fraction": {"policy": 0.0},
      "residual_rms": {"policy": 0.01},
    }
  }


def _paired_episode(seed: int, env_id: int, arm: str, recovery: float) -> Episode:
  """Build one qualifier episode carrying only the summarized metrics."""
  return Episode(
    checkpoint="model.pt",
    scenario="finite_impulse",
    seed=seed,
    env_id=env_id,
    pair=0,
    arm=arm,
    length=100,
    terminations={"controller_worker_failed": 0},
    rewards={},
    metrics={
      "recovery_dcm_error": recovery,
      "recovery_active": 1.0,
      "hazard": 0.0,
      "dcm_error": recovery,
      "com_velocity_error": 0.1,
      "zmp_error": 0.03,
      "foot_slip": 0.0,
      "projection_fraction": 0.0,
      "near_bound_fraction": 0.0,
      "max_effort_ratio": 0.5,
      "gate_mean": 0.02,
      "executed_residual_l2": 1.0e-4,
    },
  )


def verify_paired_clustering() -> None:
  """Check a second seed adds clusters instead of collapsing them to two."""
  episodes = []
  for seed in (42, 43):
    for env_id in range(16):
      baseline = 0.13 + 0.002 * env_id
      episodes.append(_paired_episode(seed, env_id, "baseline", baseline))
      episodes.append(_paired_episode(seed, env_id, "policy", baseline - 0.010))
  both = summarize(episodes)["recovery_dcm_error"]
  assert both["cluster_level"] == "seed-environment"
  assert both["paired"]["clusters"] == 32.0
  assert both["paired"]["ci_high"] < 0.0
  one = summarize([e for e in episodes if e.seed == 42])["recovery_dcm_error"]
  assert one["paired"]["clusters"] == 16.0


class _LadderEnv(_StratifiedEnv):
  """Add the terminal-length buffer the episode-length ladder reads."""

  def __init__(self) -> None:
    super().__init__()
    self.max_episode_length = 4500
    self.episode_length_buf = torch.zeros(4, dtype=torch.long)
    self.event_manager: Any = _TermManager()


def _ladder(env: _LadderEnv, push: Any) -> Any:
  """Build the ladder term over an already-built stratified impulse term."""
  ladder_cfg = _TermCfg(
    func=None,
    params={
      "mixtures": [
        QUALIFICATION_MATCHED_MIXTURES["gait"],
        QUALIFICATION_MATCHED_MIXTURES["matched"],
        QUALIFICATION_MATCHED_MIXTURES["hazard"],
      ],
      "term_name": "push_robot",
    },
  )
  env.event_manager = _TermManager({"push_robot": _TermCfg(func=push)})
  return episode_length_impulse_curriculum(ladder_cfg, env)  # ty: ignore


def verify_episode_length_ladder() -> None:
  """Check the ladder advances, holds inside the deadband, and steps back."""
  env = _LadderEnv()
  original_init = finite_impulse_curriculum.__init__
  finite_impulse_curriculum.__init__ = lambda self, cfg, env: None
  try:
    push = stratified_finite_impulse_curriculum(
      _TermCfg(
        func=None,
        params={
          "bands": QUALIFICATION_MATCHED_BANDS,
          "band_weights": QUALIFICATION_MATCHED_WEIGHTS,
          "stages": ((0, (0.10, 0.60)),),
        },
      ),
      env,  # ty: ignore[invalid-argument-type]
    )
    push._env = env  # ty: ignore[invalid-assignment]
    ladder = _ladder(env, push)
  finally:
    finite_impulse_curriculum.__init__ = original_init

  # Construction pins the easiest mixture regardless of the built weights.
  assert ladder.stage == 0
  assert push.band_weights == QUALIFICATION_MATCHED_MIXTURES["gait"]

  ids = torch.arange(4)
  params = {"mixtures": (), "smoothing": 1.0}

  # A cohort surviving 80% of the cap advances one rung, and only one.
  env.episode_length_buf = torch.full((4,), 3600, dtype=torch.long)
  state = ladder(env, ids, **params)
  assert ladder.stage == 1 and state["stage"] == 1.0
  assert push.band_weights == QUALIFICATION_MATCHED_MIXTURES["matched"]
  # Smoothing restarts, so the same evidence cannot advance twice in a row.
  assert math.isnan(ladder.smoothed)

  # Inside the deadband nothing moves.
  env.episode_length_buf = torch.full((4,), 2400, dtype=torch.long)
  ladder(env, ids, **params)
  assert ladder.stage == 1

  # Below the regress fraction it steps back down, and never past the floor.
  env.episode_length_buf = torch.full((4,), 900, dtype=torch.long)
  ladder(env, ids, **params)
  assert ladder.stage == 0
  ladder(env, ids, **params)
  assert ladder.stage == 0

  # An empty or sliced reset cohort is not evidence.
  before = ladder.smoothed
  ladder(env, torch.zeros(0, dtype=torch.long), **params)
  ladder(env, None, **params)
  assert ladder.smoothed is before or math.isnan(ladder.smoothed)
  try:
    ladder(env, ids, mixtures=(), advance_fraction=0.4, regress_fraction=0.5)
  except ValueError:
    pass
  else:
    raise AssertionError("ladder accepted an inverted deadband")


def verify_curriculum_reachability() -> None:
  """Check every reward-curriculum stage is reached inside the step budget."""
  variants = {
    "position": residual_balance_position_env_cfg(),
    "matched": residual_balance_position_matched_impulse_env_cfg(),
  }
  seen = 0
  for name, cfg in variants.items():
    for term_name, term in cfg.curriculum.items():
      stages = term.params.get("stages")
      if not stages:
        continue
      seen += 1
      last = max(int(stage["step"]) for stage in stages)
      assert last <= POLICY_STEPS_PER_ENV, f"{name}.{term_name} stage {last}"
  assert seen, "no staged reward curriculum was checked"


def verify_qualifier_power() -> None:
  """Check the Student-t interval, the one-cluster hole, and power reporting."""
  assert _t_critical(1) == math.inf
  assert math.isclose(_t_critical(2), 12.7062)
  assert math.isclose(_t_critical(16), 2.1314)
  assert all(_t_critical(count) > _t_critical(count + 1) for count in range(2, 60))
  assert _t_critical(400) > 1.959963985

  single = _cluster_stats([-0.01])
  assert single["sem"] == math.inf and single["ci_high"] == math.inf
  assert single["clusters"] == 1.0
  assert not single["ci_high"] < 0.0

  # The 16-environment paired rejection of `standard/model_180`.
  assert clusters_for_confidence(-0.01010, 0.005760, 16) == 23.0
  assert math.isnan(clusters_for_confidence(0.01, 0.005, 16))
  assert math.isnan(clusters_for_confidence(-0.01, 0.005, 1))

  # An invalidated scenario must not also produce a substantive policy verdict.
  broken = _recovery_summary(-0.00131, 0.0002, 16.0)
  broken["finite_impulse"]["worker_failure"] = {"baseline": 0.0, "policy": 1.0}
  verdict = promotion(broken)
  assert not verdict["eligible"]
  assert verdict["invalidated_scenarios"] == ["finite_impulse"]
  assert any("invalidated the run" in reason for reason in verdict["reasons"])
  assert any("unreadable" in reason for reason in verdict["reasons"])
  assert not any("below 5%" in reason for reason in verdict["reasons"])
  assert not any("hazard ratio" in reason for reason in verdict["reasons"])

  unresolved = promotion(_recovery_summary(-0.01010, 0.005760, 16.0))
  assert unresolved["invalidated_scenarios"] == []
  assert not unresolved["eligible"]
  assert any("unresolved by 16 clusters" in reason for reason in unresolved["reasons"])
  assert any("23 would resolve it" in reason for reason in unresolved["reasons"])
  small = promotion(_recovery_summary(-0.00131, 0.0002, 16.0))
  assert any("below 5%" in reason for reason in small["reasons"])
  resolved = promotion(_recovery_summary(-0.01010, 0.003, 32.0))
  assert not any("recovery DCM" in reason for reason in resolved["reasons"])


def verify_stratified_impulse() -> None:
  """Check qualifier coverage, band validation, standing exclusion, routing."""
  push = residual_balance_position_matched_impulse_env_cfg().events["push_robot"]
  assert push.func is stratified_finite_impulse_curriculum
  assert push.params["bands"] == QUALIFICATION_MATCHED_BANDS
  assert push.params["band_weights"] == QUALIFICATION_MATCHED_WEIGHTS
  assert push.params["stages"] == ((0, (0.10, 0.60)),)
  assert math.isclose(sum(QUALIFICATION_MATCHED_WEIGHTS), 1.0)
  for magnitude in (0.25, 0.40, 0.50, 0.60):
    assert any(low <= magnitude <= high for low, high in QUALIFICATION_MATCHED_BANDS)

  # Every branch of the promotion verdict must already have a valid mixture.
  assert QUALIFICATION_MATCHED_MIXTURES["matched"] == QUALIFICATION_MATCHED_WEIGHTS
  for name, weights in QUALIFICATION_MATCHED_MIXTURES.items():
    assert len(weights) == len(QUALIFICATION_MATCHED_BANDS) + 1, name
    assert math.isclose(sum(weights), 1.0), name
    assert all(value >= 0.0 for value in weights), name
    assert weights[0] > 0.0, name
  matched = QUALIFICATION_MATCHED_MIXTURES["matched"]
  gait = QUALIFICATION_MATCHED_MIXTURES["gait"]
  hazard = QUALIFICATION_MATCHED_MIXTURES["hazard"]
  assert gait[0] > matched[0] > hazard[0]
  assert gait[-1] < matched[-1] < hazard[-1]
  named = ("matched", "gait", "hazard")
  assert set(named) == set(QUALIFICATION_MATCHED_MIXTURES)
  for name in named:
    variant = residual_balance_position_matched_impulse_env_cfg(mixture=name)
    weights = variant.events["push_robot"].params["band_weights"]
    assert weights == QUALIFICATION_MATCHED_MIXTURES[name], name

  # The coverage metric is only readable as a ratio over the same window.
  metrics = residual_balance_position_matched_impulse_env_cfg().metrics
  coverage = metrics["recovery_authority_coverage"]
  assert coverage.func is mdp_recovery_authority_coverage
  for key in ("window_s", "min_normal_force", "push_term_name"):
    assert coverage.params.get(key) == metrics["recovery_active"].params.get(key), key

  env = _StratifiedEnv()
  original_init = finite_impulse_curriculum.__init__
  finite_impulse_curriculum.__init__ = lambda self, cfg, env: None
  try:
    for weights, stages in (
      ((0.2, 0.3, 0.3, 0.3), ((0, (0.10, 0.60)),)),
      ((0.5, 0.5), ((0, (0.10, 0.60)),)),
      (QUALIFICATION_MATCHED_WEIGHTS, ((0, (0.10, 0.50)),)),
      (QUALIFICATION_MATCHED_WEIGHTS, ((0, (0.10, 0.60)), (1, (0.10, 0.60)))),
    ):
      params = {
        "bands": QUALIFICATION_MATCHED_BANDS,
        "band_weights": weights,
        "stages": stages,
      }
      try:
        stratified_finite_impulse_curriculum(_TermCfg(func=None, params=params), env)
      except ValueError:
        continue
      raise AssertionError(f"accepted invalid band configuration {params}")
    params = {
      "bands": QUALIFICATION_MATCHED_BANDS,
      "band_weights": QUALIFICATION_MATCHED_WEIGHTS,
      "stages": ((0, (0.10, 0.60)),),
    }
    term = stratified_finite_impulse_curriculum(_TermCfg(func=None, params=params), env)
  finally:
    finite_impulse_curriculum.__init__ = original_init

  term.enabled = torch.ones(4, dtype=torch.bool)
  term.next_push_step = torch.zeros(4, dtype=torch.long)
  term.sampled_band = torch.tensor([-1, 0, 1, 2])
  assert torch.equal(term._due(env), torch.tensor([False, True, True, True]))

  routed: list[tuple[list[int], tuple[float, float]]] = []
  original_trigger = finite_impulse_curriculum._trigger
  finite_impulse_curriculum._trigger = (
    lambda self, env, env_ids, duration_range_s, height_range_m, stages: routed.append(
      (sorted(int(value) for value in env_ids), stages[0][1])
    )
  )
  try:
    term._trigger(env, torch.tensor([0, 1, 2, 3]), (0.08, 0.20), (0.0, 0.25), ())
  finally:
    finite_impulse_curriculum._trigger = original_trigger
  assert routed == [
    ([1], (0.10, 0.25)),
    ([2], (0.25, 0.40)),
    ([3], (0.40, 0.60)),
  ]


def verify_achievement_curriculum() -> None:
  """Check rehearsal, hysteresis, rollback, and exact resume continuity."""
  assert REQUIRED_PASS_REPORTS == 2
  assert REQUIRED_REGRESSIONS == 3
  for index, stage in enumerate(ACHIEVEMENT_STAGES):
    assert math.isclose(sum(stage.rehearsal_weights), 1.0)
    assert stage.rehearsal_weights[0] > 0.0
    assert not any(stage.rehearsal_weights[index + 2 :])
    if index:
      assert sum(stage.rehearsal_weights[1 : index + 1]) > 0.0
  assert achievement_contract_sha256(0) != achievement_contract_sha256(1)

  first = apply_qualification(AchievementState(), _qualification("pass-a", 0, True, 20))
  assert first.event == "pass_pending" and first.state.current_stage == 0
  restored = AchievementState.from_dict(first.state.to_dict())
  uninterrupted = apply_qualification(
    first.state, _qualification("pass-b", 0, True, 40)
  )
  resumed = apply_qualification(restored, _qualification("pass-b", 0, True, 40))
  assert uninterrupted == resumed
  assert resumed.event == "advanced"
  assert resumed.state.current_stage == 1
  assert resumed.state.highest_passed_stage == 0
  assert resumed.state.qualified_checkpoints[0] == "/run/model_40.pt"
  assert resumed.state.last_good_checkpoint == "/run/model_40.pt"
  assert (
    apply_qualification(resumed.state, _qualification("pass-b", 1, True, 40)).event
    == "duplicate"
  )

  state = resumed.state
  for index in range(REQUIRED_REGRESSIONS):
    decision = apply_qualification(
      state, _qualification(f"fail-{index}", 1, False, 60 + index)
    )
    state = decision.state
  assert decision.event == "rolled_back"
  assert state.current_stage == 0
  assert state.highest_passed_stage == 0
  assert state.last_good_checkpoint == "/run/model_40.pt"

  state = AchievementState(
    current_stage=2,
    highest_passed_stage=2,
    mastered=True,
    qualified_checkpoints=("stage-0.pt", "stage-1.pt", "stage-2.pt"),
    last_good_checkpoint="stage-2.pt",
  )
  for index in range(REQUIRED_REGRESSIONS):
    decision = apply_qualification(
      state, _qualification(f"hard-fail-{index}", 2, False, 80 + index)
    )
    state = decision.state
  assert state.current_stage == state.highest_passed_stage == 1
  assert state.last_good_checkpoint == "stage-1.pt"
  assert not state.mastered

  cfg = residual_balance_position_achievement_curriculum_env_cfg()
  push = cfg.events["push_robot"]
  assert push.func is achievement_finite_impulse_curriculum
  assert push.params["initial_stage"] == 0
  assert tuple(push.params["rehearsal_weights"]) == tuple(
    stage.rehearsal_weights for stage in ACHIEVEMENT_STAGES
  )
  assert set(cfg.curriculum) == {"achievement_stage"}
  assert cfg.rewards["torque_margin"].weight == TORQUE_MARGIN_WEIGHT


def verify_achievement_report_contract() -> None:
  """Check the on-disk evaluator/trainer handshake and path boundary."""
  with tempfile.TemporaryDirectory() as directory:
    run_dir = Path(directory)
    checkpoint = run_dir / "model_20.pt"
    checkpoint.write_bytes(b"checkpoint")
    stage = 0
    report = {
      "config": {
        "seeds": [42, 43],
        "scenarios": list(REQUIRED_QUALIFICATION_SCENARIOS),
        "achievement": {
          "stage": stage,
          "contract": achievement_contract(stage),
          "contract_sha256": achievement_contract_sha256(stage),
        },
      },
      "checkpoints": {
        str(checkpoint): {"promotion": {"eligible": True, "reasons": []}}
      },
    }
    path = run_dir / "qualification.json"
    path.write_text(json.dumps(report))
    evidence = read_qualification_evidence(path, stage, run_dir, 21)
    assert evidence.checkpoint == str(checkpoint)
    assert evidence.eligible
    report["config"]["seeds"] = [42]
    path.write_text(json.dumps(report))
    try:
      read_qualification_evidence(path, stage, run_dir, 21)
    except ValueError:
      pass
    else:
      raise AssertionError("achievement report accepted a single seed")


def _write_achievement_report(
  path: Path,
  checkpoint: Path,
  stage: int,
  seeds: list[int],
  eligible: bool = True,
) -> None:
  """Write one minimal valid qualifier report for bridge assertions."""
  report = {
    "config": {
      "seeds": seeds,
      "scenarios": list(REQUIRED_QUALIFICATION_SCENARIOS),
      "achievement": {
        "stage": stage,
        "contract": achievement_contract(stage),
        "contract_sha256": achievement_contract_sha256(stage),
      },
    },
    "checkpoints": {
      str(checkpoint): {
        "promotion": {
          "eligible": eligible,
          "reasons": [] if eligible else ["gate failed"],
        }
      }
    },
  }
  path.write_text(json.dumps(report))


def verify_achievement_runner_bridge() -> None:
  """Check report consumption, preservation, publication, and restored deduping."""
  with tempfile.TemporaryDirectory() as directory:
    run_dir = Path(directory)
    runner = _AchievementRunner()
    bridge = AchievementCurriculumBridge(runner, run_dir)
    report_path = run_dir / "curriculum" / "qualification.json"
    first = run_dir / "model_20.pt"
    first.write_bytes(b"first")
    _write_achievement_report(report_path, first, 0, [42, 43])
    bridge.iteration(20)
    assert bridge.state.pass_streak == 1

    second = run_dir / "model_40.pt"
    second.write_bytes(b"second")
    _write_achievement_report(report_path, second, 0, [42, 43])
    bridge.iteration(40)
    assert bridge.state.current_stage == 1
    assert runner.env.term.current_stage == 1
    preserved = run_dir / "curriculum" / "qualified_stage_0_model_40.pt"
    assert preserved.read_bytes() == b"second"
    saved = bridge.snapshot()
    assert saved is not None

    resumed_runner = _AchievementRunner()
    resumed = AchievementCurriculumBridge(resumed_runner, run_dir)
    resumed.restore(saved)
    before = resumed.state
    resumed.iteration(41)
    assert resumed.state == before
    assert resumed_runner.env.term.current_stage == 1


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


def verify_qualification_strata() -> None:
  """Check startup, sustained, and body-frame recovery direction classification."""
  episode_steps = torch.tensor([500, 600, 501, 600, 600, 600])
  push_age = torch.tensor([1 << 30, 1 << 30, 1, 50, 100, 100])
  push_velocity = torch.tensor(
    [
      [0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.4, 0.1, 0.0],
      [-0.4, 0.1, 0.0],
      [0.1, 0.4, 0.0],
      [0.1, -0.4, 0.0],
    ]
  )
  strata = classify_strata(episode_steps, push_age, push_velocity, 500, 100)
  assert strata.tolist() == [0, 1, 2, 3, 4, 5]


def verify_training_watchdog() -> None:
  """Check escalation, preserve-before-stop, and post-attach regression guards."""
  thresholds = WatchdogThresholds()
  healthy = HealthObservation(0.0, 0.0, 12_000.0, 0, 0, 0)
  assert decide_health(healthy, thresholds).level == "ok"
  worker_degraded = HealthObservation(0.0, 0.0, 12_000.0, 0, 3, 0)
  assert decide_health(worker_degraded, thresholds).level == "preserve"
  near_oom = HealthObservation(0.0, 0.0, 512.0, 2, 0, 0)
  assert decide_health(near_oom, thresholds).level == "stop"
  assert intervention_for("stop", "stop", False, False) == "preserve"
  assert intervention_for("stop", "stop", True, False) == "stop"
  assert intervention_for("stop", "warn", False, False) == "warn"
  assert intervention_for("preserve", "stop", True, False) is None

  samples = [
    {"eligible": True, "hazard_ratio": 0.8, "recovery_gain": 0.10},
    {"eligible": True, "hazard_ratio": 0.9, "recovery_gain": 0.12},
  ]
  baseline = qualification_baseline(samples)
  assert math.isclose(baseline["hazard_ratio"], 0.85)
  regressions = qualification_regressions(
    baseline,
    {"eligible": False, "hazard_ratio": 1.0, "recovery_gain": 0.04},
  )
  assert len(regressions) == 3


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
  verify_request_pricing()
  verify_projection()
  verify_recovery_detector()
  verify_rollout_schedule()
  verify_masked_policy_objective()
  verify_impulse_curricula()
  verify_episode_length_ladder()
  verify_curriculum_reachability()
  verify_qualifier_power()
  verify_paired_clustering()
  verify_stratified_impulse()
  verify_achievement_curriculum()
  verify_achievement_report_contract()
  verify_achievement_runner_bridge()
  verify_environment_variants()
  verify_effective_training_manifest()
  verify_reward_audit()
  verify_qualification_strata()
  verify_training_watchdog()
  verify_resume_curriculum_synchronization()
  print("improvement contract assertions passed")


if __name__ == "__main__":
  main()
