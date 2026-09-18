"""Deterministic balance curricula contracts."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any, get_args, get_type_hints

import pytest
import torch
from conftest import requires_controller
from mjlab.managers.reward_manager import RewardTermCfg

from mc_mjlab.mdp.curricula import episode_length_impulse_curriculum
from mc_mjlab.mdp.disturbances import (
  achievement_finite_impulse_curriculum,
  finite_impulse_curriculum,
  stratified_finite_impulse_curriculum,
)
from mc_mjlab.mdp.metrics import (
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
from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import (
  AUTHORITY_SETS,
  MC_RTC_YAML,
  QUALIFICATION_MATCHED_BANDS,
  QUALIFICATION_MATCHED_MIXTURES,
  QUALIFICATION_MATCHED_WEIGHTS,
  TORQUE_MARGIN_WEIGHT,
  make_residual_balance_env_cfg,
  residual_balance_position_achievement_curriculum_env_cfg,
  residual_balance_position_env_cfg,
  residual_balance_position_matched_impulse_env_cfg,
)
from mc_mjlab.tasks.residual_balance.residual_balance_ppo_cfg import (
  POLICY_STEPS_PER_ENV,
)

pytestmark = requires_controller(MC_RTC_YAML)


class _TermManager:
  """Expose the ordinary manager contract to manifest assertions."""

  def __init__(self, terms: dict[str, RewardTermCfg] | None = None) -> None:
    self.cfg = terms or {}
    self.active_terms = list(self.cfg)

  def get_term_cfg(self, name: str) -> RewardTermCfg:
    """Return one configured fake term."""
    return self.cfg[name]


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
    self.event_manager = _TermManager(
      {"push_robot": RewardTermCfg(func=self.term, weight=1.0)}
    )
    self.curriculum_manager = _StageCurriculumManager()


class _AchievementRunner:
  """Provide the runner surface used by the curriculum bridge."""

  def __init__(self) -> None:
    self.env = _AchievementEnv()
    self.is_distributed = False


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


class _LadderEnv(_StratifiedEnv):
  """Add the terminal-length buffer the episode-length ladder reads."""

  def __init__(self) -> None:
    super().__init__()
    self.max_episode_length = 4500
    self.episode_length_buf = torch.zeros(4, dtype=torch.long)
    self.event_manager: Any = _TermManager()


def _ladder(env: _LadderEnv, push: Any) -> Any:
  """Build the ladder term over an already-built stratified impulse term."""
  ladder_cfg = RewardTermCfg(
    func=None,
    params={
      "mixtures": [
        QUALIFICATION_MATCHED_MIXTURES["gait"],
        QUALIFICATION_MATCHED_MIXTURES["matched"],
        QUALIFICATION_MATCHED_MIXTURES["hazard"],
      ],
      "term_name": "push_robot",
    },
    weight=1.0,
  )
  env.event_manager = _TermManager({"push_robot": RewardTermCfg(func=push, weight=1.0)})
  return episode_length_impulse_curriculum(ladder_cfg, env)  # ty: ignore


def test_episode_length_ladder() -> None:
  """Check the ladder advances, holds inside the deadband, and steps back."""
  env = _LadderEnv()
  original_init = finite_impulse_curriculum.__init__
  finite_impulse_curriculum.__init__ = lambda self, cfg, env: None
  try:
    push = stratified_finite_impulse_curriculum(
      RewardTermCfg(
        func=None,
        params={
          "bands": QUALIFICATION_MATCHED_BANDS,
          "band_weights": QUALIFICATION_MATCHED_WEIGHTS,
          "stages": ((0, (0.10, 0.60)),),
        },
        weight=1.0,
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


def test_curriculum_reachability() -> None:
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


def test_stratified_impulse() -> None:
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
  # Ankle roll degraded every lateral recovery; the ablation must drop it.
  from mc_mjlab.robots import robot_module as robot_cfg

  pitch = residual_balance_position_matched_impulse_env_cfg(
    authority_set="ankle_pitch"
  ).actions["mc_rtc_residual"]
  ankle = residual_balance_position_matched_impulse_env_cfg().actions["mc_rtc_residual"]
  pitch_joints = set(pitch.residual_actuator_names)  # ty: ignore[unresolved-attribute]
  roll = {
    joint
    for side in ("left", "right")
    for joint in robot_cfg.get_leg_joints("HRP5P", side)[-1:]
  }
  # The qualifier must be able to evaluate every set the env cfg can build.
  hints = get_type_hints(residual_balance_position_env_cfg)
  declared = get_args(hints["authority_set"])
  assert set(declared) == set(AUTHORITY_SETS), (declared, AUTHORITY_SETS)
  assert pitch_joints, "the ankle-pitch authority set is empty"
  assert not pitch_joints & roll, sorted(pitch_joints & roll)
  ankle_joints = set(ankle.residual_actuator_names)  # ty: ignore[unresolved-attribute]
  assert pitch_joints < ankle_joints
  assert ankle_joints & roll == roll

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
        stratified_finite_impulse_curriculum(
          RewardTermCfg(func=None, params=params, weight=1.0), env
        )
      except ValueError:
        continue
      raise AssertionError(f"accepted invalid band configuration {params}")
    params = {
      "bands": QUALIFICATION_MATCHED_BANDS,
      "band_weights": QUALIFICATION_MATCHED_WEIGHTS,
      "stages": ((0, (0.10, 0.60)),),
    }
    term = stratified_finite_impulse_curriculum(
      RewardTermCfg(func=None, params=params, weight=1.0), env
    )
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


def test_achievement_curriculum() -> None:
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


def test_achievement_report_contract() -> None:
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


def test_achievement_runner_bridge() -> None:
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


def test_environment_variants() -> None:
  """Check deployable observations and staged-randomization configuration."""
  standard = make_residual_balance_env_cfg("position", num_envs=1, disturbance="none")
  actor = standard.observations["actor"].terms
  critic = standard.observations["critic"].terms
  for name in ("left_foot_lin_vel", "right_foot_lin_vel"):
    assert name not in actor and name in critic
  robust = make_residual_balance_env_cfg(
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


def test_retired_options_are_rejected() -> None:
  """Only supported authority and robustness variants can be built."""
  assert AUTHORITY_SETS == ("uniform", "ankle", "ankle_pitch")
  # Every value here is outside the declared Literal on purpose: the assertion
  # is that the builder rejects it at runtime, not just in the type checker.
  retired: tuple[dict[str, Any], ...] = (
    {"authority_set": "sagittal"},
    {"authority_set": "hardware"},
    {"randomization_stage": 2},
  )
  for kwargs in retired:
    with pytest.raises(ValueError):
      make_residual_balance_env_cfg("position", **kwargs)
  with pytest.raises(TypeError):
    # The point of the assertion: the retired dial is gone from the signature.
    make_residual_balance_env_cfg("position", controller_history=10)  # ty: ignore[unknown-argument]


def test_impulse_constructor_keys_do_not_hide_typos() -> None:
  """Constructor-only keys are accepted only by the variant that owns them."""
  term = object.__new__(finite_impulse_curriculum)
  with pytest.raises(TypeError, match="bands"):
    term(None, None, (0.08, 0.20), (0.0, 0.25), ((0, (0.1, 0.2)),), bands=())
