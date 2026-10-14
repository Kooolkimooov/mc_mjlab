"""The frozen safety task keeps its measured controller and training contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, cast

import pytest
import torch
from conftest import requires_controller
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import mc_mjlab.tasks  # noqa: F401
from mc_mjlab.actions.residual_feedback_action import (
  ResidualFeedbackJointPositionActionCfg,
)
from mc_mjlab.tasks.locomanip.mdp import accessors, metrics, rewards
from mc_mjlab.tasks.locomanip_safety import TASK_ID
from mc_mjlab.tasks.locomanip_safety.evaluation import MASS_GRID_KG, TRANSITION_GRID_KG
from mc_mjlab.tasks.locomanip_safety.locomanip_safety_env_cfg import (
  SAFETY_CONFIG,
  TRANSITION_MIDPOINT_KG,
)
from mc_mjlab.tasks.locomanip_safety.runner import validate_authority_report

pytestmark = requires_controller(SAFETY_CONFIG)


def test_registered_task_keeps_profile_and_ppo_budget() -> None:
  cfg = load_env_cfg(TASK_ID)
  action = cast(
    ResidualFeedbackJointPositionActionCfg, cfg.actions[accessors.ACTION_NAME]
  )
  agent = cast(RslRlOnPolicyRunnerCfg, load_rl_cfg(TASK_ID))
  assert Path(action.mc_rtc_config_path) == SAFETY_CONFIG
  assert cfg.scene.num_envs == 256
  assert MASS_GRID_KG == (10, 100, 500, 525, 550, 575, 600, 625, 650)
  assert TRANSITION_GRID_KG == MASS_GRID_KG[2:]
  assert cfg.events["cart_payload"].params["midpoint_kg"] == TRANSITION_MIDPOINT_KG
  assert "contact_dcm_stability" not in cfg.rewards
  assert cfg.rewards["is_alive"].weight == 1.0
  assert {
    "termination_penalty",
    "upright",
    "collapse_risk",
    "rapid_tilt",
    "foot_slip",
    "residual_magnitude",
    "residual_rate",
  } <= set(cfg.rewards)
  assert cfg.metrics["contact_dcm_stability"].reduce == "mean"
  assert cfg.metrics["forward_cart_lag"].reduce == "max"
  assert agent.experiment_name == TASK_ID
  assert agent.max_iterations == 1000
  assert agent.num_steps_per_env == 64
  assert agent.save_interval == 20
  assert agent.algorithm.num_mini_batches == 2
  assert agent.algorithm.num_learning_epochs == 2
  assert (
    cfg.scene.num_envs * agent.num_steps_per_env * agent.max_iterations == 16_384_000
  )


def test_forward_lag_clamps_signed_error_without_hiding_overshoot(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  signed = torch.tensor([[0.13, 0, 0], [-0.06, 0, 0]])
  monkeypatch.setattr(accessors, "object_position_error", lambda *a, **k: signed)
  env = cast(ManagerBasedRlEnv, NS())
  assert metrics.forward_cart_lag(env).tolist() == pytest.approx([0.13, 0.0])
  assert signed[:, 0].tolist() == pytest.approx([0.13, -0.06])


def test_dcm_metric_uses_raw_score_independent_of_reward_weight(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  cfg = load_env_cfg(TASK_ID)
  term_cfg = cfg.metrics["contact_dcm_stability"]
  contacts = NS(dcm_offset=lambda *a: (torch.tensor([0.0, 0.1]), None))
  ground = NS(measured_offset=lambda *a: (None, torch.tensor([21.0, 19.0])))
  monkeypatch.setattr(rewards, "contact_zmp_sensors", lambda *a: (contacts, ground))
  term = rewards.contact_dcm_stability(term_cfg, cast(ManagerBasedRlEnv, NS()))
  actual = term(cast(ManagerBasedRlEnv, NS()), **term_cfg.params)
  torch.testing.assert_close(actual, torch.tensor([1.0, 0.0]))


def test_training_gate_rejects_stale_or_missing_authority(
  tmp_path: Path,
) -> None:
  path = tmp_path / "authority.json"
  report: dict[str, Any] = {
    "profile_sha256": hashlib.sha256(SAFETY_CONFIG.read_bytes()).hexdigest(),
    "all_channels_delivered": True,
    "measurable_near_failure": True,
    "encoder_bias": False,
  }
  path.write_text(json.dumps(report))
  validate_authority_report(path)
  report["profile_sha256"] = "different"
  path.write_text(json.dumps(report))
  with pytest.raises(RuntimeError, match="different controller profile"):
    validate_authority_report(path)
  report["profile_sha256"] = hashlib.sha256(SAFETY_CONFIG.read_bytes()).hexdigest()
  report["measurable_near_failure"] = False
  path.write_text(json.dumps(report))
  with pytest.raises(RuntimeError, match="authority"):
    validate_authority_report(path)
