"""Scenario isolation, rotation, correlated error and partial state resets."""

from __future__ import annotations

import math
from dataclasses import fields

import pytest
import torch
from evaluation.study_scenarios import (
  CorrelatedNoise,
  CorrelatedNoiseModelCfg,
  RotationNoiseCfg,
  Scenario,
  scenarios,
)


def test_grid_is_the_planned_one_mechanism_per_entry() -> None:
  grid = scenarios()
  assert len(grid) == 3 + 12 + 8 + 4 + 2 + 3 + 2 + 2
  default = Scenario("default")
  changed = {
    name: [
      f.name
      for f in fields(Scenario)
      if f.name != "name" and getattr(scenario, f.name) != getattr(default, f.name)
    ]
    for name, scenario in grid.items()
  }
  combined = {"encoder-bias+white-noise"}
  assert all(len(keys) == 1 for name, keys in changed.items() if name not in combined)
  assert grid["actuator-delay/10ms"].actuator_delay_ms == 10
  assert grid["velocity-bias/y/-0.05"].velocity_bias == (-0.0, -0.05, -0.0)


def test_mounting_rotation_preserves_norm_and_tilts_gravity() -> None:
  gravity = torch.tensor([[0.0, 0.0, -1.0]])
  rotated = RotationNoiseCfg(roll_pitch_deg=(0.0, 2.0)).apply(gravity)
  assert torch.allclose(rotated.norm(dim=1), torch.ones(1))
  assert rotated[0, 0].item() == pytest.approx(-math.sin(math.radians(2.0)))


def test_correlated_error_matches_rms_and_resets_only_selected_rows() -> None:
  torch.manual_seed(0)
  cfg = CorrelatedNoiseModelCfg(rms=0.05, correlation_s=0.1, step_dt=0.02)
  model = CorrelatedNoise(cfg, num_envs=4000, device="cpu")
  data = torch.zeros(4000, 3)
  samples = torch.stack([model(data) for _ in range(200)])
  assert samples[50:].std().item() == pytest.approx(0.05, rel=0.05)
  lag = (samples[100:-1] * samples[101:]).mean() / samples[100:].var()
  assert lag.item() == pytest.approx(math.exp(-0.2), abs=0.03)

  before = model(data).clone()
  model.reset(torch.tensor([0]))
  after = model._state
  assert after is not None and torch.equal(after[1:], before[1:])
