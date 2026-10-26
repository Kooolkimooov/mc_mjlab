"""Scenario isolation, rotation, correlated error and partial state resets."""

from __future__ import annotations

import math
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
from evaluation.study_noise import KeyedNoise, KeyedNoiseCfg
from evaluation.study_scenarios import (
  RotationNoiseCfg,
  Scenario,
  scenarios,
)
from mjlab.utils.noise import ConstantNoiseCfg, UniformNoiseCfg


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
  cfg = KeyedNoiseCfg(
    noise_cfg=ConstantNoiseCfg(),
    input_ids=tuple(str(i) for i in range(400)),
    channel="velocity",
    samples=201,
    rms=0.05,
    correlation_s=0.1,
    step_dt=0.02,
  )
  model = KeyedNoise(cfg, num_envs=400, device="cpu")
  env = SimpleNamespace(episode_length_buf=torch.zeros(400, dtype=torch.long))
  model.bind(env)
  data = torch.zeros(400, 3)
  frames = []
  for tick in range(200):
    env.episode_length_buf[:] = tick
    frames.append(model(data))
  samples = torch.stack(frames)
  assert samples[50:].std().item() == pytest.approx(0.05, rel=0.05)
  lag = (samples[100:-1] * samples[101:]).mean() / samples[100:].var()
  assert lag.item() == pytest.approx(math.exp(-0.2), abs=0.03)

  before = model(data).clone()
  model.reset(torch.tensor([0]))
  env.episode_length_buf[0] = 0
  after = model(data)
  assert torch.equal(after[1:], before[1:])
  assert torch.equal(after[0], samples[0, 0])


@pytest.mark.parametrize("correlated", [False, True])
def test_noise_matches_across_batching_retries_and_read_counts(
  correlated: bool,
) -> None:
  def sample(ids: tuple[str, ...], ticks: list[int]) -> torch.Tensor:
    cfg = KeyedNoiseCfg(
      input_ids=ids,
      channel="velocity",
      samples=10,
      step_dt=0.02,
      noise_cfg=ConstantNoiseCfg()
      if correlated
      else UniformNoiseCfg(n_min=-0.02, n_max=0.02),
      rms=0.05,
      correlation_s=0.1 if correlated else None,
    )
    model = KeyedNoise(cfg, len(ids), "cpu")
    model.bind(SimpleNamespace(episode_length_buf=torch.tensor(ticks)))
    data = torch.zeros(len(ids), 3)
    first = model(data)
    torch.rand(1000)
    assert torch.equal(first, model(data))
    return first

  forward = sample(("a", "b"), [3, 7])
  assert torch.equal(forward, sample(("b", "a"), [7, 3]).flip(0))
  assert torch.equal(forward[1:], sample(("b",), [7]))
