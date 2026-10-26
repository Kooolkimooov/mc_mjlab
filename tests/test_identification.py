"""Synthetic identification within declared tolerances, logs and calibration gates."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from mc_mjlab.identification.calibration import (
  CalibrationArtifact,
  JointCalibration,
  LagCompliance,
  load_calibration,
)
from mc_mjlab.identification.fits import (
  first_order_response,
  fit_delay_lag,
  fit_free_decay,
  fit_static_push,
  frequency_response,
  held_out_error,
  sine_probe,
  step_probe,
)
from mc_mjlab.identification.log import IdentificationLog

DT = 0.001


@pytest.mark.parametrize("field", ["delay_s", "tau_s", "stiffness_nm_per_rad"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_calibration_rejected_through_loader(
  tmp_path: Path, field: str, value: float
) -> None:
  joint = {"delay_s": 0.0, "tau_s": 0.01, "stiffness_nm_per_rad": 1000.0, field: value}
  path = tmp_path / "calibration.json"
  path.write_text(json.dumps({"source": "robot", "joints": {"RSP": joint}}))
  with pytest.raises(ValueError):
    load_calibration(path, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("dt", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_model_timestep_rejected(dt: float) -> None:
  with pytest.raises(ValueError, match="timestep"):
    LagCompliance(None, ("RSP",), 1, dt)


def test_unverified_artifact_cannot_enable_model(tmp_path: Path) -> None:
  artifact = CalibrationArtifact(
    str(tmp_path / "missing.json"),
    "0" * 64,
    "robot",
    {"RSP": JointCalibration(0.0, 0.01)},
  )
  with pytest.raises(FileNotFoundError):
    LagCompliance(artifact, ("RSP",), 1, DT)


def test_delay_and_lag_recovered_and_predict_a_held_out_segment() -> None:
  rng = np.random.default_rng(3)
  command = step_probe(0.02, 0.1, 1.0, DT) + sine_probe(0.005, 3.0, 1.0, DT)
  response = first_order_response(command, DT, 0.008, 0.03, 0.0)
  noisy = response + rng.normal(0.0, 1e-4, response.shape)
  fit = fit_delay_lag(command[:600], noisy[:600], DT)
  assert fit.delay_s == pytest.approx(0.008, abs=DT)
  assert fit.tau_s == pytest.approx(0.03, rel=0.1)
  assert held_out_error(fit, command[600:], response[600:], DT) < 5e-4


def test_free_decay_frequency_and_damping() -> None:
  t = np.arange(0, 3, DT)
  zeta, natural = 0.05, 4.0
  damped = natural * math.sqrt(1 - zeta**2)
  signal = np.exp(-zeta * 2 * np.pi * natural * t) * np.cos(2 * np.pi * damped * t)
  fit = fit_free_decay(signal + 0.3, DT)
  assert fit.frequency_hz == pytest.approx(natural, rel=0.02)
  assert fit.damping_ratio == pytest.approx(zeta, rel=0.1)


def test_static_push_and_frequency_response() -> None:
  displacement = np.linspace(0.0, 0.01, 50)
  fit = fit_static_push(2000.0 * displacement + 1.5, displacement)
  assert fit.stiffness == pytest.approx(2000.0) and fit.r2 == pytest.approx(1.0)
  command = sine_probe(0.01, 5.0, 2.0, DT)
  response = first_order_response(command, DT, 0.0, 1 / (2 * np.pi * 5.0), 0.0)
  gain, phase = frequency_response(command, response, DT, 5.0)
  assert gain == pytest.approx(1 / math.sqrt(2), rel=0.03)
  assert phase == pytest.approx(-math.pi / 4, abs=0.05)


def test_log_round_trip_and_rejections(tmp_path: Path) -> None:
  time = np.arange(5) * DT
  log = IdentificationLog(
    joint_names=("RSP", "REP"),
    source="simulator",
    frames={"joint": "motor side"},
    channels={
      "time_s": time,
      "target_position_rad": np.zeros((5, 2)),
      "encoder_position_rad": np.ones((5, 2)),
    },
  )
  log.save(tmp_path / "log.npz")
  loaded = IdentificationLog.load(tmp_path / "log.npz")
  assert loaded.joint_names == log.joint_names
  assert np.array_equal(loaded.channels["encoder_position_rad"], np.ones((5, 2)))
  log.channels["time_s"] = time[::-1].copy()
  with pytest.raises(ValueError, match="increasing"):
    log.validate()


def test_models_stay_disabled_without_a_robot_calibration(tmp_path: Path) -> None:
  disabled = LagCompliance(None, ("RSP",), num_envs=2, dt=DT)
  command = torch.ones(2, 1)
  assert disabled.lag(command) is command

  record = {"source": "simulator", "joints": {"RSP": {"delay_s": 0.0, "tau_s": 0.01}}}
  path = tmp_path / "calibration.json"
  path.write_text(json.dumps(record))
  digest = hashlib.sha256(path.read_bytes()).hexdigest()
  with pytest.raises(ValueError, match="robot"):
    load_calibration(path, digest)
  with pytest.raises(ValueError, match="hash"):
    load_calibration(path, "0" * 64)


def test_calibrated_lag_is_stable_and_resets_only_selected_rows(tmp_path: Path) -> None:
  record = {
    "source": "robot",
    "joints": {
      "RSP": {"delay_s": 0.003, "tau_s": 0.0002, "stiffness_nm_per_rad": 1000.0},
      "REP": {"delay_s": 0.0, "tau_s": 0.05},
    },
  }
  path = tmp_path / "calibration.json"
  path.write_text(json.dumps(record))
  artifact = load_calibration(path, hashlib.sha256(path.read_bytes()).hexdigest())
  model = LagCompliance(artifact, ("RSP", "REP"), num_envs=2, dt=DT)
  model.lag(torch.zeros(2, 2))
  outputs = torch.stack([model.lag(torch.ones(2, 2)) for _ in range(200)])
  assert torch.all(outputs <= 1.0) and torch.all(outputs.diff(dim=0) >= 0)
  assert outputs[2, 0, 0] == 0 and outputs[3, 0, 0] > 0.99
  model.reset(torch.tensor([0]))
  after = model.lag(torch.full((2, 2), 5.0))
  assert torch.all(after[0] == 5.0) and torch.all(after[1] < 5.0)
  deflected = model.link_position(torch.zeros(1, 2), torch.tensor([[10.0, 10.0]]))
  assert torch.allclose(deflected, torch.tensor([[-0.01, 0.0]]))
