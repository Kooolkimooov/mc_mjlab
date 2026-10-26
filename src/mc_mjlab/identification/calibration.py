"""Opt-in command lag and joint compliance, enabled only by a measured robot calibration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class JointCalibration:
  """One joint's fitted delay, lag and optional link stiffness."""

  delay_s: float
  tau_s: float
  stiffness_nm_per_rad: float | None = None

  def __post_init__(self) -> None:
    if self.delay_s < 0 or self.tau_s <= 0:
      raise ValueError("calibration needs delay >= 0 and tau > 0")
    if self.stiffness_nm_per_rad is not None and self.stiffness_nm_per_rad <= 0:
      raise ValueError("calibrated stiffness must be positive")


@dataclass(frozen=True)
class CalibrationArtifact:
  """A hashed calibration file; only robot-measured artifacts may enable models."""

  path: str
  sha256: str
  source: str
  joints: dict[str, JointCalibration]


def load_calibration(path: Path, expected_sha256: str) -> CalibrationArtifact:
  """Verify the file hash and that it records robot measurements."""
  data = path.read_bytes()
  digest = hashlib.sha256(data).hexdigest()
  if digest != expected_sha256:
    raise ValueError(f"calibration {path} hash {digest} != {expected_sha256}")
  record = json.loads(data)
  if record.get("source") != "robot":
    raise ValueError("only robot-measured calibrations enable lag/compliance models")
  joints = {
    name: JointCalibration(**values) for name, values in record["joints"].items()
  }
  return CalibrationArtifact(str(path), digest, record["source"], joints)


class LagCompliance:
  """Delay plus first-order lag on targets, and link deflection under effort."""

  def __init__(
    self,
    artifact: CalibrationArtifact | None,
    joint_names: tuple[str, ...],
    num_envs: int,
    dt: float,
    device: str = "cpu",
  ) -> None:
    self.enabled = artifact is not None
    self.artifact = artifact
    if artifact is None:
      return

    missing = [name for name in joint_names if name not in artifact.joints]
    if missing:
      raise ValueError(f"calibration lacks joints {missing}")
    calibrations = [artifact.joints[name] for name in joint_names]
    self._lags = [round(c.delay_s / dt) for c in calibrations]
    # Exact discretization: 0 < alpha < 1 for every tau > 0, so never unstable.
    self._alpha = torch.tensor(
      [1.0 - math.exp(-dt / c.tau_s) for c in calibrations], device=device
    )
    self._compliance = torch.tensor(
      [
        0.0 if c.stiffness_nm_per_rad is None else 1.0 / c.stiffness_nm_per_rad
        for c in calibrations
      ],
      device=device,
    )
    depth = max(self._lags) + 1
    self._history = torch.zeros(depth, num_envs, len(joint_names), device=device)
    self._state = torch.zeros(num_envs, len(joint_names), device=device)
    self._columns = torch.arange(len(joint_names), device=device)
    self._lag_index = torch.tensor(self._lags, device=device)
    self._primed = torch.zeros(num_envs, dtype=torch.bool, device=device)

  def lag(self, command: torch.Tensor) -> torch.Tensor:
    """Delayed, low-passed command; the identity while disabled."""
    if not self.enabled:
      return command
    fresh = ~self._primed
    self._history[:, fresh] = command[fresh]
    self._state[fresh] = command[fresh]
    self._primed[:] = True
    self._history = torch.roll(self._history, 1, dims=0)
    self._history[0] = command
    delayed = self._history[self._lag_index, :, self._columns].T
    self._state = self._state + self._alpha * (delayed - self._state)
    return self._state

  def link_position(
    self, motor_position: torch.Tensor, effort: torch.Tensor
  ) -> torch.Tensor:
    """Link angle behind a compliant transmission; the motor angle while disabled."""
    if not self.enabled:
      return motor_position
    return motor_position - effort * self._compliance

  def reset(self, env_ids: torch.Tensor | slice) -> None:
    """Re-prime only the selected rows from their next command."""
    if self.enabled:
      self._primed[env_ids] = False
