"""Transparent recovery features and calibrated residual-authority gate."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


FEATURE_NAMES = ("dcm_error", "base_ang_speed", "tilt", "load_deviation")
mjtSensor = vars(mujoco)["mjtSensor"]


def _wrench_sensor(mj_model, suffix: str, sensor_type: int) -> tuple[int, int]:
  """Resolve the data offset and site id of one suffix-matched wrench sensor."""
  for index in range(mj_model.nsensor):
    sensor = mj_model.sensor(index)
    if sensor.name.endswith(suffix) and int(sensor.type[0]) == sensor_type:
      return int(sensor.adr[0]), int(sensor.objid[0])
  raise ValueError(f"missing detector sensor '*{suffix}'")


class RecoveryFeatureExtractor:
  """Compute deployable recovery features from controller and robot sensors."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    controller_term,
    sensor_names: tuple[str, ...] = ("LeftFootForceSensor", "RightFootForceSensor"),
  ) -> None:
    self.env = env
    self.term = controller_term
    self.asset = env.scene[controller_term.cfg.entity_name]
    force_cols: list[int] = []
    torque_cols: list[int] = []
    site_ids: list[int] = []
    for name in sensor_names:
      address, site_id = _wrench_sensor(
        env.sim.mj_model, f"{name}_fsensor", mjtSensor.mjSENS_FORCE
      )
      torque_address, _ = _wrench_sensor(
        env.sim.mj_model, f"{name}_tsensor", mjtSensor.mjSENS_TORQUE
      )
      force_cols.extend((address, address + 1, address + 2))
      torque_cols.extend((torque_address, torque_address + 1, torque_address + 2))
      site_ids.append(site_id)
    self.force_cols = torch.tensor(force_cols, device=env.device, dtype=torch.long)
    self.torque_cols = torch.tensor(torque_cols, device=env.device, dtype=torch.long)
    self.site_ids = torch.tensor(site_ids, device=env.device, dtype=torch.long)
    self.root_body_id = self.asset.indexing.root_body_id

  def measure(self) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detector features and the signed horizontal DCM error."""
    data = self.env.sim.data
    count = len(self.site_ids)
    rotation = data.site_xmat[:, self.site_ids].reshape(-1, count, 3, 3)
    force_s = data.sensordata[:, self.force_cols].reshape(-1, count, 3, 1)
    torque_s = data.sensordata[:, self.torque_cols].reshape(-1, count, 3, 1)
    force_w = -(rotation @ force_s).squeeze(-1)
    torque_w = -(rotation @ torque_s).squeeze(-1)
    total_force = force_w.sum(dim=1)
    com = data.subtree_com[:, self.root_body_id]
    com_vel = data.subtree_linvel[:, self.root_body_id]
    commanded = self.term.controller_vector("control_com_vel")
    normal_force = total_force[:, 2].clamp(min=20.0)
    site_pos = data.site_xpos[:, self.site_ids]
    lever = site_pos - com.unsqueeze(1)
    moment = (torque_w + torch.cross(lever, force_w, dim=-1)).sum(dim=1)
    height = -com[:, 2]
    measured = torch.stack(
      (
        (height * total_force[:, 0] - moment[:, 1]) / normal_force,
        (moment[:, 0] + height * total_force[:, 1]) / normal_force,
      ),
      dim=-1,
    )
    omega = torch.sqrt(9.81 / com[:, 2].clamp(min=0.1)).unsqueeze(-1)
    dcm_error = (com_vel[:, :2] - commanded[:, :2]) / omega - measured
    dcm = torch.linalg.vector_norm(dcm_error, dim=1)
    angular_speed = torch.linalg.vector_norm(self.asset.data.root_link_ang_vel_b, dim=1)
    gravity = self.asset.data.projected_gravity_b
    tilt = torch.acos((-gravity[:, 2]).clamp(-1.0, 1.0))
    body_ids = self.asset.indexing.body_ids
    mass = self.env.sim.model.body_mass[:, body_ids].sum(dim=1)
    load_deviation = (total_force[:, 2] / (mass * 9.81) - 1.0).abs()
    features = torch.stack((dcm, angular_speed, tilt, load_deviation), dim=1)
    return features, dcm_error

  def __call__(self) -> torch.Tensor:
    """Return the four nonnegative detector features in ``FEATURE_NAMES`` order."""
    return self.measure()[0]


@dataclass(frozen=True)
class RecoveryCalibration:
  """Calibrated monotonic score and temporal-filter parameters."""

  centers: tuple[float, float, float, float]
  scales: tuple[float, float, float, float]
  threshold: float
  activation_span: float
  attack_s: float = 0.10
  decay_tau_s: float = 0.50
  cutoff: float = 1.0e-3
  max_active_s: float = 1.75
  rearm_s: float = 0.25
  rearm_score: float = 0.50
  onset_feature: str = "base_ang_speed"
  onset_delta: float = 0.035

  @classmethod
  def from_json(cls, path: str | Path) -> RecoveryCalibration:
    """Read calibration values while rejecting a mismatched feature order."""
    payload = json.loads(Path(path).read_text())
    if tuple(payload["feature_names"]) != FEATURE_NAMES:
      raise ValueError(
        f"detector feature mismatch: {payload['feature_names']} != {FEATURE_NAMES}"
      )
    return cls(
      centers=tuple(payload["centers"]),
      scales=tuple(payload["scales"]),
      threshold=float(payload["threshold"]),
      activation_span=float(payload["activation_span"]),
      attack_s=float(payload.get("attack_s", 0.10)),
      decay_tau_s=float(payload.get("decay_tau_s", 0.50)),
      cutoff=float(payload.get("cutoff", 1.0e-3)),
      max_active_s=float(payload.get("max_active_s", 1.75)),
      rearm_s=float(payload.get("rearm_s", 0.25)),
      rearm_score=float(payload.get("rearm_score", 0.50)),
      onset_feature=str(payload.get("onset_feature", "base_ang_speed")),
      onset_delta=float(payload.get("onset_delta", 0.035)),
    )


def detector_target(
  features: torch.Tensor, calibration: RecoveryCalibration
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return monotonic max-normalized score and smooth activation target."""
  centers = features.new_tensor(calibration.centers)
  scales = features.new_tensor(calibration.scales)
  score = ((features - centers) / scales).amax(dim=1)
  level = ((score - calibration.threshold) / calibration.activation_span).clamp(
    0.0, 1.0
  )
  return score, level.square() * (3.0 - 2.0 * level)


def filter_authority(
  previous: torch.Tensor,
  target: torch.Tensor,
  calibration: RecoveryCalibration,
  dt: float,
) -> torch.Tensor:
  """Apply smooth attack, exponential decay, and an exact-zero cutoff."""
  attack = 1.0 - math.exp(-dt / calibration.attack_s)
  decay = math.exp(-dt / calibration.decay_tau_s)
  rising = previous + attack * (target - previous)
  falling = torch.maximum(target, previous * decay)
  authority = torch.where(target > previous, rising, falling)
  return torch.where(
    authority >= calibration.cutoff, authority, torch.zeros_like(authority)
  )


class RecoveryFilter:
  """Bound sensor-triggered recovery bursts with a refractory interval."""

  def __init__(
    self, num_envs: int, device: torch.device | str, calibration: RecoveryCalibration
  ) -> None:
    self.calibration = calibration
    self.authority = torch.zeros(num_envs, device=device)
    self.active_age = torch.zeros_like(self.authority)
    self.quiet_age = torch.zeros_like(self.authority)
    self.active = torch.zeros(num_envs, device=device, dtype=torch.bool)
    self.armed = torch.zeros(num_envs, device=device, dtype=torch.bool)
    self.previous_onset = torch.zeros_like(self.authority)
    self.primed = torch.zeros(num_envs, device=device, dtype=torch.bool)

  def update(
    self,
    score: torch.Tensor,
    onset_signal: torch.Tensor,
    target: torch.Tensor,
    dt: float,
  ) -> torch.Tensor:
    """Advance one deployable sensor-triggered authority burst."""
    onset = onset_signal - self.previous_onset >= self.calibration.onset_delta
    starts = self.armed & ~self.active & self.primed & onset & (target > 0.0)
    self.active |= starts
    self.armed &= ~starts
    self.active_age[starts] = 0.0
    self.quiet_age[starts] = 0.0

    filtered = filter_authority(self.authority, target, self.calibration, dt)
    self.authority = torch.where(self.active, filtered, torch.zeros_like(filtered))
    self.active_age[self.active] += dt
    expired = self.active & (self.active_age >= self.calibration.max_active_s)
    ended = self.active & (target == 0.0) & (self.authority == 0.0)
    stopped = expired | ended
    self.active[stopped] = False
    self.authority[stopped] = 0.0

    nominal = score <= self.calibration.rearm_score
    quiet = ~self.active & nominal
    self.quiet_age[quiet] += dt
    self.quiet_age[~quiet] = 0.0
    self.armed |= quiet & (self.quiet_age >= self.calibration.rearm_s)
    self.previous_onset.copy_(onset_signal)
    self.primed.fill_(True)
    return self.authority

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Restore the inactive, armed state across episode boundaries."""
    ids = slice(None) if env_ids is None else env_ids
    self.authority[ids] = 0.0
    self.active_age[ids] = 0.0
    self.quiet_age[ids] = 0.0
    self.active[ids] = False
    self.armed[ids] = False
    self.previous_onset[ids] = 0.0
    self.primed[ids] = False


class RecoveryAuthority:
  """Stateful calibrated detector producing one residual-authority value per env."""

  def __init__(self, env, controller_term, calibration_path: str | Path) -> None:
    self.calibration = RecoveryCalibration.from_json(calibration_path)
    self.extractor = RecoveryFeatureExtractor(env, controller_term)
    self.dt = env.step_dt
    self.filter = RecoveryFilter(env.num_envs, env.device, self.calibration)
    self.onset_index = FEATURE_NAMES.index(self.calibration.onset_feature)
    self.score = torch.zeros(env.num_envs, device=env.device)
    self.dcm_error = torch.zeros(env.num_envs, 2, device=env.device)

  @property
  def authority(self) -> torch.Tensor:
    """Current residual authority in ``[0, 1]``."""
    return self.filter.authority

  def update(self) -> torch.Tensor:
    """Measure the current state and advance the temporal authority filter."""
    features, self.dcm_error = self.extractor.measure()
    self.score, target = detector_target(features, self.calibration)
    return self.filter.update(
      self.score, features[:, self.onset_index], target, self.dt
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Clear filter state across episode boundaries."""
    ids = slice(None) if env_ids is None else env_ids
    self.filter.reset(ids)
    self.score[ids] = 0.0
    self.dcm_error[ids] = 0.0
