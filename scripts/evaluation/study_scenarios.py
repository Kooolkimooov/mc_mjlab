"""Typed robustness scenarios: sensitivity settings, never hardware measurements."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import mujoco
import torch
import warp as wp
from mjlab.utils.noise.noise_cfg import ConstantNoiseCfg, NoiseCfg, NoiseModelCfg
from mjlab.utils.noise.noise_model import NoiseModel

from mc_mjlab.tasks.locomanip.cart import CART_FLOOR_PAIR

IMU_TERMS = ("base_lin_vel", "base_ang_vel", "projected_gravity")
FOOT_GEOMS = ("robot/left_foot_collision", "robot/right_foot_collision")


@dataclass(frozen=True)
class Scenario:
  """One isolated mechanism; observation corruption leaves the policy interface intact."""

  name: str
  encoder_bias: bool = False
  white_noise: bool = False
  velocity_bias: tuple[float, float, float] | None = None
  imu_rotation_deg: tuple[float, float] | None = None
  velocity_error: tuple[float, float] | None = None
  pd_gain_scale: float = 1.0
  actuator_delay_ms: int = 0
  foot_friction_scale: float = 1.0
  cart_friction_scale: float = 1.0

  def configure(self, cfg: Any) -> None:
    """Edit the environment config before it is built."""
    actor = cfg.observations["actor"]
    corrupted = self._observation_noise(cfg.decimation * cfg.sim.mujoco.timestep)
    if self.white_noise or corrupted:
      actor.enable_corruption = True
    if corrupted and not self.white_noise:
      for name, term in actor.terms.items():
        actor.terms[name] = replace(term, noise=corrupted.get(name))
    elif corrupted:
      for name, noise in corrupted.items():
        actor.terms[name] = replace(actor.terms[name], noise=noise)

    if self.actuator_delay_ms:
      lag = round(self.actuator_delay_ms * 1e-3 / cfg.sim.mujoco.timestep)
      articulation = cfg.scene.entities["robot"].articulation
      articulation.actuators = tuple(
        replace(a, delay_min_lag=lag, delay_max_lag=lag) for a in articulation.actuators
      )

  def install(self, env: Any) -> dict[str, Any]:
    """Apply post-build model changes once; returns what was changed."""
    applied: dict[str, Any] = {}
    if self.pd_gain_scale != 1.0:
      applied["pd_gains"] = _scale_installed_gains(env, self.pd_gain_scale)
    if self.foot_friction_scale != 1.0:
      floor = _plane_geom(env.sim.mj_model)
      applied["foot_friction"] = _scale_geom_friction(
        env, (*FOOT_GEOMS, floor), self.foot_friction_scale
      )
    if self.cart_friction_scale != 1.0:
      applied["cart_friction"] = _scale_pair_friction(
        env, CART_FLOOR_PAIR, self.cart_friction_scale
      )
    return applied

  def _observation_noise(self, step_dt: float) -> dict[str, Any]:
    noise: dict[str, Any] = {}
    if self.velocity_bias is not None:
      noise["base_lin_vel"] = ConstantNoiseCfg(bias=self.velocity_bias)
    if self.imu_rotation_deg is not None:
      rotation = RotationNoiseCfg(roll_pitch_deg=self.imu_rotation_deg)
      noise |= dict.fromkeys(IMU_TERMS, rotation)
    if self.velocity_error is not None:
      rms, correlation = self.velocity_error
      noise["base_lin_vel"] = CorrelatedNoiseModelCfg(
        rms=rms, correlation_s=correlation, step_dt=step_dt
      )
    return noise


@dataclass
class RotationNoiseCfg(NoiseCfg):
  """A fixed sensor-mounting rotation of a 3-vector channel."""

  roll_pitch_deg: tuple[float, float] = (0.0, 0.0)

  def apply(self, data: torch.Tensor) -> torch.Tensor:
    roll, pitch = (math.radians(a) for a in self.roll_pitch_deg)
    rx = torch.tensor(
      [
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)],
      ]
    )
    ry = torch.tensor(
      [
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)],
      ]
    )
    return data @ (ry @ rx).T.to(data.device, data.dtype)


class CorrelatedNoise(NoiseModel):
  """Additive Ornstein-Uhlenbeck error, stationary from each row's reset."""

  def __init__(
    self, noise_model_cfg: CorrelatedNoiseModelCfg, num_envs: int, device: str
  ) -> None:
    super().__init__(noise_model_cfg, num_envs, device)
    self._cfg = noise_model_cfg
    self._decay = math.exp(-noise_model_cfg.step_dt / noise_model_cfg.correlation_s)
    self._state: torch.Tensor | None = None

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._state is None:
      return
    rows = slice(None) if env_ids is None else env_ids
    self._state[rows] = torch.randn_like(self._state[rows]) * self._cfg.rms

  def __call__(self, data: torch.Tensor) -> torch.Tensor:
    if self._state is None:
      self._state = torch.randn_like(data) * self._cfg.rms
    innovation = torch.randn_like(data) * self._cfg.rms
    self._state = self._decay * self._state + math.sqrt(1 - self._decay**2) * innovation
    return data + self._state


@dataclass(kw_only=True)
class CorrelatedNoiseModelCfg(NoiseModelCfg, class_type=CorrelatedNoise):
  """Per-axis RMS error with an exponential correlation time."""

  rms: float
  correlation_s: float
  step_dt: float
  # NoiseModel requires one; the model's own state replaces it.
  noise_cfg: NoiseCfg = field(default_factory=ConstantNoiseCfg)


def scenarios() -> dict[str, Scenario]:
  """The plan's sensitivity grid, one mechanism per entry."""
  grid = [
    Scenario("encoder-bias", encoder_bias=True),
    Scenario("white-noise", white_noise=True),
    Scenario("encoder-bias+white-noise", encoder_bias=True, white_noise=True),
  ]
  axes = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}
  for label, (ax, ay, az) in axes.items():
    for value in (-0.05, -0.02, 0.02, 0.05):
      bias = (value * ax, value * ay, value * az)
      grid.append(Scenario(f"velocity-bias/{label}/{value:+g}", velocity_bias=bias))
  for axis, label in enumerate(("roll", "pitch")):
    for value in (-2.0, -1.0, 1.0, 2.0):
      angles = (value, 0.0) if axis == 0 else (0.0, value)
      grid.append(Scenario(f"imu-rotation/{label}/{value:+g}", imu_rotation_deg=angles))
  for rms in (0.02, 0.05):
    for correlation in (0.1, 0.5):
      grid.append(
        Scenario(
          f"velocity-error/{rms:g}/{correlation:g}", velocity_error=(rms, correlation)
        )
      )
  grid += [Scenario(f"pd-gains/{s:g}", pd_gain_scale=s) for s in (0.9, 1.1)]
  grid += [
    Scenario(f"actuator-delay/{ms}ms", actuator_delay_ms=ms) for ms in (2, 4, 10)
  ]
  grid += [Scenario(f"foot-friction/{s:g}", foot_friction_scale=s) for s in (0.9, 1.1)]
  grid += [Scenario(f"cart-friction/{s:g}", cart_friction_scale=s) for s in (0.9, 1.1)]
  return {scenario.name: scenario for scenario in grid}


def _scale_installed_gains(env: Any, scale: float) -> dict[str, float]:
  # The action term has already installed PDgains_sim; default_* are armature gains.
  for actuator in env.scene["robot"].actuators:
    actuator.stiffness.mul_(scale)
    actuator.damping.mul_(scale)
  return {"scale": scale}


def _plane_geom(model: Any) -> str:
  planes = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
    for i in range(model.ngeom)
    if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
  ]
  if len(planes) != 1:
    raise ValueError("friction scenarios need exactly one ground plane")
  return planes[0]


def _scale_geom_friction(env: Any, names: tuple[str, ...], scale: float) -> dict:
  ids = [
    mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in names
  ]
  if min(ids) < 0:
    raise ValueError(f"missing geoms among {names}")
  friction = _raw(env.sim.model.geom_friction)
  before = friction[0, ids, 0].tolist()
  friction[:, ids, 0] *= scale
  return dict(zip(names, before, strict=True)) | {"scale": scale}


def _scale_pair_friction(env: Any, name: str, scale: float) -> dict:
  pair = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_PAIR, name)
  if pair < 0:
    raise ValueError(f"missing contact pair {name}")
  friction = _raw(env.sim.model.pair_friction)
  before = friction[0, pair, :2].tolist()
  friction[:, pair, :2] *= scale
  return {"tangential_before": before, "scale": scale}


def _raw(field: Any) -> torch.Tensor:
  # The proxy expands unbatched fields with stride 0; write the warp memory itself.
  return wp.to_torch(field.wp_array)
