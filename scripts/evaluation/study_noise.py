"""Condition-keyed observation corruption indexed by episode time."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from mjlab.utils.noise.noise_cfg import ConstantNoiseCfg, NoiseModelCfg, UniformNoiseCfg
from mjlab.utils.noise.noise_model import NoiseModel


class KeyedNoise(NoiseModel):
  """Replay independent per-condition streams without advancing on observation reads."""

  def __init__(
    self, noise_model_cfg: KeyedNoiseCfg, num_envs: int, device: str
  ) -> None:
    super().__init__(noise_model_cfg, num_envs, device)
    self.cfg = noise_model_cfg
    if len(self.cfg.input_ids) != num_envs:
      raise ValueError("one noise input identity per environment required")
    self.env: Any = None
    self.table: torch.Tensor | None = None
    self.last_noise: torch.Tensor | None = None

  def bind(self, env: Any) -> None:
    """Use episode counters so a partial reset cannot advance surviving rows."""
    self.env = env

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """The environment resets the selected clocks; the immutable streams stay fixed."""
    del env_ids

  def __call__(self, data: torch.Tensor) -> torch.Tensor:
    if self.env is None:
      raise RuntimeError("condition-keyed noise must be bound before observations")
    if self.table is None:
      self.table = self._table(data)
    ticks = self.env.episode_length_buf.long()
    if bool(((ticks < 0) | (ticks >= self.cfg.samples)).any()):
      raise ValueError("episode exceeds the recorded noise timeline")
    self.last_noise = self.table[torch.arange(len(ticks), device=data.device), ticks]
    return data + self.last_noise

  def _table(self, data: torch.Tensor) -> torch.Tensor:
    if data.ndim != 2:
      raise ValueError("study noise expects flat continuous observation channels")
    samples = []
    for identifier in self.cfg.input_ids:
      digest = hashlib.sha256(f"{identifier}/{self.cfg.channel}".encode()).digest()
      rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
      shape = (self.cfg.samples, data.shape[1])
      noise = self.cfg.noise_cfg
      if isinstance(noise, UniformNoiseCfg) and noise.operation == "add":
        values = rng.uniform(noise.n_min, noise.n_max, size=shape)
      elif isinstance(noise, ConstantNoiseCfg) and self.cfg.correlation_s is not None:
        values = rng.normal(size=shape) * self.cfg.rms
      else:
        raise TypeError("study supports additive uniform or correlated Gaussian noise")
      samples.append(values)
    table = np.stack(samples)
    if self.cfg.correlation_s is not None:
      decay = math.exp(-self.cfg.step_dt / self.cfg.correlation_s)
      innovation = math.sqrt(
        -math.expm1(-2 * self.cfg.step_dt / self.cfg.correlation_s)
      )
      for tick in range(1, self.cfg.samples):
        table[:, tick] = decay * table[:, tick - 1] + innovation * table[:, tick]
    return torch.as_tensor(table, device=data.device, dtype=data.dtype)


@dataclass(kw_only=True)
class KeyedNoiseCfg(NoiseModelCfg, class_type=KeyedNoise):
  """Declared corruption with explicit input identities and a bounded time horizon."""

  input_ids: tuple[str, ...]
  channel: str
  samples: int
  step_dt: float
  rms: float = 0.0
  correlation_s: float | None = None

  def __post_init__(self) -> None:
    if self.samples < 1 or not math.isfinite(self.step_dt) or self.step_dt <= 0:
      raise ValueError("positive samples and finite positive timestep required")
    if self.correlation_s is not None and (
      not math.isfinite(self.correlation_s)
      or self.correlation_s <= 0
      or not math.isfinite(self.rms)
      or self.rms < 0
    ):
      raise ValueError("finite positive correlation and nonnegative RMS required")


def bind_noise(env: Any) -> None:
  """Bind runtime clocks without putting simulator objects in configuration records."""
  for group in env.observation_manager._group_obs_class_instances.values():
    for noise in group.values():
      if isinstance(noise, KeyedNoise):
        noise.bind(env)
