"""Joint-dynamics fits for identification logs; numpy only, no simulator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LagFit:
  """Pure delay plus first-order lag from command to response."""

  delay_s: float
  tau_s: float
  rmse: float
  r2: float


@dataclass(frozen=True)
class DecayFit:
  """Damped oscillation around a settled offset."""

  frequency_hz: float
  damping_ratio: float
  peaks: int


@dataclass(frozen=True)
class StiffnessFit:
  """Linear static compliance: displacement per unit load."""

  stiffness: float
  offset: float
  r2: float


def first_order_response(
  command: np.ndarray, dt: float, delay_s: float, tau_s: float, initial: float
) -> np.ndarray:
  """Exact zero-order-hold discretization; stable for every tau > 0."""
  if tau_s <= 0 or delay_s < 0:
    raise ValueError("positive time constant and non-negative delay required")
  lag = int(round(delay_s / dt))
  delayed = np.concatenate((np.full(lag, command[0]), command[: len(command) - lag]))
  alpha = 1.0 - np.exp(-dt / tau_s)
  out = np.empty_like(delayed, dtype=float)
  state = initial
  for i, value in enumerate(delayed):
    state += alpha * (value - state)
    out[i] = state
  return out


def fit_delay_lag(
  command: np.ndarray,
  response: np.ndarray,
  dt: float,
  max_delay_s: float = 0.05,
) -> LagFit:
  """Whole-sample delay and log-spaced time constant, then a local refinement."""
  _check_series(command, response)
  lags = np.arange(int(round(max_delay_s / dt)) + 1)
  errors = _lag_errors(command, response, dt, lags, np.geomspace(dt / 2, 0.5, 60))
  lag_index, tau_index = np.unravel_index(np.argmin(errors), errors.shape)
  coarse = np.geomspace(dt / 2, 0.5, 60)[tau_index]
  taus = np.geomspace(coarse / 1.15, coarse * 1.15, 61)
  fine = _lag_errors(command, response, dt, lags[lag_index : lag_index + 1], taus)[0]
  error = float(fine.min())
  delay = float(lags[lag_index] * dt)
  tau = float(taus[int(np.argmin(fine))])
  return LagFit(delay, tau, float(np.sqrt(error)), _r2(response, error))


def _lag_errors(
  command: np.ndarray,
  response: np.ndarray,
  dt: float,
  lags: np.ndarray,
  taus: np.ndarray,
) -> np.ndarray:
  """Mean squared error for every (lag, tau) pair in one pass over time."""
  alpha = 1.0 - np.exp(-dt / taus)
  state = np.full((len(lags), len(taus)), response[0], dtype=float)
  total = np.zeros_like(state)
  for i in range(len(command)):
    delayed = command[np.maximum(i - lags, 0)]
    state += alpha * (delayed[:, None] - state)
    total += (state - response[i]) ** 2
  return total / len(command)


def fit_free_decay(signal: np.ndarray, dt: float) -> DecayFit:
  """Frequency from peak spacing, damping from the logarithmic decrement."""
  centered = signal - signal[-max(1, len(signal) // 10) :].mean()
  inner = centered[1:-1]
  peaks = (
    np.flatnonzero((inner > centered[:-2]) & (inner >= centered[2:]) & (inner > 0)) + 1
  )
  if len(peaks) < 3:
    raise ValueError("free decay needs at least three positive peaks")
  period = float(np.mean(np.diff(peaks))) * dt
  amplitudes = centered[peaks]
  decrement = float(np.mean(np.log(amplitudes[:-1] / amplitudes[1:])))
  zeta = decrement / np.sqrt(4 * np.pi**2 + decrement**2)
  damped_frequency = 1.0 / period
  natural = damped_frequency / np.sqrt(max(1.0 - zeta**2, 1e-12))
  return DecayFit(float(natural), float(zeta), len(peaks))


def fit_static_push(load: np.ndarray, displacement: np.ndarray) -> StiffnessFit:
  """Least-squares load = stiffness * displacement + offset."""
  _check_series(load, displacement)
  design = np.stack((displacement, np.ones_like(displacement)), axis=1)
  (stiffness, offset), *_ = np.linalg.lstsq(design, load, rcond=None)
  error = float(np.mean((design @ np.array([stiffness, offset]) - load) ** 2))
  return StiffnessFit(float(stiffness), float(offset), _r2(load, error))


def frequency_response(
  command: np.ndarray, response: np.ndarray, dt: float, frequency_hz: float
) -> tuple[float, float]:
  """Gain and phase (rad) at one probe frequency by sine/cosine correlation."""
  _check_series(command, response)
  t = np.arange(len(command)) * dt
  basis = np.exp(-2j * np.pi * frequency_hz * t)
  u = np.sum((command - command.mean()) * basis)
  y = np.sum((response - response.mean()) * basis)
  ratio = y / u
  return float(abs(ratio)), float(np.angle(ratio))


def step_probe(
  amplitude: float, start_s: float, duration_s: float, dt: float
) -> np.ndarray:
  """A small residual step held from ``start_s`` to the end."""
  t = np.arange(round(duration_s / dt)) * dt
  return np.where(t >= start_s, amplitude, 0.0)


def sine_probe(
  amplitude: float, frequency_hz: float, duration_s: float, dt: float
) -> np.ndarray:
  """A single-frequency residual probe."""
  t = np.arange(round(duration_s / dt)) * dt
  return amplitude * np.sin(2 * np.pi * frequency_hz * t)


def held_out_error(
  fit: LagFit, command: np.ndarray, response: np.ndarray, dt: float
) -> float:
  """RMSE of a fitted lag predicting a segment it was not fitted on."""
  _check_series(command, response)
  model = first_order_response(command, dt, fit.delay_s, fit.tau_s, response[0])
  return float(np.sqrt(np.mean((model - response) ** 2)))


def _r2(target: np.ndarray, mse: float) -> float:
  variance = float(np.var(target))
  return 1.0 - mse / variance if variance > 0 else float("nan")


def _check_series(a: np.ndarray, b: np.ndarray) -> None:
  if a.shape != b.shape or a.ndim != 1 or len(a) < 3:
    raise ValueError("two equal-length one-dimensional series required")
  if not (np.isfinite(a).all() and np.isfinite(b).all()):
    raise ValueError("series must be finite")
