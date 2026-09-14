"""Capture live reward values without evaluating stateful terms twice."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

SCHEMA_VERSION = 1
_ACTIVE_EPSILON = 1.0e-12
_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
#: A kernel target sits here relative to its own gated median; a guard sits far
#: above its p99 instead. docs/reward-shaping.md#target-placement
KERNEL_BAND = (1.3, 1.5)
GUARD_MINIMUM_RATIO = 2.0


def kernel_error(score: float, std: float) -> float:
  """Invert ``exp(-(error/std)**2)`` back to the error that produced it."""
  if not 0.0 < score <= 1.0 or not std > 0.0:
    return float("nan")
  return std * math.sqrt(-math.log(score))


def placement_verdict(
  role: str,
  target: float,
  raw: Mapping[str, Any],
) -> dict[str, Any]:
  """Judge one reward target against the distribution its own gate admits."""
  nonzero = float(raw.get("nonzero_fraction", float("nan")))
  if role == "kernel":
    measured = kernel_error(float(raw.get("gated_q50", float("nan"))), target)
    ratio = (
      target / measured if math.isfinite(measured) and measured > 0.0 else math.nan
    )
    low, high = KERNEL_BAND
    verdict = (
      "not_measured"
      if not math.isfinite(ratio)
      else "pass"
      if low <= ratio <= high
      else "saturating"
      if ratio < low
      else "floored"
    )
  elif role == "guard":
    measured = float(raw.get("gated_q99", raw.get("q99", float("nan"))))
    ratio = (
      target / measured if math.isfinite(measured) and measured > 0.0 else math.nan
    )
    verdict = (
      "not_measured"
      if not math.isfinite(nonzero)
      else "pass"
      if nonzero <= _ACTIVE_EPSILON
      else "armed"
    )
  else:
    raise ValueError(f"unknown placement role: {role}")
  return {
    "role": role,
    "target": target,
    "measured": measured,
    "ratio": ratio,
    "nonzero_fraction": nonzero,
    "verdict": verdict,
  }


class RewardAuditShapeError(ValueError):
  """Report a reward that violates the manager's per-environment shape contract."""


class _RewardProbe:
  """Capture one manager-resolved reward callable's latest output."""

  def __init__(self, name: str, func: Any, num_envs: int) -> None:
    self.name = name
    self.func = func
    self.num_envs = num_envs
    self.latest: torch.Tensor | None = None

  def __call__(self, env: Any, **params: Any) -> torch.Tensor:
    value = self.func(env, **params)
    if not isinstance(value, torch.Tensor):
      raise RewardAuditShapeError(
        f"reward {self.name!r} returned {type(value).__name__}, expected a tensor"
      )
    if value.shape != (self.num_envs,):
      raise RewardAuditShapeError(
        f"reward {self.name!r} returned shape {tuple(value.shape)}, "
        f"expected ({self.num_envs},)"
      )
    self.latest = value.detach()
    return value

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> Any:
    """Forward manager resets to a stateful wrapped term."""
    reset = getattr(self.func, "reset", None)
    return reset(env_ids=env_ids) if callable(reset) else None

  def __getattr__(self, name: str) -> Any:
    return getattr(self.func, name)


@dataclass
class _SampleSeries:
  """Accumulate finite samples and audit counts for one arm and quantity."""

  parts: list[torch.Tensor] = field(default_factory=list)
  sample_count: int = 0
  nonfinite_count: int = 0
  active_count: int = 0
  total: float = 0.0

  def append(self, values: torch.Tensor) -> None:
    """Append one device tensor while retaining only compact CPU samples."""
    values = values.detach().float().cpu().flatten()
    finite = torch.isfinite(values)
    selected = values[finite]
    self.sample_count += values.numel()
    self.nonfinite_count += int((~finite).sum().item())
    self.active_count += int((selected.abs() > _ACTIVE_EPSILON).sum().item())
    self.total += float(selected.double().sum().item())
    if selected.numel():
      self.parts.append(selected)

  def summary(self) -> dict[str, Any]:
    """Describe the finite distribution and retain invalid-sample counts."""
    values = torch.cat(self.parts) if self.parts else torch.empty(0)
    finite_count = int(values.numel())
    quantiles = (
      torch.quantile(values, torch.tensor(_QUANTILES)).tolist()
      if finite_count
      else [float("nan")] * len(_QUANTILES)
    )
    # A gated term's zeros are its off-gate steps, and a target is placed
    # against what the gate admits. docs/reward-shaping.md#target-placement
    gated = values[values.abs() > _ACTIVE_EPSILON]
    gated_quantiles = (
      torch.quantile(gated, torch.tensor(_QUANTILES)).tolist()
      if gated.numel()
      else [float("nan")] * len(_QUANTILES)
    )
    return {
      "gated_count": int(gated.numel()),
      "gated_q50": gated_quantiles[2],
      "gated_q90": gated_quantiles[3],
      "gated_q99": gated_quantiles[4],
      "sample_count": self.sample_count,
      "finite_count": finite_count,
      "nonfinite_count": self.nonfinite_count,
      "nonzero_fraction": (
        self.active_count / self.sample_count if self.sample_count else float("nan")
      ),
      "mean": self.total / finite_count if finite_count else float("nan"),
      "min": float(values.min().item()) if finite_count else float("nan"),
      "q01": quantiles[0],
      "q10": quantiles[1],
      "q50": quantiles[2],
      "q90": quantiles[3],
      "q99": quantiles[4],
      "max": float(values.max().item()) if finite_count else float("nan"),
    }


@dataclass
class _TermSamples:
  """Hold raw and live-weighted samples for one reward and rollout arm."""

  raw: _SampleSeries = field(default_factory=_SampleSeries)
  weighted_rate: _SampleSeries = field(default_factory=_SampleSeries)


def _weight_summary(values: list[float]) -> dict[str, Any]:
  """Summarize a reward's effective live weight over sampled manager calls."""
  if not values:
    return {
      "first": float("nan"),
      "last": float("nan"),
      "min": float("nan"),
      "max": float("nan"),
      "distinct": [],
    }
  return {
    "first": values[0],
    "last": values[-1],
    "min": min(values),
    "max": max(values),
    "distinct": sorted(set(values)),
  }


def _term_sign(weights: list[float]) -> str:
  """Classify whether a live reward term pays, penalizes, or changes sign."""
  if not weights or all(weight == 0.0 for weight in weights):
    return "inactive"
  if all(weight >= 0.0 for weight in weights):
    return "reward"
  if all(weight <= 0.0 for weight in weights):
    return "penalty"
  return "mixed"


class RewardAuditRecorder:
  """Instrument a reward manager and aggregate per-arm live reward distributions."""

  def __init__(
    self,
    manager: Any,
    arms: Mapping[str, torch.Tensor | list[int] | tuple[int, ...]],
    step_dt: float,
  ) -> None:
    self.manager = manager
    self.step_dt = step_dt
    self.num_envs = int(manager.num_envs)
    self.arms = {
      name: torch.as_tensor(ids, dtype=torch.long, device=manager.device)
      for name, ids in arms.items()
    }
    self._validate_arms()
    self._original_compute = manager.compute
    self._original_funcs: dict[str, Any] = {}
    self._probes: dict[str, _RewardProbe] = {}
    self._weights: dict[str, list[float]] = {name: [] for name in manager.active_terms}
    self._samples = {
      arm: {name: _TermSamples() for name in manager.active_terms} for arm in self.arms
    }
    self._denominators: dict[str, dict[str, _SampleSeries]] = {
      arm: {} for arm in self.arms
    }
    self.steps = 0
    self._installed = False

  def install(self) -> RewardAuditRecorder:
    """Wrap the manager's already-resolved term callables and compute method."""
    if self._installed:
      return self
    if hasattr(self.manager, "_mc_mjlab_reward_audit"):
      raise RuntimeError("reward manager already has an audit recorder installed")
    for name, cfg in zip(
      self.manager.active_terms, self.manager._term_cfgs, strict=True
    ):
      self._original_funcs[name] = cfg.func
      probe = _RewardProbe(name, cfg.func, self.num_envs)
      self._probes[name] = probe
      cfg.func = probe
    self.manager.compute = self._compute
    self.manager._mc_mjlab_reward_audit = self
    self._installed = True
    return self

  def restore(self) -> None:
    """Restore the reward manager's original callables and compute method."""
    if not self._installed:
      return
    for name, cfg in zip(
      self.manager.active_terms, self.manager._term_cfgs, strict=True
    ):
      cfg.func = self._original_funcs[name]
    self.manager.compute = self._original_compute
    del self.manager._mc_mjlab_reward_audit
    self._installed = False

  def __enter__(self) -> RewardAuditRecorder:
    return self.install()

  def __exit__(self, *_args: Any) -> None:
    self.restore()

  def capture_denominator(self, name: str, values: torch.Tensor) -> None:
    """Capture a per-environment conditional indicator from the same step."""
    if values.shape != (self.num_envs,):
      raise RewardAuditShapeError(
        f"conditional denominator {name!r} returned shape {tuple(values.shape)}, "
        f"expected ({self.num_envs},)"
      )
    for arm, ids in self.arms.items():
      series = self._denominators[arm].setdefault(name, _SampleSeries())
      series.append(values[ids])

  def report(self) -> dict[str, Any]:
    """Build a JSON-compatible audit report from all captured samples."""
    weights = {name: _weight_summary(values) for name, values in self._weights.items()}
    arms = {}
    for arm, terms in self._samples.items():
      arm_terms = {}
      for name, samples in terms.items():
        raw = samples.raw.summary()
        weighted = samples.weighted_rate.summary()
        arm_terms[name] = {
          "manager_output_shape": [self.num_envs],
          "sign": _term_sign(self._weights[name]),
          "effective_weight": weights[name],
          "raw": raw,
          "weighted_rate_per_second": weighted,
          "integrated_contribution_per_env": (
            samples.weighted_rate.total * self.step_dt / int(self.arms[arm].numel())
          ),
        }
      arms[arm] = {
        "env_ids": self.arms[arm].cpu().tolist(),
        "terms": arm_terms,
        "conditional_denominators": {
          name: series.summary() for name, series in self._denominators[arm].items()
        },
      }
    return {
      "schema_version": SCHEMA_VERSION,
      "steps": self.steps,
      "step_dt_s": self.step_dt,
      "manager_scale_by_dt": bool(self.manager._scale_by_dt),
      "expected_reward_shape": [self.num_envs],
      "arms": arms,
    }

  def issues(self) -> list[str]:
    """Return audit failures that the manager would otherwise sanitize."""
    problems = []
    for arm, terms in self._samples.items():
      for name, samples in terms.items():
        if samples.raw.nonfinite_count:
          problems.append(
            f"{arm}/{name}: {samples.raw.nonfinite_count} non-finite raw values"
          )
    return problems

  def _validate_arms(self) -> None:
    """Require non-empty, disjoint, in-range environment assignments."""
    seen: set[int] = set()
    for name, ids in self.arms.items():
      values = ids.cpu().tolist()
      if not values:
        raise ValueError(f"reward-audit arm {name!r} has no environments")
      if len(set(values)) != len(values):
        raise ValueError(f"reward-audit arm {name!r} repeats an environment")
      if any(value < 0 or value >= self.num_envs for value in values):
        raise ValueError(f"reward-audit arm {name!r} has an out-of-range environment")
      overlap = seen.intersection(values)
      if overlap:
        raise ValueError(f"reward-audit arms overlap at environments {sorted(overlap)}")
      seen.update(values)

  def _compute(self, dt: float) -> torch.Tensor:
    """Run every term once, then remove synthetic zero-weight contributions."""
    weights = {
      name: float(cfg.weight)
      for name, cfg in zip(
        self.manager.active_terms, self.manager._term_cfgs, strict=True
      )
    }
    zero_terms: list[tuple[int, str, Any, torch.Tensor]] = []
    for index, (name, cfg) in enumerate(
      zip(self.manager.active_terms, self.manager._term_cfgs, strict=True)
    ):
      self._probes[name].latest = None
      if weights[name] == 0.0:
        zero_terms.append((index, name, cfg, self.manager._episode_sums[name].clone()))
        cfg.weight = 1.0
    try:
      self._original_compute(dt)
    finally:
      for _, name, cfg, _ in zero_terms:
        cfg.weight = weights[name]
    for index, name, _, episode_before in zero_terms:
      self.manager._episode_sums[name].copy_(episode_before)
      self.manager._step_reward[:, index].zero_()
    scale = dt if self.manager._scale_by_dt else 1.0
    torch.sum(self.manager._step_reward, dim=1, out=self.manager._reward_buf)
    self.manager._reward_buf.mul_(scale)
    self._capture(weights)
    return self.manager._reward_buf

  def _capture(self, weights: Mapping[str, float]) -> None:
    """Copy one manager call into arm-specific raw and weighted distributions."""
    for name in self.manager.active_terms:
      value = self._probes[name].latest
      if value is None:
        raise RuntimeError(f"reward manager did not evaluate term {name!r}")
      weight = weights[name]
      self._weights[name].append(weight)
      for arm, ids in self.arms.items():
        selected = value[ids]
        self._samples[arm][name].raw.append(selected)
        self._samples[arm][name].weighted_rate.append(selected * weight)
    self.steps += 1
