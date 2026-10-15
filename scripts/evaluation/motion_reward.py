"""Score measured rewards with counterfactual costs on applied slowdown."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

_MOTION_COST_TERMS = ("braking", "braking_start", "holding")


def score_braking(
  frames: Sequence[dict[str, Any]],
  *,
  step_dt: float,
  gamma: float,
  costs: Sequence[float],
  event_costs: Sequence[float] = (0.0,),
  holding_costs: Sequence[float] = (0.0,),
  reward_scales: Mapping[str, float] | None = None,
  decision_time_s: float | None = None,
) -> dict[str, Any]:
  """Compare first-episode returns from the shared intervention decision time."""
  if not step_dt > 0 or not 0 < gamma <= 1:
    raise ValueError("positive timestep and gamma in (0, 1] required")
  if not costs or any(not math.isfinite(c) or c < 0 for c in costs):
    raise ValueError("finite nonnegative braking costs required")
  if not event_costs or any(not math.isfinite(c) or c < 0 for c in event_costs):
    raise ValueError("finite nonnegative event costs required")
  if not holding_costs or any(not math.isfinite(c) or c < 0 for c in holding_costs):
    raise ValueError("finite nonnegative holding costs required")
  reward_scales = dict(reward_scales or {})
  if any(
    name in _MOTION_COST_TERMS or not math.isfinite(scale) or scale < 0
    for name, scale in reward_scales.items()
  ):
    raise ValueError("finite nonnegative scales for non-motion rewards required")
  groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
  for frame in frames:
    groups[frame["arm"]].append(frame)
  triggers = [f["trigger_time_s"] for f in frames if f["trigger_time_s"] is not None]
  if not triggers and decision_time_s is None:
    raise ValueError("braking was never triggered")
  decision_time = min(triggers) if decision_time_s is None else decision_time_s
  if not math.isfinite(decision_time) or decision_time < 0:
    raise ValueError("decision time must be finite and nonnegative")
  output: dict[str, Any] = {}
  for arm, samples in groups.items():
    _validate_samples(samples, step_dt)
    if any(not reward_scales.keys() <= f["reward_rates"].keys() for f in samples):
      raise ValueError("cannot rescale an uncaptured reward term")
    if arm != "baseline" and all(f["trigger_time_s"] is None for f in samples):
      raise ValueError(f"{arm}: braking was never triggered")
    after = [f for f in samples if f["time_s"] > decision_time + step_dt * 1e-6]
    if not after:
      raise ValueError(f"{arm}: no measured rewards after the decision")
    terms = {
      name: math.fsum(
        gamma**i * f["reward_rates"][name] * reward_scales.get(name, 1.0) * step_dt
        for i, f in enumerate(after)
      )
      for name in after[0]["reward_rates"]
      if name not in _MOTION_COST_TERMS
    }
    original = math.fsum(
      gamma**i * _base_reward(f, step_dt, reward_scales) for i, f in enumerate(after)
    )
    braking = math.fsum(gamma**i * _braking(f) * step_dt for i, f in enumerate(after))
    events = []
    previous_state = 0.0
    for frame in samples:
      count = frame.get(
        "braking_events", float(frame["motion_state"] == 1 and previous_state != 1)
      )
      if frame["time_s"] > decision_time + step_dt * 1e-6:
        events.append(count)
      previous_state = frame["motion_state"]
    discounted_events = math.fsum(gamma**i * count for i, count in enumerate(events))
    final = samples[-1]
    failed = any(
      bool(final[k]) for k in ("fell_over", "collapsed", "controller_failed")
    )
    held = [f for f in after if f["motion_state"] == 2 and f["holding"]]
    holding_seconds = math.fsum(
      gamma**i * step_dt * (f["motion_state"] == 2 and bool(f["holding"]))
      for i, f in enumerate(after)
    )
    output[arm] = {
      "completion_time_s": final["time_s"],
      "complete": bool(final["complete"]),
      "failed": failed,
      "truncated": bool(final["truncated"]),
      "observed_outcome": (
        "failure"
        if failed
        else "safe_completion"
        if final["complete"]
        else "incomplete"
      ),
      "steps_from_decision": len(after),
      "discounted_original": original,
      "discounted_terms": terms,
      "discounted_braking_seconds": braking,
      "discounted_braking_events": discounted_events,
      "discounted_holding_seconds": holding_seconds,
      "braking_events": sum(events),
      "event_count_source": "native" if "braking_events" in final else "sampled_edges",
      "undiscounted_original": math.fsum(
        _base_reward(f, step_dt, reward_scales) for f in after
      ),
      "braking_seconds": math.fsum(_braking(f) * step_dt for f in after),
      "mean_held_reward_rate": (
        math.fsum(sum(f["reward_rates"].values()) for f in held) / len(held)
        if held
        else None
      ),
      "returns": {
        str(c): {
          "discounted": original - c * braking,
          "undiscounted": math.fsum(
            _base_reward(f, step_dt, reward_scales) - c * _braking(f) * step_dt
            for f in after
          ),
        }
        for c in costs
      },
      "event_returns": {
        str(c): {
          str(event_cost): original - c * braking - event_cost * discounted_events
          for event_cost in event_costs
        }
        for c in costs
      },
      "holding_returns": {
        str(c): {
          str(event_cost): {
            str(hold_cost): original
            - c * braking
            - event_cost * discounted_events
            - hold_cost * holding_seconds
            for hold_cost in holding_costs
          }
          for event_cost in event_costs
        }
        for c in costs
      },
    }
  baseline = output.get("baseline")
  for arm, values in output.items():
    if arm == "baseline" or baseline is None:
      continue
    difference = (
      values["discounted_braking_seconds"] - baseline["discounted_braking_seconds"]
    )
    values["cost_at_equal_baseline_return"] = (
      (values["discounted_original"] - baseline["discounted_original"]) / difference
      if difference > 0
      else None
    )
  return {
    "decision_time_s": decision_time,
    "step_dt": step_dt,
    "gamma": gamma,
    "truncation_bootstrap_included": False,
    "registered_motion_costs_removed": True,
    "reward_scales": reward_scales,
    "arms": output,
  }


def _validate_samples(samples: Sequence[dict[str, Any]], step_dt: float) -> None:
  """Reject incomplete traces, infrastructure failures and reward mismatches."""
  for i, frame in enumerate(samples):
    if not math.isclose(frame["time_s"], (i + 1) * step_dt, abs_tol=1e-6):
      raise ValueError("trace must contain consecutive policy steps from reset")
    if frame["worker_failed"]:
      raise ValueError("worker failure is not a measured physical outcome")
    if not all(
      math.isfinite(v)
      for v in (
        frame["reward"],
        frame["applied_scale"],
        *frame["reward_rates"].values(),
      )
    ):
      raise ValueError("nonfinite reward or applied motion scale")
    expected = math.fsum(frame["reward_rates"].values()) * step_dt
    if not math.isclose(frame["reward"], expected, rel_tol=1e-5, abs_tol=1e-6):
      raise ValueError("reward terms disagree with the dispatched timestep reward")
    if i < len(samples) - 1 and (frame["terminated"] or frame["truncated"]):
      raise ValueError("trace continues beyond its first episode")
  if not (samples[-1]["terminated"] or samples[-1]["truncated"]):
    raise ValueError("trace has no terminal observation")


def _braking(frame: dict[str, Any]) -> float:
  """Charge actual applied slowdown during bilateral Hold."""
  return 1.0 - min(max(frame["applied_scale"], 0.0), 1.0) if frame["holding"] else 0.0


def _base_reward(
  frame: dict[str, Any], step_dt: float, reward_scales: Mapping[str, float]
) -> float:
  """Remove captured motion penalties before applying counterfactual costs."""
  rates = frame["reward_rates"]
  original = frame["reward"] - step_dt * sum(
    rates.get(key, 0.0) for key in _MOTION_COST_TERMS
  )
  return original + step_dt * sum(
    (scale - 1.0) * rates[name] for name, scale in reward_scales.items()
  )
