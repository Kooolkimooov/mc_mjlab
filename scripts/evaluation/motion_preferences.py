"""Check motion reward preferences on matched, frozen controller trajectories."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.motion_reward import score_braking
from mc_mjlab.rl.controller_provenance import provenance_signature

PROFILE_TERMS = (
  "contact_dcm_stability",
  "termination_penalty",
  "braking",
  "braking_start",
  "holding",
)


@dataclass
class MotionArm:
  """Keep one fully observed episode and its captured reward configuration."""

  name: str
  manifest: dict[str, Any]
  samples: list[dict[str, Any]]

  def score(self, weights: dict[str, float], time: float) -> float:
    """Rescore captured non-motion terms and actual motion costs from an anchor."""
    captured = self.manifest["reward_weights"]
    scales = {key: weights[key] / captured[key] for key in PROFILE_TERMS[:2]}
    b, e, h = (-weights[key] for key in PROFILE_TERMS[2:])
    report = score_braking(
      self.samples,
      step_dt=self.manifest["step_dt"],
      gamma=self.manifest["gamma"],
      costs=[b],
      event_costs=[e],
      holding_costs=[h],
      reward_scales=scales,
      decision_time_s=time,
    )
    arm = self.samples[0]["arm"]
    return report["arms"][arm]["holding_returns"][str(b)][str(e)][str(h)]

  @property
  def failed(self) -> bool:
    return any(
      self.samples[-1][key] for key in ("fell_over", "collapsed", "controller_failed")
    )

  @property
  def complete(self) -> bool:
    return bool(self.samples[-1]["complete"]) and not self.failed


def check_motion_preferences(
  runs: list[Path], weights: dict[str, float], minimum_margin: float = 0.05
) -> dict[str, Any]:
  """Require safe completion, appropriate normal travel and safe fallback ordering."""
  if not math.isfinite(minimum_margin) or minimum_margin < 0:
    raise ValueError("minimum margin must be finite and nonnegative")
  groups: dict[str, list[MotionArm]] = defaultdict(list)
  signatures = set()
  actual_payloads_recorded = True
  for run in runs:
    manifest = json.loads((run / "report.json").read_text())
    signatures.add(
      (
        provenance_signature(manifest["provenance"]),
        manifest["motion_decision_period_s"],
        manifest["step_dt"],
        manifest["gamma"],
        tuple(
          sorted(
            (name, weight)
            for name, weight in manifest["reward_weights"].items()
            if name not in PROFILE_TERMS
          )
        ),
      )
    )
    identifier = manifest.get(
      "condition_id", f"motion/{manifest['seed']}/{manifest['mass_kg']:g}"
    )
    identifier += f"/bias={manifest['encoder_bias']}"
    frames: dict[str, list[dict]] = defaultdict(list)
    for line in (run / "trace.jsonl").read_text().splitlines():
      frame = json.loads(line)
      frames[frame["arm"]].append(frame)
    for arm, samples in frames.items():
      if arm != "baseline" and samples[-1]["braking_count"] < 1:
        raise ValueError(f"{run}/{arm}: no native braking was measured")
      if any(
        abs(f["native_input_scale"] - f["dispatched_scale"]) > 1e-6 for f in samples
      ):
        raise ValueError(f"{run}/{arm}: native input delivery mismatch")
      if not math.isclose(
        sum(f["braking_events"] for f in samples),
        samples[-1]["braking_count"],
        abs_tol=1e-6,
      ):
        raise ValueError(f"{run}/{arm}: native event count mismatch")
      groups[identifier].append(MotionArm(f"{run.name}/{arm}", manifest, samples))
    actual_payloads_recorded &= "actual_mass_kg" in manifest
    if any(
      not math.isclose(mass, manifest["mass_kg"], rel_tol=1e-4, abs_tol=1e-3)
      for mass in manifest.get("actual_mass_kg", [])
    ):
      raise ValueError(f"{run}: actual payload differs from declared mass")
  if len(signatures) != 1:
    raise ValueError(
      "comparisons require matching controller, timing and other rewards"
    )
  comparisons = []
  outcomes = []
  for identifier, arms in groups.items():
    triggers = [
      f["trigger_time_s"]
      for arm in arms
      for f in arm.samples
      if f["trigger_time_s"] is not None
    ]
    if not triggers:
      raise ValueError(f"{identifier}: no measured intervention")
    start = min(triggers)
    for arm in arms:
      score = arm.score(weights, start)
      outcomes.append(
        {
          "condition": identifier,
          "arm": arm.name,
          "complete": arm.complete,
          "failed": arm.failed,
          "return": score,
        }
      )
    for good in arms:
      for bad in arms:
        if not _preferred(good, bad):
          continue
        for anchor, time in _anchors(good, bad, start).items():
          margin = good.score(weights, time) - bad.score(weights, time)
          comparisons.append(
            {
              "condition": identifier,
              "preferred": good.name,
              "other": bad.name,
              "anchor": anchor,
              "time_s": time,
              "margin": margin,
              "passed": margin >= minimum_margin,
            }
          )
  if not comparisons:
    raise ValueError("no measured outcome preferences to check")
  return {
    "weights": weights,
    "minimum_required_margin": minimum_margin,
    "minimum_observed_margin": min(row["margin"] for row in comparisons),
    "all_preferences_satisfied": all(row["passed"] for row in comparisons),
    "condition_count": len(groups),
    "episode_count": len(outcomes),
    "actual_payloads_recorded": actual_payloads_recorded,
    "truncation_bootstrap_included": False,
    "outcomes": outcomes,
    "comparisons": comparisons,
  }


def _preferred(good: MotionArm, bad: MotionArm) -> bool:
  """Rank observed safety outcomes without assuming that more braking is better."""
  if good is bad or good.failed:
    return False
  if good.complete:
    return not bad.complete or (
      good.samples[0]["arm"] == "baseline" and bad.samples[0]["arm"] != "baseline"
    )
  return bad.failed


def _anchors(good: MotionArm, bad: MotionArm, start: float) -> dict[str, float]:
  """Include post-hold decisions when both compared arms actually held."""
  anchors = {"intervention": start}
  held_good = [f for f in good.samples if f["motion_state"] == 2]
  held_bad = [f for f in bad.samples if f["motion_state"] == 2]
  if not held_good or not held_bad:
    return anchors
  end = min(good.samples[-1]["time_s"], bad.samples[-1]["time_s"])
  held_time = max(held_good[0]["time_s"], held_bad[0]["time_s"])
  if held_time < end:
    anchors["held"] = held_time
  for arm, held in ((good, held_good), (bad, held_bad)):
    resumed = [
      f
      for f in arm.samples
      if f["time_s"] > held[-1]["time_s"] and f["motion_state"] == 0
    ]
    if resumed and resumed[0]["time_s"] < end:
      anchors[f"release/{arm.name}"] = resumed[0]["time_s"]
  return anchors
