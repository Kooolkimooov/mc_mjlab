"""Reward preference checks reject stalls, invalid inputs and mixed controllers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evaluation.motion_preferences import check_motion_preferences


def recorded_run(path: Path) -> Path:
  """Write a safe completion and an unnecessary longer hold with captured costs."""
  path.mkdir()
  weights = {
    "contact_dcm_stability": 1.0,
    "termination_penalty": -200.0,
    "braking": -0.03,
    "braking_start": -0.22,
    "holding": -0.05,
  }
  report = {
    "seed": 43,
    "mass_kg": 10,
    "encoder_bias": True,
    "step_dt": 0.02,
    "gamma": 1.0,
    "motion_decision_period_s": 2.0,
    "reward_weights": weights,
    "actual_mass_kg": [10, 10],
    "provenance": {
      "controller_name": "test",
      "files": [],
      "artifacts": [{"sha256": "same"}],
    },
  }
  samples = []
  for arm, steps, dcm in (("baseline", 2, 0.2), ("hold_forever", 5, 0.3)):
    for i in range(steps):
      held = arm == "hold_forever"
      events = float(held and i == 0)
      rates = {
        "contact_dcm_stability": dcm,
        "termination_penalty": 0.0,
        "braking": -0.03 * held,
        "braking_start": -0.22 * events / 0.02,
        "holding": -0.05 * held,
      }
      samples.append(
        {
          "arm": arm,
          "time_s": (i + 1) * 0.02,
          "trigger_time_s": 0.0 if held else None,
          "reward_rates": rates,
          "reward": sum(rates.values()) * 0.02,
          "applied_scale": float(not held),
          "holding": float(held),
          "motion_state": 2 if held else 0,
          "braking_events": events,
          "braking_count": float(held),
          "native_input_scale": float(not held),
          "dispatched_scale": float(not held),
          "worker_failed": 0,
          "controller_failed": 0,
          "fell_over": 0,
          "collapsed": 0,
          "complete": float(not held and i == steps - 1),
          "terminated": not held and i == steps - 1,
          "truncated": held and i == steps - 1,
        }
      )
  (path / "report.json").write_text(json.dumps(report))
  (path / "trace.jsonl").write_text("".join(json.dumps(f) + "\n" for f in samples))
  return path


def test_stall_preference_is_rejected_and_retuned_profile_passes(
  tmp_path: Path,
) -> None:
  run = recorded_run(tmp_path / "run")
  bad = {
    "contact_dcm_stability": 1.0,
    "termination_penalty": -200.0,
    "braking": 0.0,
    "braking_start": 0.0,
    "holding": 0.0,
  }
  assert not check_motion_preferences([run], bad, 0.001)["all_preferences_satisfied"]
  good = {
    **bad,
    "contact_dcm_stability": 0.25,
    "termination_penalty": -400.0,
    "braking": -0.03,
    "braking_start": -0.22,
    "holding": -0.1,
  }
  report = check_motion_preferences([run], good, 0.001)
  assert report["all_preferences_satisfied"]
  assert report["episode_count"] == 2
  assert report["actual_payloads_recorded"]


@pytest.mark.parametrize(
  "fault", ["provenance", "payload", "delivery", "worker", "counter", "timing"]
)
def test_invalid_data_cannot_pass_the_preference_check(
  tmp_path: Path, fault: str
) -> None:
  one = recorded_run(tmp_path / "one")
  two = recorded_run(tmp_path / "two")
  report = json.loads((two / "report.json").read_text())
  samples = [
    json.loads(line) for line in (two / "trace.jsonl").read_text().splitlines()
  ]
  if fault == "provenance":
    report["provenance"]["artifacts"][0]["sha256"] = "changed"
  elif fault == "payload":
    report["actual_mass_kg"][0] = 100
  elif fault == "delivery":
    samples[0]["native_input_scale"] = 0.5
  elif fault == "worker":
    samples[0]["worker_failed"] = 1
  elif fault == "timing":
    report["gamma"] = 0.5
  else:
    samples[-1]["braking_count"] = 2
  (two / "report.json").write_text(json.dumps(report))
  (two / "trace.jsonl").write_text("".join(json.dumps(f) + "\n" for f in samples))
  weights = report["reward_weights"]
  with pytest.raises(ValueError):
    check_motion_preferences([one, two], weights)
