"""Checkpoint ablations isolate motion and joint channels without mutating input."""

from __future__ import annotations

import torch
from probe_locomanip_policy_channels import select_channels, summarize


def test_select_channels_preserves_each_cohort_and_original_actions() -> None:
  original = torch.arange(24, dtype=torch.float32).reshape(8, 3)
  untouched = original.clone()
  selected = select_channels(original)
  torch.testing.assert_close(original, untouched)
  torch.testing.assert_close(selected[0::4], torch.zeros(2, 3))
  torch.testing.assert_close(selected[1::4], original[1::4])
  torch.testing.assert_close(selected[2::4, :-1], original[2::4, :-1])
  torch.testing.assert_close(selected[2::4, -1], torch.zeros(2))
  torch.testing.assert_close(selected[3::4, :-1], torch.zeros(2, 2))
  torch.testing.assert_close(selected[3::4, -1], original[3::4, -1])

  with torch.inference_mode():
    inferred = original.clone()
  selected = select_channels(inferred)
  selected[1] = 0.0
  torch.testing.assert_close(selected[1], torch.zeros(3))


def test_braking_cycles_are_counted_separately_from_explicit_holds() -> None:
  frames = [
    {
      "mass_kg": 100.0,
      "arm": "policy",
      "time_s": (i + 1) * 0.02,
      "complete": 0.0,
      "fell_over": 0.0,
      "collapsed": 0.0,
      "controller_failed": 0.0,
      "worker_failed": 0.0,
      "robot_x": 0.01 * i,
      "cart_x": 0.02 * i,
      "reference_x": 0.02 * i,
      "root_z": 0.79,
      "tilt_rad": 0.05,
      "motion_state": state,
      "motion_request": 0.2,
      "applied_scale": 0.8,
      "joint_request_norm": 0.1,
      "native_input_scale": 0.8,
      "dispatched_scale": 0.8,
    }
    for i, state in enumerate([0, 1, 1, 0, 1, 0, 2])
  ]
  result = summarize(frames, 0.02)
  assert result["braking_starts"] == 2
  assert result["braking_seconds"] == 0.06
  assert result["holding_seconds"] == 0.02
  assert result["explicit_hold_requests"] == 0
  assert result["input_delivery_verified"]
  assert not result["complete"]
  assert not result["failed"]
