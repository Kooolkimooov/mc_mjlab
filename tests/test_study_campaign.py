"""Bounded campaign scheduling must drain milestones and report missing results."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import Mock

import pytest
import run_locomanip_study as campaign


@pytest.mark.parametrize("missing", [False, True])
def test_completed_training_drains_saved_milestones(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: bool
) -> None:
  run = tmp_path / "training" / "task" / "only-run"
  run.mkdir(parents=True)
  expected = list(campaign.EVALUATION_ITERATIONS)
  if missing:
    expected.remove(400)
  for iteration in expected:
    (run / f"model_{iteration}.pt").write_bytes(b"checkpoint")
  visited = []

  def evaluate(
    checkpoint: Path, iteration: int, args: argparse.Namespace, state: dict
  ) -> None:
    visited.append(iteration)
    state["evaluations"].append({"iteration": iteration, "exit_code": 0})

  monkeypatch.setattr(campaign, "evaluate", evaluate)
  monkeypatch.setattr(campaign.time, "sleep", lambda duration: None)
  state = {"task": "task", "evaluations": []}
  campaign.monitor(Mock(poll=lambda: 0), argparse.Namespace(output=tmp_path), state)
  assert visited == expected
  assert state["status"] == ("needs_attention" if missing else "run_finished")
  assert state["missing_evaluations"] == ([400] if missing else [])


def test_ambiguous_run_directory_is_rejected(tmp_path: Path) -> None:
  for name in ("first", "second"):
    (tmp_path / "training" / "task" / name).mkdir(parents=True)
  with pytest.raises(RuntimeError, match="more than one"):
    campaign.monitor(
      Mock(poll=lambda: 0),
      argparse.Namespace(output=tmp_path),
      {"task": "task"},
    )
