"""Safety CLI preflights, tuning ranks and probe paths remain unambiguous."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import requires_controller

from mc_mjlab.tasks.locomanip_safety.locomanip_safety_env_cfg import SAFETY_CONFIG

pytestmark = requires_controller(SAFETY_CONFIG)


def tuning_rows(repetitions: int, failing: bool) -> list[dict[str, Any]]:
  """Make a complete small light-and-transition sweep."""
  return [
    {
      "mass_kg": mass,
      "complete": not (failing and mass == 550),
      "observed": True,
      "worker_failure": False,
      "combined_failure": failing and mass == 550,
      "terminal_position_error_m": 0.1,
    }
    for mass in (10, 100, 500, 525, 550)
    for _ in range(repetitions)
  ]


def test_rank_tunings_accepts_two_repeats_and_a_never_failing_tuning() -> None:
  from evaluation.locomanip_tuning import rank_tunings

  ranks = rank_tunings(
    {"all-safe": tuning_rows(2, False), "transition": tuning_rows(2, True)}, 2
  )
  assert [row["tuning"] for row in ranks] == ["all-safe", "transition"]
  assert ranks[0]["first_failing_mass_kg"] is None
  assert ranks[0]["threshold_midpoint_kg"] is None
  assert ranks[1]["threshold_midpoint_kg"] == 537.5
  assert rank_tunings({"short": tuning_rows(2, False)}, 3) == []


def test_variant_accepts_an_empty_user_yaml_without_demo_fsm(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  from evaluation.locomanip_tuning import write_variant

  user = tmp_path / ".config" / "mc_rtc"
  user.mkdir(parents=True)
  (user / "mc_rtc.yaml").write_text("")
  monkeypatch.setattr(Path, "home", lambda: tmp_path)
  project = tmp_path / "project.yaml"
  project.write_text("Enabled: LocomanipController\n")
  effective = {
    "states": {"LMC::PushCart_": {"configs": {}}},
    "transitions": {},
    "init": "LMC::PushCart_",
    "CompletionState": "LMC::PushCart_",
  }
  target = tmp_path / "variant.yaml"
  write_variant(project, effective, "current", target)
  result = yaml.safe_load(target.read_text())
  assert result["OverwriteConfigList"]["MjlabSafetyFrozen"]["DemoFSM"]["init"] == (
    "MjlabSafety::LMC::PushCart_"
  )


def test_checkpoint_steps_resolve_final_and_preflight_missing_run(
  tmp_path: Path,
) -> None:
  from screen_locomanip_safety import checkpoint_steps

  params = tmp_path / "params"
  params.mkdir()
  (params / "agent.yaml").write_text("max_iterations: 600\n")
  for step in (200, 400):
    (tmp_path / f"model_{step}.pt").touch()
  with pytest.raises(ValueError, match="final checkpoint"):
    checkpoint_steps(tmp_path)
  (tmp_path / "model_599.pt").touch()
  assert checkpoint_steps(tmp_path) == (200, 400, 599)


def test_screen_uses_absolute_script_from_any_cwd(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  import screen_locomanip_safety as screen

  run_dir = tmp_path / "run"
  params = run_dir / "params"
  params.mkdir(parents=True)
  (params / "agent.yaml").write_text("max_iterations: 201\n")
  (run_dir / "model_200.pt").touch()
  calls = []

  def fake_subprocess(command: list[str], **kwargs: Any) -> None:
    calls.append((command, kwargs))
    output = Path(command[command.index("--output") + 1])
    output.mkdir(parents=True)
    (output / "report.json").write_text(
      json.dumps(
        {
          "complete_matched_data": True,
          "per_mass_protection": True,
          "no_mass_regression": True,
          "pooled": {"relative_reduction": -0.1},
        }
      )
    )

  monkeypatch.setattr(screen.subprocess, "run", fake_subprocess)
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "screen_locomanip_safety.py",
      "--run-dir",
      str(run_dir),
      "--output",
      str(tmp_path / "out"),
    ],
  )
  monkeypatch.chdir(tmp_path)
  screen.main()
  assert len(calls) == 1
  assert calls[0][0][3] == str(screen.EVALUATOR)
  assert calls[0][1]["cwd"] == screen.REPO_ROOT
  assert (
    json.loads((tmp_path / "out" / "selection.json").read_text())["selected_step"]
    is None
  )


def test_decimal_mass_keeps_every_schedule_filename_distinct(tmp_path: Path) -> None:
  from probe_locomanip_reactive import probe_paths

  paths = [
    probe_paths(tmp_path, 537.5, name) for name in ("constant", "pulse", "reactive")
  ]
  assert len({path for pair in paths for path in pair}) == 6
  assert paths[0][0].name == "537.5-constant.jsonl"


def test_training_entry_point_preflights_before_delegation(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  import train_locomanip_safety as training

  calls = []
  monkeypatch.setattr(sys, "argv", ["train_locomanip_safety.py", training.TASK_ID])
  monkeypatch.setattr(training, "authority_report_path", lambda: tmp_path)
  monkeypatch.setattr(
    training, "validate_authority_report", lambda path: calls.append(("gate", path))
  )
  monkeypatch.setattr(training, "train_main", lambda: calls.append(("train", None)))
  training.main()
  assert calls == [("gate", tmp_path), ("train", None)]


def test_evaluator_batches_mixed_masses_and_rejects_bad_cli_inputs(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  import evaluate_locomanip_safety as evaluator

  calls = []

  def fake_run(task: str, rows: list[dict[str, Any]], **kwargs: Any) -> dict:
    calls.append({row["mass_kg"] for row in rows})
    return {
      "rows": [
        {
          **row,
          "observed": True,
          "worker_failure": False,
          "combined_failure": False,
          "complete": True,
          "terminations": {},
        }
        for row in rows
      ]
    }

  monkeypatch.setattr(evaluator, "run_conditions", fake_run)
  arguments = [
    "evaluate_locomanip_safety.py",
    "--task",
    "other-task",
    "--output",
    str(tmp_path),
    "--masses",
    "10",
    "100",
    "--transition-masses",
    "100",
    "--repetitions",
    "2",
    "--batch-size",
    "8",
  ]
  monkeypatch.setattr(sys, "argv", arguments)
  evaluator.main()
  assert calls == [{10.0, 100.0}]
  assert json.loads((tmp_path / "report.json").read_text())["complete_matched_data"]

  monkeypatch.setattr(sys, "argv", [*arguments, "--seeds", "42", "42"])
  with pytest.raises(SystemExit):
    evaluator.main()
  monkeypatch.setattr(sys, "argv", [*arguments[:], "--repetitions", "0"])
  with pytest.raises(SystemExit):
    evaluator.main()
