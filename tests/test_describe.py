"""Configuration inspection contracts without simulation or controller startup."""

from __future__ import annotations

import builtins
import copy
import ctypes
import dataclasses
import functools
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from enum import Enum
from pathlib import Path
from typing import Any

import mjlab  # noqa: F401
import numpy as np
import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions.actions import JointPositionActionCfg
from mjlab.managers import (
  CurriculumTermCfg,
  EventTermCfg,
  MetricsTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation
from mjlab.tasks import registry

import mc_rtc_interface as native
from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionActionCfg,
)
from mc_mjlab.cli import describe
from mc_mjlab.cli.rendering import render_text
from mc_mjlab.cli.report import collect_report
from mc_mjlab.cli.serialization import qualified_name, serialize


def _probe(env: Any, gain: float = 0.25, selection: slice = slice(None)) -> Any:
  """Expose signature defaults without allowing inspection to execute the term."""
  raise AssertionError("inspection called a manager term")


@pytest.fixture
def task(monkeypatch: pytest.MonkeyPatch) -> str:
  """Register distinct train/play configs in a registry isolated from other tests."""
  monkeypatch.setattr(registry, "_REGISTRY", registry._REGISTRY.copy())
  disabled: Any = None
  env = ManagerBasedRlEnvCfg(
    decimation=5,
    scene=SceneCfg(num_envs=4),
    episode_length_s=1.01,
    actions={
      "joint": JointPositionActionCfg(entity_name="robot", actuator_names=(".*",))
    },
    observations={
      "actor": ObservationGroupCfg(
        terms={
          "z_first": ObservationTermCfg(func=_probe, params={"gain": 0.5}),
          "a_second": ObservationTermCfg(func=_probe),
          "disabled": disabled,
        }
      )
    },
    rewards={
      "z_zero": RewardTermCfg(func=_probe, weight=0.0),
      "a_negative": RewardTermCfg(func=_probe, weight=-2.0),
      "disabled": disabled,
    },
    terminations={"timeout": TerminationTermCfg(func=_probe, time_out=True)},
    events={"reset": EventTermCfg(func=_probe, mode="reset")},
    curriculum={"difficulty": CurriculumTermCfg(func=_probe)},
    metrics={"quality": MetricsTermCfg(func=_probe, reduce="last")},
  )
  play = copy.deepcopy(env)
  play.scene.num_envs = 1
  play.observations["actor"].enable_corruption = False
  registry.register_mjlab_task(
    "Test-Describe", env, play, RslRlOnPolicyRunnerCfg(num_steps_per_env=12)
  )
  return "Test-Describe"


@pytest.fixture(autouse=True)
def forbid_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
  """Fail immediately if any inspection test constructs a runtime object."""

  def forbidden(*args: Any, **kwargs: Any) -> None:
    raise AssertionError("inspection constructed a runtime object")

  for cls in (ManagerBasedRlEnv, Scene, Simulation, MjlabOnPolicyRunner):
    monkeypatch.setattr(cls, "__init__", forbidden)
  monkeypatch.setattr(native, "ControllersManager", forbidden)


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("play", [False, True])
def test_json_selection_overrides_and_registry_immutability(
  task: str, full: bool, play: bool, capfd: pytest.CaptureFixture[str]
) -> None:
  """Override copies while preserving all registered fields and configured order."""
  before = serialize(registry.load_env_cfg(task, play=play))
  before_agent = serialize(registry.load_rl_cfg(task))
  args = [
    task,
    "--format",
    "json",
    "--env.scene.num-envs",
    "8",
    "--agent.algorithm.learning-rate",
    "0.0003",
  ]
  args += ["--full"] if full else []
  args += ["--play"] if play else []
  describe.main(args)
  report = json.loads(capfd.readouterr().out)

  assert report["inspection"] == "configuration"
  assert report["mode"] == ("play" if play else "training")
  assert report["detail"] == ("full" if full else "summary")
  assert report["scene"]["num_envs"] == 8
  assert report["training"]["algorithm"]["learning_rate"] == 0.0003
  assert report["training"]["rollout_transitions"] == 96
  assert report["runner"] == qualified_name(MjlabOnPolicyRunner)
  assert report["timing"]["policy_dt_s"] == pytest.approx(0.01)
  assert report["timing"]["episode_step_limit"] == 101
  assert report["observations"]["actor"]["terms"] == ["z_first", "a_second", "disabled"]
  assert report["observations"]["actor"]["disabled_terms"] == ["disabled"]
  assert list(report["rewards"]) == ["z_zero", "a_negative", "disabled"]
  assert report["rewards"]["z_zero"]["weight"] == 0.0
  assert report["rewards"]["disabled"] is None
  assert report["terminations"]["timeout"]["time_out"] is True
  assert "controllers" not in report
  assert ("configuration" in report) == full
  if full:
    env = report["configuration"]["env"]
    assert set(env) == {
      field.name for field in dataclasses.fields(ManagerBasedRlEnvCfg)
    }
    term = env["observations"]["actor"]["terms"]["z_first"]
    assert term["func"]["callable"] == qualified_name(_probe)
    assert term["func"]["parameter_defaults"]["gain"] == 0.25
    assert term["params"]["gain"] == 0.5
    assert env["scene"]["num_envs"] == 8
  assert serialize(registry.load_env_cfg(task, play=play)) == before
  assert serialize(registry.load_rl_cfg(task)) == before_agent


def test_registered_play_selection(
  task: str, capfd: pytest.CaptureFixture[str]
) -> None:
  """Use the registered play environment rather than a modified training copy."""
  describe.main([task, "--play", "--format", "json"])
  assert json.loads(capfd.readouterr().out)["scene"]["num_envs"] == 1
  describe.main([task, "--format", "json"])
  assert json.loads(capfd.readouterr().out)["scene"]["num_envs"] == 4


@pytest.mark.parametrize("full", [False, True])
def test_text_output(task: str, full: bool, capfd: pytest.CaptureFixture[str]) -> None:
  """Keep text compact by default and include all fields on request."""
  describe.main([task] + (["--full"] if full else []))
  output = capfd.readouterr().out
  assert "Configuration inspection: Test-Describe" in output
  assert "z_zero:" in output and "weight=0.0" in output
  assert output.index("z_first") < output.index("a_second")
  assert "disabled (None)" in output
  assert ("Complete configuration" in output) == full
  assert ("parameter_defaults" in output) == full


@pytest.mark.parametrize(
  ("mode", "colored"),
  [
    ("auto", False),
    ("always", True),
    ("never", False),
  ],
)
def test_text_color_modes(
  task: str, mode: str, colored: bool, capfd: pytest.CaptureFixture[str]
) -> None:
  """Color text on request while keeping redirected output plain by default."""
  describe.main([task, "--color", mode])
  output = capfd.readouterr().out
  assert ("\x1b[" in output) is colored
  if colored:
    assert "\x1b[1;36mConfiguration inspection:" in output
    assert "\x1b[1;34mActions:\x1b[0m" in output
    assert "\x1b[33mz_zero\x1b[0m" in output


def test_auto_color_respects_terminal_preferences(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Honor terminal, NO_COLOR and dumb-terminal settings in automatic mode."""
  monkeypatch.delenv("NO_COLOR", raising=False)
  monkeypatch.setenv("TERM", "xterm-256color")
  assert describe._use_color("auto", True)
  assert not describe._use_color("auto", False)
  monkeypatch.setenv("NO_COLOR", "1")
  assert not describe._use_color("auto", True)
  assert describe._use_color("always", False)
  assert not describe._use_color("never", True)
  monkeypatch.delenv("NO_COLOR")
  monkeypatch.setenv("TERM", "dumb")
  assert not describe._use_color("auto", True)


def test_json_ignores_color_mode(task: str, capfd: pytest.CaptureFixture[str]) -> None:
  """Keep JSON machine readable even when text color was explicitly requested."""
  describe.main([task, "--format", "json", "--color", "always"])
  output = capfd.readouterr().out
  assert "\x1b[" not in output
  assert json.loads(output)["task_id"] == task


# Runs in a fresh interpreter: this one has already imported everything help must not.
HELP_WITHOUT_LOADING = """
import importlib.abc, sys
from importlib.metadata import entry_points

FORBIDDEN = ("mjlab", "mujoco", "torch", "mc_rtc", "mc_rtc_interface",
             "mc_mjlab.actions", "mc_mjlab.tasks")
attempted = []

class Forbid(importlib.abc.MetaPathFinder):
  def find_spec(self, name, path=None, target=None):
    if any(name == root or name.startswith(root + ".") for root in FORBIDDEN):
      attempted.append(name)
      raise ImportError(name)

sys.meta_path.insert(0, Forbid())
(command,) = entry_points(group="console_scripts", name="describe")
code = 0
try:
  command.load()(["--help"])
except SystemExit as done:
  code = done.code
sys.exit(f"help imported {attempted}" if attempted else code)
"""


def test_top_level_help_skips_loading() -> None:
  """Expose the installed command's help without importing what tasks would load."""
  result = subprocess.run(
    [sys.executable, "-c", HELP_WITHOUT_LOADING], capture_output=True, text=True
  )
  assert result.returncode == 0, result.stderr
  options = ("--full", "--play", "--format", "--color")
  assert all(option in result.stdout for option in options)


def test_task_help_exposes_overrides(
  task: str, capfd: pytest.CaptureFixture[str]
) -> None:
  """Include concrete nested environment and agent arguments in task help."""
  with pytest.raises(SystemExit) as error:
    describe.main([task, "--help"])
  assert error.value.code == 0
  output = capfd.readouterr().out
  assert "--env.scene.num-envs" in output
  assert "--agent.algorithm.learning-rate" in output
  assert "--format" in output


@pytest.mark.parametrize(
  "args",
  [
    ["Unknown-Describe-Task", "--format", "json"],
    ["Test-Describe", "--format", "yaml"],
    ["Test-Describe", "--format", "json", "--env.scene.num-envs", "oops"],
    ["Test-Describe", "--format", "json", "--env.not-a-field", "1"],
    ["Test-Describe", "--format", "json", "--env.scene.num-envs"],
    ["Test-Describe", "--format", "json", "--env.sim.mujoco.timestep", "0"],
    [],
  ],
)
def test_errors_leave_stdout_empty(
  task: str, args: list[str], capfd: pytest.CaptureFixture[str]
) -> None:
  """Fail invalid invocations without producing partial JSON or tracebacks."""
  with pytest.raises(SystemExit) as error:
    describe.main(args)
  assert error.value.code != 0
  captured = capfd.readouterr()
  assert captured.out == ""
  assert captured.err and "Traceback" not in captured.err


def test_loading_chatter_stays_on_stderr(
  task: str, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
  """Route both Python and native registration output away from JSON stdout."""
  original_load = describe._load_task

  def noisy_load(task_id: str, play: bool) -> Any:
    print("Python registration chatter")
    os.write(1, b"native registration chatter\n")
    ctypes.CDLL(None).puts(b"buffered C registration chatter")
    return original_load(task_id, play)

  monkeypatch.setattr(describe, "_load_task", noisy_load)
  describe.main([task, "--format", "json"])
  captured = capfd.readouterr()
  assert json.loads(captured.out)["task_id"] == task
  assert "Python registration chatter" in captured.err
  assert "native registration chatter" in captured.err
  assert "buffered C registration chatter" in captured.err


def test_generic_inspection_does_not_import_controller_actions(
  task: str, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Keep generic reports independent of mc_rtc action imports."""
  original_import = builtins.__import__

  def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
    assert not name.startswith("mc_mjlab.actions")
    return original_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", guarded_import)
  report = collect_report(
    task, registry.load_env_cfg(task), registry.load_rl_cfg(task), MjlabOnPolicyRunner
  )
  assert "controllers" not in report


def test_every_controller_action_and_absent_files(
  task: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Discover files for arbitrary action names while tolerating missing sources."""
  env = registry.load_env_cfg(task)
  project = tmp_path / "mc_rtc.yaml"
  project.write_text("Enabled: ExampleController\nMainRobot: EXAMPLE\n")
  installed = tmp_path / "lib/mc_controller/etc/ExampleController.yaml"
  installed.parent.mkdir(parents=True)
  installed.write_text("test: true\n")
  monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path / "lib"))
  monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
  env.actions = {
    "arbitrary": McRtcResidualJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      frameskip=3,
      mc_rtc_config_path=str(project),
      pd_gains_path=str(tmp_path / "missing.dat"),
    ),
    "second": McRtcResidualJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      mc_rtc_config_path=str(tmp_path / "absent.yaml"),
    ),
  }
  report = collect_report(task, env, registry.load_rl_cfg(task), MjlabOnPolicyRunner)
  assert list(report["controllers"]) == ["arbitrary", "second"]
  first = report["controllers"]["arbitrary"]
  assert first["controller_name"] == "ExampleController"
  assert first["main_robot"] == "EXAMPLE"
  assert first["installed_controller_configs"][0]["path"] == str(installed)
  assert first["installed_controller_configs"][0]["present"] is True
  assert all(not record["present"] for record in first["user_controller_configs"])
  assert first["files"][-1]["role"] == "pd_gains"
  assert first["files"][-1]["present"] is False
  assert report["actions"]["arbitrary"]["controller_period_s"] == pytest.approx(0.006)
  second = report["controllers"]["second"]
  assert second["controller_name"] is None and second["config_errors"]
  assert second["files"][0]["present"] is False
  colored = render_text(report, color=True)
  assert "\x1b[1;97;44m[present]\x1b[0m" in colored
  assert "\x1b[1;30;43m[missing]\x1b[0m" in colored
  plain = render_text(report)
  assert "[present]" in plain and "[missing]" in plain


class _Choice(Enum):
  """A representative configuration enumeration."""

  FIRST = "first"


def test_serialization_of_nested_values() -> None:
  """Serialize nested native values and partial bindings with stable callable defaults."""
  values = {
    "path": Path("relative/config.yaml"),
    "enum": _Choice.FIRST,
    "array": np.array([[1, 2], [3, 4]]),
    "scalar": np.float32(0.5),
    "tensor": torch.tensor([1.0, 2.0], requires_grad=True),
    "slice": slice(1, None, 2),
    "partial": functools.partial(_probe, "bound-env", gain=0.75),
    "type": dict,
    "nonfinite": [float("inf"), float("nan")],
    "dtype": torch.float32,
    "opaque": object(),
  }
  result = serialize(values)
  encoded = json.dumps(result, allow_nan=False)
  assert "0x" not in encoded
  assert list(result) == list(values)
  assert result["path"] == "relative/config.yaml"
  assert result["enum"]["name"] == "FIRST"
  assert result["array"] == [[1, 2], [3, 4]]
  assert result["tensor"] == [1.0, 2.0]
  assert result["scalar"] == 0.5
  assert result["slice"] == {"slice": [1, None, 2]}
  assert result["partial"]["callable"] == qualified_name(_probe)
  assert result["partial"]["partial_args"] == ["bound-env"]
  assert result["partial"]["partial_keywords"] == {"gain": 0.75}
  assert result["partial"]["parameter_defaults"]["gain"] == 0.25
  assert result["type"] == {"callable": "builtins.dict"}


def test_recursive_values() -> None:
  """Bound recursive containers without mistaking repeated sibling values for cycles."""
  repeated = {"value": 1}
  values = [repeated, repeated]
  values.append(values)
  result = serialize(values)
  assert result[:2] == [{"value": 1}, {"value": 1}]
  assert result[-1] == {"recursive_reference": "builtins.list"}


@pytest.fixture
def installed_command() -> Iterator[list[str]]:
  """Find the installed describe command in this uv environment."""
  command = Path(sys.executable).parent / "describe"
  assert command.is_file(), "install project console scripts before testing"
  yield [str(command)]


def test_custom_runner_selection(task: str, capfd: pytest.CaptureFixture[str]) -> None:
  """Report a registered custom runner without constructing it."""

  class CustomRunner:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
      raise AssertionError("inspection constructed a custom runner")

  registry.register_mjlab_task(
    "Test-Describe-Custom",
    registry.load_env_cfg(task),
    registry.load_env_cfg(task, play=True),
    registry.load_rl_cfg(task),
    runner_cls=CustomRunner,
  )
  describe.main(["Test-Describe-Custom", "--format", "json"])
  assert json.loads(capfd.readouterr().out)["runner"] == qualified_name(CustomRunner)


def test_mjlab_boolean_and_collection_overrides(
  task: str, capfd: pytest.CaptureFixture[str]
) -> None:
  """Honor mjlab's explicit boolean values and Python collection syntax."""
  describe.main(
    [
      task,
      "--format",
      "json",
      "--env.observations.actor.enable-corruption",
      "True",
      "--agent.actor.hidden-dims",
      "(32,16)",
    ]
  )
  report = json.loads(capfd.readouterr().out)
  assert report["observations"]["actor"]["enable_corruption"] is True
  assert report["training"]["actor"]["hidden_dims"] == [32, 16]


@pytest.mark.parametrize("family", ["Residual-Balance", "Zero-Residual", "Locomanip"])
def test_installed_command_against_discovered_tasks(
  installed_command: list[str], family: str
) -> None:
  """Smoke-test available controller-backed task families through the installed entry point."""
  tasks = [
    task for task in registry.list_tasks() if task.startswith(f"Mc-Mjlab-{family}-")
  ]
  if not tasks:
    pytest.skip(f"no {family} task registered in this workspace")
  task = tasks[0]
  result = subprocess.run(
    installed_command + [task, "--play", "--full", "--format", "json"],
    capture_output=True,
    text=True,
    timeout=60,
  )
  assert result.returncode == 0, result.stderr
  report = json.loads(result.stdout)
  assert report["task_id"] == task
  assert report["controllers"]
  assert "configuration" in report
  assert "0x" not in result.stdout
