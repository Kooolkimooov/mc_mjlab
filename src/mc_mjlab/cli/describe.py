"""Describe registered task configuration without starting simulation."""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import os
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass
class TaskConfig:
  """Expose concrete copied environment and agent defaults through Tyro."""

  env: Any
  agent: Any


def main(argv: Sequence[str] | None = None) -> None:
  """Parse command options, apply task overrides and emit one complete report."""
  parser = _parser()
  options, overrides = parser.parse_known_args(argv)
  if options.task_id is None:
    if options.help:
      parser.print_help()
      return
    parser.error("a TaskID is required; use 'uv run list-envs' to discover tasks")

  try:
    with _loading_output():
      defaults, runner_cls, tyro_flags = _load_task(options.task_id, options.play)

    if options.help:
      parser.print_help()
      _parse_overrides(defaults, ["--help"], options.task_id, tyro_flags)
      return

    with _loading_output():
      config = _parse_overrides(defaults, overrides, options.task_id, tyro_flags)
      from mc_mjlab.cli.report import collect_report

      report = collect_report(
        options.task_id,
        config.env,
        config.agent,
        runner_cls,
        play=options.play,
        full=options.full,
      )

      if options.format == "json":
        output = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
      else:
        from mc_mjlab.cli.rendering import render_text

        output = render_text(
          report, color=_use_color(options.color, sys.stdout.isatty())
        )
  except (KeyError, ValueError, OSError, ImportError) as error:
    parser.exit(2, f"describe: error: {error}\n")

  print(output)


def _parser() -> argparse.ArgumentParser:
  """Build lightweight command help before importing mjlab or its tasks."""
  parser = argparse.ArgumentParser(
    prog="describe",
    description="Inspect registered task configuration without starting simulation.",
    epilog=(
      "Use 'describe TaskID --help' for --env.* and --agent.* Tyro overrides. "
      "Registration prerequisites still apply; first-use asset linking is possible."
    ),
    add_help=False,
    allow_abbrev=False,
  )
  parser.add_argument(
    "task_id", nargs="?", metavar="TaskID", help="registered mjlab task"
  )
  parser.add_argument(
    "--full", action="store_true", help="include every env/agent field"
  )
  parser.add_argument(
    "--play", action="store_true", help="inspect the play environment"
  )
  parser.add_argument(
    "--format",
    choices=("text", "json"),
    default="text",
    help="output format (default: text)",
  )
  parser.add_argument(
    "--color",
    choices=("auto", "always", "never"),
    default="auto",
    help="text colors (default: auto; ignored for JSON)",
  )
  parser.add_argument(
    "-h", "--help", action="store_true", help="show command or task help"
  )
  return parser


def _use_color(mode: str, is_terminal: bool) -> bool:
  """Enable ANSI colors for terminals unless the user or environment opts out."""
  if mode == "always":
    return True
  if mode == "never" or "NO_COLOR" in os.environ:
    return False
  return is_terminal and os.environ.get("TERM") != "dumb"


def _load_task(task_id: str, play: bool) -> tuple[TaskConfig, type, Any]:
  """Import mjlab first, then use public copy loaders and the standard runner fallback."""
  import mjlab
  import mjlab.tasks  # noqa: F401
  from mjlab.rl import MjlabOnPolicyRunner
  from mjlab.tasks.registry import (
    list_tasks,
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
  )

  if task_id not in list_tasks():
    raise ValueError(f"unknown task '{task_id}'; use 'uv run list-envs'")
  defaults = TaskConfig(
    env=load_env_cfg(task_id, play=play), agent=load_rl_cfg(task_id)
  )
  return defaults, load_runner_cls(task_id) or MjlabOnPolicyRunner, mjlab.TYRO_FLAGS


def _parse_overrides(
  defaults: TaskConfig, args: Sequence[str], task_id: str, flags: Any
) -> TaskConfig:
  """Apply mjlab's Tyro conventions to the task's concrete dataclass defaults."""
  import tyro

  return tyro.cli(
    TaskConfig,
    args=args,
    default=defaults,
    config=flags,
    prog=f"describe {task_id}",
    description="Environment and agent configuration overrides.",
  )


@contextlib.contextmanager
def _loading_output() -> Iterator[None]:
  """Route Python and native loading chatter to stderr while building the report."""
  fflush = ctypes.CDLL(None).fflush
  sys.stdout.flush()
  fflush(None)
  saved_stdout = os.dup(1)
  try:
    os.dup2(2, 1)
    with contextlib.redirect_stdout(sys.stderr):
      yield
  finally:
    sys.stderr.flush()
    fflush(None)
    os.dup2(saved_stdout, 1)
    os.close(saved_stdout)


if __name__ == "__main__":
  main()
