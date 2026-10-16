"""Render configuration reports as compact terminal text."""

from __future__ import annotations

import json
import textwrap
from typing import Any

_SECTIONS = {
  "Actions",
  "Observations",
  "Rewards",
  "Terminations",
  "Commands",
  "Events",
  "Curriculum",
  "Metrics",
  "Recorders",
  "Training",
  "Controllers",
}


def render_text(report: dict[str, Any], *, color: bool = False) -> str:
  """Render summary fields in configured order and append full details when selected."""
  lines = [
    f"Configuration inspection: {report['task_id']} ({report['mode']}; {report['detail']})",
    f"Runner: {report['runner']}",
  ]
  scene = report["scene"]
  lines += [
    f"Environments: {scene['num_envs']} | Entities: {_names(scene['entities'])}",
    f"Sensors: {_names(scene['sensors'])}",
  ]
  timing = report["timing"]
  lines += [
    f"Physics: {timing['physics_dt_s']:g} s | Policy: {timing['policy_dt_s']:g} s "
    f"({timing['policy_frequency_hz']:g} Hz; decimation={timing['decimation']})",
    f"Episode: {timing['episode_duration_s']:g} s "
    f"({timing['episode_step_limit']} configured steps)",
  ]

  for section in (
    "actions",
    "observations",
    "rewards",
    "terminations",
    "commands",
    "events",
    "curriculum",
    "metrics",
    "recorders",
  ):
    lines.append(
      f"\n{section.title()}:" if report[section] else f"\n{section.title()}: none"
    )
    for name, term in report[section].items():
      lines.extend(_term_lines(name, term))

  lines.append("\nTraining:")
  for name, value in report["training"].items():
    lines.extend(_wrapped(f"  {name}: {_compact(value)}"))

  if "controllers" in report:
    lines.extend(_controller_lines(report["controllers"]))
  lines += [
    "\nConfigured values; dimensions, entity indices, live curricula and merged "
    "controller YAML are outside this inspection.",
  ]
  if "configuration" in report:
    lines += [
      "\nComplete configuration (callable parameter_defaults are declared defaults; "
      "params and partial arguments are explicit):",
      json.dumps(
        report["configuration"], indent=2, ensure_ascii=False, allow_nan=False
      ),
    ]
  if color:
    lines = [_colorize(line) for line in lines]
  return "\n".join(lines)


def _colorize(line: str) -> str:
  """Highlight headings, term names and file status after line wrapping."""
  prefix = "\n" if line.startswith("\n") else ""
  body = line[len(prefix) :]
  if body.startswith("Configuration inspection:"):
    return prefix + _ansi(body, "1;36")
  heading, separator, remainder = body.partition(":")
  if heading in _SECTIONS:
    return prefix + _ansi(heading + separator, "1;34") + remainder
  if body and not body.startswith(" ") and body.endswith(":"):
    return prefix + _ansi(body, "1;34")
  if body.startswith("  ") and not body.startswith("    ") and ":" in body:
    name, separator, remainder = body[2:].partition(":")
    return prefix + "  " + _ansi(name, "33") + separator + remainder
  if body.endswith("[present]"):
    return prefix + body.removesuffix("[present]") + _ansi("[present]", "1;97;44")
  if body.endswith("[missing]"):
    return prefix + body.removesuffix("[missing]") + _ansi("[missing]", "1;30;43")
  return line


def _ansi(value: str, code: str) -> str:
  """Wrap text in one standard terminal color sequence."""
  return f"\x1b[{code}m{value}\x1b[0m"


def _names(names: list[str]) -> str:
  """Join configured names without hiding empty collections."""
  return ", ".join(names) or "none"


def _term_lines(name: str, term: dict[str, Any] | None) -> list[str]:
  """Render a named term and its compact settings without filtering disabled entries."""
  if term is None:
    return [f"  {name}: disabled (None)"]
  identity = term.get("func", term.get("type"))
  if identity is None:
    return _wrapped(f"  {name}: {_compact(term)}")
  settings = {key: value for key, value in term.items() if key not in {"type", "func"}}
  lines = [f"  {name}: {identity}"]
  if settings:
    lines.extend(_wrapped(f"    {_compact(settings)}"))
  return lines


def _compact(value: Any) -> str:
  """Render serialized containers on a single logical line."""
  if isinstance(value, dict):
    return ", ".join(f"{key}={_compact(item)}" for key, item in value.items()) or "{}"
  return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _wrapped(line: str) -> list[str]:
  """Wrap long settings without changing configured ordering."""
  return textwrap.wrap(
    line,
    width=100,
    subsequent_indent="    ",
    break_long_words=False,
    break_on_hyphens=False,
  ) or [line]


def _controller_lines(controllers: dict[str, Any]) -> list[str]:
  """Render all controller actions and their present or absent source files."""
  lines = ["\nControllers:"]
  for name, controller in controllers.items():
    lines.append(
      f"  {name}: controller={controller['controller_name']}, "
      f"robot={controller['robot_name']}, MainRobot={controller['main_robot']}"
    )
    for group in ("files", "installed_controller_configs", "user_controller_configs"):
      records = controller[group]
      if not records:
        lines.append(f"    {group}: none found")
      for record in records:
        status = "present" if record["present"] else "missing"
        lines.append(f"    {record['role']}: {record['path']} [{status}]")
    lines.extend(
      f"    config: {error}" for error in controller.get("config_errors", [])
    )
  return lines
