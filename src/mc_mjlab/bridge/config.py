"""Reading the few values this side has to agree with out of the mc_rtc config."""

from __future__ import annotations

from pathlib import Path


def read_config_key(path: Path, key: str) -> str:
  """The value of top-level ``key``, comments stripped."""
  for raw in path.read_text().splitlines():
    if raw[:1].isspace():  # nested: not the top-level key we were asked for
      continue
    line = raw.split("#", 1)[0].rstrip()
    if line.startswith(f"{key}:"):
      return line.split(":", 1)[1].strip()
  raise ValueError(f"No {key} key found in {path}")


def get_main_robot_name(path: Path) -> str:
  """The config's ``MainRobot``, which selects the matching mjlab entity."""
  return read_config_key(path, "MainRobot")


def get_controller_name(path: Path) -> str:
  """The config's ``Enabled`` controller -- the base policy the residual rides."""
  value = read_config_key(path, "Enabled")
  if value.startswith("["):
    first = value.strip("[]").split(",")[0].strip().strip("\"'")
    if not first:
      raise ValueError(f"Empty Enabled list in {path}")
    return first
  if not value:
    raise ValueError(
      f"Enabled in {path} has no inline value. A block list (`- name` on the "
      "following lines) is not supported; write `Enabled: [name, ...]`."
    )
  return value
