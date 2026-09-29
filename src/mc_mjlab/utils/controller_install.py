"""What the sourced workspace actually installed for a named mc_rtc controller."""

from __future__ import annotations

import os
from pathlib import Path

#: Every extension mc_rtc reads a controller configuration from.
CONFIG_SUFFIXES = ("conf", "yaml", "yml")


def controller_library_dirs() -> list[Path]:
  """Every ``LD_LIBRARY_PATH`` entry a controller could be installed under."""
  return [
    Path(value).expanduser()
    for value in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    if value
  ]


def controller_config_paths(controller_name: str) -> list[Path]:
  """The installed mc_rtc files that configure this controller."""
  paths: set[Path] = set()
  for library_dir in controller_library_dirs():
    for suffix in CONFIG_SUFFIXES:
      paths.update(library_dir.glob(f"*/etc/{controller_name}.{suffix}"))
      # Per-robot overrides, merged at the root by BaselineWalkingController.
      paths.update(library_dir.glob(f"*/{controller_name}/*.{suffix}"))
  return sorted(path.resolve() for path in paths if path.is_file())


def user_controller_config_paths(controller_name: str) -> list[Path]:
  """The ``~/.config/mc_rtc`` overrides mc_rtc merges over the installed config."""
  # Candidates, not matches: an absent one is a file that would take effect the
  # moment it appeared, which provenance has to notice. docs/coupling.md#user_controller_config_paths
  root = Path.home() / ".config" / "mc_rtc" / "controllers"
  paths = {root / f"{controller_name}.{suffix}" for suffix in CONFIG_SUFFIXES}
  for suffix in CONFIG_SUFFIXES:
    paths.update((root / controller_name).glob(f"*.{suffix}"))
  return sorted(paths)


def controller_is_installed(controller_name: str) -> bool:
  """Whether the workspace ships this controller, as module or as configuration."""
  # A variant like LogisticController_ismpc ships config but no module of its own.
  if controller_config_paths(controller_name):
    return True
  return any(
    (library_dir / "mc_controller" / f"{controller_name}.so").is_file()
    for library_dir in controller_library_dirs()
  )
