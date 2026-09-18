"""What the sourced workspace actually installed for a named mc_rtc controller."""

from __future__ import annotations

import os
from pathlib import Path


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
    for suffix in ("conf", "yaml", "yml"):
      paths.update(library_dir.glob(f"*/etc/{controller_name}.{suffix}"))
      # Per-robot overrides, merged at the root by BaselineWalkingController.
      paths.update(library_dir.glob(f"*/{controller_name}/*.{suffix}"))
  return sorted(path.resolve() for path in paths if path.is_file())


def controller_is_installed(controller_name: str) -> bool:
  """Whether the workspace ships this controller, as module or as configuration."""
  # A variant like LogisticController_ismpc ships config but no module of its own.
  if controller_config_paths(controller_name):
    return True
  return any(
    (library_dir / "mc_controller" / f"{controller_name}.so").is_file()
    for library_dir in controller_library_dirs()
  )
