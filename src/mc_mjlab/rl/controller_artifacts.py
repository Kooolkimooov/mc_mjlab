"""Identify installed controller binaries and their source revisions."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from mc_mjlab.utils.controller_install import controller_library_dirs


def controller_artifact_paths(controller_name: str) -> list[Path]:
  """Find the selected controller module and its companion library."""
  for name in _controller_names(controller_name):
    paths = set()
    for library_dir in controller_library_dirs():
      paths.update(
        path.resolve()
        for path in (
          library_dir / "mc_controller" / f"{name}.so",
          library_dir / "mc_controller" / f"{name}_controller.so",
        )
        if path.is_file()
      )
      paths.update(
        path.resolve() for path in library_dir.glob(f"lib{name}.so*") if path.is_file()
      )
    if paths:
      return sorted(paths)
  return []


def collect_controller_artifacts(controller_name: str) -> list[dict]:
  """Record exact binary bytes and the build's source Git revision if found."""
  return [_artifact_record(path) for path in controller_artifact_paths(controller_name)]


def _controller_names(name: str) -> list[str]:
  """Try an exact module before stripping configuration-variant suffixes."""
  names = [name]
  while "_" in name:
    name = name.rsplit("_", 1)[0]
    names.append(name)
  return names


def _artifact_record(path: Path) -> dict:
  """Hash a binary without copying it into every training checkpoint."""
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  source = _source_directory(path)
  revision = _git_revision(source) if source is not None else None
  dirty = _git_dirty(source) if revision is not None and source is not None else None
  return {
    "source": str(path),
    "sha256": digest.hexdigest(),
    "git_source": str(source) if source is not None else None,
    "git_commit": revision,
    "git_dirty": dirty,
  }


def _source_directory(binary: Path) -> Path | None:
  """Map an installed binary through its CMake install manifest."""
  library_dir = next(
    (parent for parent in binary.parents if parent.name == "lib"), None
  )
  if library_dir is None:
    return None
  workspace = library_dir.parent.parent
  build_roots = [workspace / "build", *workspace.glob("src/*/build")]
  for root in build_roots:
    for manifest in root.glob("*/install_manifest.txt"):
      if str(binary) not in manifest.read_text().splitlines():
        continue
      cache = manifest.with_name("CMakeCache.txt")
      if not cache.is_file():
        continue
      for line in cache.read_text().splitlines():
        if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL="):
          source = Path(line.split("=", 1)[1])
          return source if source.is_dir() else None
  return None


def _git_revision(source: Path) -> str | None:
  """Read the checkout commit without assuming the install matches it."""
  result = subprocess.run(
    ["git", "-C", str(source), "rev-parse", "HEAD"],
    capture_output=True,
    text=True,
    check=False,
  )
  return result.stdout.strip() if result.returncode == 0 else None


def _git_dirty(source: Path) -> bool | None:
  """Mark local source edits without mistaking a commit for the built bytes."""
  result = subprocess.run(
    ["git", "-C", str(source), "status", "--porcelain"],
    capture_output=True,
    text=True,
    check=False,
  )
  return bool(result.stdout.strip()) if result.returncode == 0 else None
