"""Installed controller binaries retain byte and source identities."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from mc_mjlab.rl import controller_artifacts
from mc_mjlab.rl.controller_provenance import provenance_matches


def test_variant_records_module_library_and_source_commit(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  workspace = tmp_path / "workspace"
  library = workspace / "install/lib"
  module = library / "mc_controller/Example_controller.so"
  companion = library / "libExample.so"
  source = workspace / "src/example"
  build = workspace / "build/example"
  for directory in (module.parent, source, build):
    directory.mkdir(parents=True)
  module.write_bytes(b"module-v1")
  companion.write_bytes(b"companion-v1")
  (source / "README").write_text("source\n")
  subprocess.run(["git", "init", "-q", str(source)], check=True)
  subprocess.run(["git", "-C", str(source), "add", "README"], check=True)
  subprocess.run(
    [
      "git",
      "-C",
      str(source),
      "-c",
      "user.name=Test",
      "-c",
      "user.email=test@example.com",
      "commit",
      "-qm",
      "initial",
    ],
    check=True,
  )
  commit = subprocess.check_output(
    ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
  ).strip()
  (build / "install_manifest.txt").write_text(f"{module}\n{companion}\n")
  (build / "CMakeCache.txt").write_text(f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\n")
  monkeypatch.setattr(
    controller_artifacts, "controller_library_dirs", lambda: [library]
  )

  records = controller_artifacts.collect_controller_artifacts("Example_variant")
  assert [record["source"] for record in records] == sorted(
    [str(module), str(companion)]
  )
  assert {record["sha256"] for record in records} == {
    hashlib.sha256(b"module-v1").hexdigest(),
    hashlib.sha256(b"companion-v1").hexdigest(),
  }
  assert {record["git_commit"] for record in records} == {commit}
  assert {record["git_dirty"] for record in records} == {False}

  original = {"controller_name": "Example_variant", "files": [], "artifacts": records}
  module.write_bytes(b"module-v2")
  changed = {
    **original,
    "artifacts": controller_artifacts.collect_controller_artifacts("Example_variant"),
  }
  assert not provenance_matches(original, changed)
  assert provenance_matches(
    {"controller_name": "Example_variant", "files": []}, changed
  )
  same_binary_different_checkout = {
    **original,
    "artifacts": [{**record, "git_commit": "another"} for record in records],
  }
  assert provenance_matches(original, same_binary_different_checkout)


def test_missing_module_has_no_invented_commit(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  monkeypatch.setattr(
    controller_artifacts, "controller_library_dirs", lambda: [tmp_path]
  )
  assert controller_artifacts.collect_controller_artifacts("Missing") == []
