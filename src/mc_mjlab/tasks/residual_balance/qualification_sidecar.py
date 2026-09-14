"""A checkpoint's own qualification verdict, written beside it after a sweep."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

SUFFIX = ".qualification.json"
#: Set to 1 to downgrade an unqualified full resume from a refusal to a warning.
ALLOW_UNQUALIFIED_ENV = "MC_MJLAB_ALLOW_UNQUALIFIED_RESUME"


def sidecar_path(checkpoint: Path | str) -> Path:
  """The verdict path for one checkpoint."""
  path = Path(checkpoint)
  return path.with_name(path.name + SUFFIX)


def checkpoint_digest(checkpoint: Path | str) -> str:
  """Hash a checkpoint so a stale verdict beside a rewritten file is detectable."""
  digest = hashlib.sha256()
  with Path(checkpoint).open("rb") as stream:
    for block in iter(lambda: stream.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest()


def write(checkpoint: Path | str, promotion: dict[str, Any], config: dict) -> Path:
  """Record one checkpoint's verdict, passing or failing, beside the checkpoint."""
  path = sidecar_path(checkpoint)
  document = {
    "checkpoint": str(Path(checkpoint).resolve()),
    "checkpoint_sha256": checkpoint_digest(checkpoint),
    "qualified_unix": time.time(),
    "promotion": promotion,
    "config": config,
  }
  temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
  temporary.write_text(json.dumps(document, indent=2, allow_nan=True) + "\n")
  temporary.replace(path)
  return path


def read(checkpoint: Path | str) -> dict[str, Any] | None:
  """Read a checkpoint's verdict, returning ``None`` while it is absent."""
  try:
    return json.loads(sidecar_path(checkpoint).read_text())
  except (FileNotFoundError, json.JSONDecodeError):
    return None


def _complaint(checkpoint: Path | str, verdict: dict[str, Any] | None) -> str | None:
  """The reason this checkpoint is not a valid resume point, if it is not."""
  if verdict is None:
    return f"no qualification verdict beside {checkpoint}; sweep it first"
  if verdict.get("checkpoint_sha256") != checkpoint_digest(checkpoint):
    return f"the verdict beside {checkpoint} was written for different bytes"
  promotion = verdict.get("promotion") or {}
  if not promotion.get("eligible"):
    unmeasured = promotion.get("not_measured") or []
    reasons = promotion.get("reasons") or []
    detail = "; ".join(reasons) or "no reason recorded"
    prefix = f"{len(unmeasured)} criteria not measured; " if unmeasured else ""
    return f"{checkpoint} did not qualify: {prefix}{detail}"
  return None


def enforce(checkpoint: Path | str, full_resume: bool) -> None:
  """Refuse a full resume from a checkpoint no sweep has qualified."""
  complaint = _complaint(checkpoint, read(checkpoint))
  if complaint is None:
    return
  if not full_resume:
    # The qualifier itself loads actor-only; refusing here would deadlock it.
    print(f"[mc_mjlab] {complaint}")
    return
  if os.environ.get(ALLOW_UNQUALIFIED_ENV) == "1":
    print(f"[mc_mjlab] {complaint} -- continuing, {ALLOW_UNQUALIFIED_ENV}=1")
    return
  raise RuntimeError(
    f"{complaint}. Qualify it with scripts/qualify_checkpoints.py, or set "
    f"{ALLOW_UNQUALIFIED_ENV}=1 to resume anyway."
  )
