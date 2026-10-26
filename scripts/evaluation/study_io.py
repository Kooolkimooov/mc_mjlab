"""Strict JSON records for study outputs, writing non-finite floats as null."""

from __future__ import annotations

import json
import math
from pathlib import Path


def write_json(path: Path, value: object) -> None:
  """Write indented JSON that a strict parser accepts."""
  path.write_text(json.dumps(finite_json(value), indent=2, allow_nan=False) + "\n")


def finite_json(value: object) -> object:
  """Replace NaN and infinities recursively; a missing value is not a number."""
  if isinstance(value, float) and not math.isfinite(value):
    return None
  if isinstance(value, dict):
    return {key: finite_json(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return [finite_json(item) for item in value]
  return value
