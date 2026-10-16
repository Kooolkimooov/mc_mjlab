"""Convert configuration values to ordered, address-free JSON data."""

from __future__ import annotations

import dataclasses
import enum
import functools
import inspect
import math
from collections.abc import Mapping, Sequence
from pathlib import PurePath
from typing import Any

import numpy as np
import torch


def qualified_name(value: Any) -> str:
  """Name a callable or type without its runtime representation."""
  if isinstance(value, functools.partial):
    return qualified_name(value.func)
  module = getattr(value, "__module__", type(value).__module__)
  name = getattr(value, "__qualname__", type(value).__qualname__)
  return f"{module}.{name}"


def serialize(value: Any) -> Any:
  """Preserve configured order and callable defaults in JSON-compatible values."""
  return _serialize(value, set())


def _serialize(value: Any, ancestors: set[int]) -> Any:
  """Serialize one value while bounding recursive configuration references."""
  if isinstance(value, enum.Enum):
    return {"enum": qualified_name(type(value)), "name": value.name}
  if value is None or isinstance(value, (str, bool, int)):
    return value
  if isinstance(value, float):
    return value if math.isfinite(value) else str(value)
  if isinstance(value, PurePath):
    return str(value)
  if isinstance(value, (torch.dtype, torch.device)):
    return str(value)

  identity = id(value)
  if identity in ancestors:
    return {"recursive_reference": qualified_name(type(value))}
  ancestors = ancestors | {identity}

  if isinstance(value, (np.ndarray, np.generic)):
    return _serialize(value.tolist(), ancestors)
  if isinstance(value, torch.Tensor):
    return _serialize(value.detach().cpu().tolist(), ancestors)
  if isinstance(value, slice):
    return {"slice": _serialize([value.start, value.stop, value.step], ancestors)}
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    return {
      field.name: _serialize(getattr(value, field.name), ancestors)
      for field in dataclasses.fields(value)
    }
  if isinstance(value, Mapping):
    return {
      _mapping_key(key): _serialize(item, ancestors) for key, item in value.items()
    }
  if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
    return [_serialize(item, ancestors) for item in value]
  if isinstance(value, (set, frozenset)):
    return [_serialize(item, ancestors) for item in sorted(value, key=_mapping_key)]
  if callable(value):
    return _callable_record(value, ancestors)
  if hasattr(value, "__dict__"):
    return {
      "type": qualified_name(type(value)),
      "attributes": _serialize(vars(value), ancestors),
    }
  return {"type": qualified_name(type(value))}


def _mapping_key(value: Any) -> str:
  """Render configuration mapping keys without object addresses."""
  if isinstance(value, enum.Enum):
    return f"{qualified_name(type(value))}.{value.name}"
  if isinstance(value, (str, bool, int, float, PurePath)) or value is None:
    return str(value)
  if isinstance(value, tuple):
    return "(" + ", ".join(_mapping_key(item) for item in value) + ")"
  return qualified_name(type(value))


def _callable_record(value: Any, ancestors: set[int]) -> dict[str, Any]:
  """Separate declared parameter defaults from partial argument bindings."""
  record: dict[str, Any] = {"callable": qualified_name(value)}
  target = value
  if isinstance(value, functools.partial):
    record["partial_args"] = _serialize(value.args, ancestors)
    record["partial_keywords"] = _serialize(value.keywords, ancestors)
    target = value.func

  try:
    signature = inspect.signature(target)
  except (TypeError, ValueError):
    return record

  record["parameter_defaults"] = {
    name: _serialize(parameter.default, ancestors)
    for name, parameter in signature.parameters.items()
    if parameter.default is not inspect.Parameter.empty
  }
  return record
