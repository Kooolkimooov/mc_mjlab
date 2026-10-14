"""Isolated controller variants from the native controller's effective configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import mc_control
import yaml

TUNINGS = {
  "current": (0.03, [0.0, 0.0, 0.98]),
  "full-005": (0.05, [1.0, 1.0, 1.0]),
  "full-006": (0.06, [1.0, 1.0, 1.0]),
  "full-007": (0.07, [1.0, 1.0, 1.0]),
}
INITIAL_GRID = (500.0, 525.0, 550.0, 575.0, 600.0, 625.0, 650.0)
LIGHT_GRID = (10.0, 100.0)


def effective_configuration(project: Path) -> dict[str, Any]:
  """Read the native merged config after robot and user overrides have applied."""
  controller = mc_control.MCGlobalController(  # ty: ignore[unresolved-attribute]
    str(project.resolve())
  )
  result = yaml.safe_load(controller.controller().config().dump())
  del controller
  return result


def write_variant(
  project: Path,
  effective: dict[str, Any],
  tuning: str,
  destination: Path,
  native_logs: Path | None = None,
) -> Path:
  """Freeze the effective controller and reload its FSM with one changed adaptation."""
  payload = deepcopy(effective)
  for key in (
    "OverwriteConfigKeys",
    "OverwriteConfigList",
    "StatesLibraries",
    "StatesFiles",
  ):
    payload.pop(key, None)
  source_fsm = payload.get("DemoFSM")
  if source_fsm is None:
    source_fsm = {key: payload[key] for key in ("states", "transitions", "init")}
  fsm = deepcopy(source_fsm)
  alpha, projection = TUNINGS[tuning]
  adaptation = fsm["states"]["LMC::PushCart_"]["configs"].setdefault(
    "HandWrenchAdaptation", {}
  )
  adaptation.update(
    enable=True, alpha=alpha, forceProjection=projection, momentProjection=[0.0] * 3
  )
  names = {name: f"MjlabSafety::{name}" for name in fsm["states"]}
  fsm = _rename_states(fsm, names)
  payload["CompletionState"] = names[payload["CompletionState"]]
  payload["states"] = deepcopy(fsm["states"])
  payload["DemoFSM"] = fsm
  config = {}
  user = Path.home() / ".config/mc_rtc/mc_rtc.yaml"
  if user.exists():
    config.update(yaml.safe_load(user.read_text()) or {})
  config.update(yaml.safe_load(project.read_text()) or {})
  config["OverwriteConfigKeys"] = ["MjlabSafetyFrozen"]
  config["OverwriteConfigList"] = {"MjlabSafetyFrozen": payload}
  config["Log"] = native_logs is not None
  if native_logs is not None:
    native_logs.mkdir(parents=True, exist_ok=True)
    config.update(
      LogDirectory=str(native_logs.resolve()), LogTemplate=f"safety-{tuning}"
    )
  destination.parent.mkdir(parents=True, exist_ok=True)
  destination.write_text(yaml.safe_dump(config, sort_keys=False))
  return destination


def rank_tunings(
  reports: dict[str, list[dict[str, Any]]], repetitions: int
) -> list[dict[str, Any]]:
  """Rank consistent completion, next-point failure, then placement error."""
  if repetitions < 1:
    raise ValueError("repetitions must be positive")
  ranked = []
  for name, rows in reports.items():
    masses = sorted({row["mass_kg"] for row in rows})
    grouped = {mass: [row for row in rows if row["mass_kg"] == mass] for mass in masses}
    safe = {
      mass: len(group) == repetitions
      and all(
        row["complete"] and row["observed"] and not row["worker_failure"]
        for row in group
      )
      for mass, group in grouped.items()
    }
    if not all(safe.get(mass, False) for mass in LIGHT_GRID):
      continue
    passed = [mass for mass in masses if mass >= 500 and safe[mass]]
    # A safe envelope is contiguous; a higher isolated completion cannot hide a hole.
    last = None
    first = None
    for mass in (mass for mass in masses if mass >= 500):
      if not safe[mass]:
        first = mass
        break
      last = mass
    if last is None or not passed:
      continue
    next_rows = grouped[first] if first is not None else []
    ranked.append(
      {
        "tuning": name,
        "last_passing_mass_kg": last,
        "first_failing_mass_kg": first,
        "threshold_midpoint_kg": (last + first) / 2 if first is not None else None,
        "next_failure_rate": (
          sum(row["combined_failure"] for row in next_rows) / len(next_rows)
          if next_rows
          else 0.0
        ),
        "placement_error_m": sum(
          row["terminal_position_error_m"] for row in grouped[last]
        )
        / len(grouped[last]),
      }
    )
  return sorted(
    ranked,
    key=lambda row: (
      -row["last_passing_mass_kg"],
      row["next_failure_rate"],
      row["placement_error_m"],
    ),
  )


def _rename_states(value: Any, names: dict[str, str]) -> Any:
  """Give reloaded states fresh names because the native factory forbids replacement."""
  if isinstance(value, dict):
    return {
      names.get(key, key): _rename_states(item, names) for key, item in value.items()
    }
  if isinstance(value, list):
    return [_rename_states(item, names) for item in value]
  return names.get(value, value) if isinstance(value, str) else value
