"""Agreement of identical zero-policy conditions across study runs and arms."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

OUTCOMES = (
  "observed",
  "worker_failure",
  "warmup_failed",
  "complete",
  "combined_failure",
  "timeout",
)


def load_sources(runs: list[Path]) -> dict[str, dict[str, dict]]:
  """Each run's arms as independent sources, keyed by condition identifier."""
  sources: dict[str, dict[str, dict]] = defaultdict(dict)
  for run in runs:
    for row in json.loads((run / "episodes.json").read_text()):
      sources[f"{run.name}/{row['arm']}"][row["condition_id"]] = row
  return dict(sources)


def compare_sources(sources: dict[str, dict[str, dict]]) -> dict[str, Any]:
  """Outcome disagreements, continuous spreads and per-source mean shifts."""
  conditions = sorted(set().union(*(rows.keys() for rows in sources.values())))
  incomplete = [c for c in conditions if any(c not in r for r in sources.values())]
  disagreements = []
  displacement, duration = [], []
  peaks: dict[str, list[float]] = defaultdict(list)
  for condition in conditions:
    rows = [r[condition] for r in sources.values() if condition in r]
    for key in (*OUTCOMES, "terminations"):
      labels = {json.dumps(row.get(key), sort_keys=True) for row in rows}
      if len(labels) > 1:
        disagreements.append({"condition_id": condition, "field": key})
    displacement.append(_spread([row["robot_displacement_m"] for row in rows]))
    duration.append(_spread([row["time_s"] for row in rows]))
    if not all("pd_demand" in row for row in rows):
      continue
    for group in rows[0]["pd_demand"].keys() - {"rate_trace"}:
      values = [row["pd_demand"][group]["peak_abs_nm"] for row in rows]
      peaks[group].append(_spread(values) / max(float(np.mean(values)), 1e-9))

  return {
    "sources": sorted(sources),
    "conditions": len(conditions),
    "incomplete_conditions": incomplete,
    "outcome_disagreements": disagreements,
    "max_displacement_spread_m": max(displacement, default=None),
    "max_time_spread_s": max(duration, default=None),
    "max_relative_pd_peak_spread": {g: max(v) for g, v in peaks.items()},
    "mean_displacement_m": _source_means(sources, conditions, "robot_displacement_m"),
    "mean_time_s": _source_means(sources, conditions, "time_s"),
  }


def run_timing(run: Path) -> dict[str, Any]:
  """Summed rollout wall time and environment steps over a run's batch files."""
  wall, steps, telemetry = 0.0, 0, set()
  for path in sorted(run.glob("*-batch-*-attempt-*.json")):
    batch = json.loads(path.read_text())
    if "rollout_wall_s" not in batch:
      return {"run": run.name, "rollout_wall_s": None}
    wall += batch["rollout_wall_s"]
    steps += batch["env_steps"]
    telemetry.add(batch["pd_telemetry"])
  return {
    "run": run.name,
    "pd_telemetry": sorted(telemetry),
    "rollout_wall_s": wall,
    "env_steps": steps,
    "ms_per_env_step": 1000 * wall / steps if steps else None,
  }


def _spread(values: list[float]) -> float:
  return float(max(values) - min(values))


def _source_means(
  sources: dict[str, dict[str, dict]], conditions: list[str], key: str
) -> dict[str, float]:
  shared = [c for c in conditions if all(c in rows for rows in sources.values())]
  return {
    name: float(np.mean([rows[c][key] for c in shared])) if shared else float("nan")
    for name, rows in sources.items()
  }
