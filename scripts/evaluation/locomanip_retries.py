"""Retry only unmatched or unusable paired safety conditions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from evaluation.locomanip_promotion import usable_outcome


def retry_cohort(
  rows: list[dict[str, Any]],
  run_once: Callable[[list[dict[str, Any]], int], list[dict[str, Any]]],
  max_retries: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
  """Rerun both arms of each incomplete pair with the same condition inputs."""
  if max_retries < 0:
    raise ValueError("max_retries must be nonnegative")
  if not rows:
    raise ValueError("cohort must be nonempty")
  expected = {(row["condition_id"], row["arm"]): row for row in rows}
  if len(expected) != len(rows):
    raise ValueError("duplicate expected condition arm")
  identifiers = {row["condition_id"] for row in rows}
  if any(
    (identifier, arm) not in expected
    for identifier in identifiers
    for arm in ("baseline", "policy")
  ):
    raise ValueError("each condition must have baseline and policy arms")
  latest: dict[tuple[str, str], dict[str, Any]] = {}
  history = []
  pending = rows
  for attempt in range(max_retries + 1):
    for row in pending:
      latest.pop((row["condition_id"], row["arm"]), None)
    received = run_once(pending, attempt)
    seen = set()
    for row in received:
      key = (row["condition_id"], row["arm"])
      if key not in expected:
        raise ValueError(f"unexpected condition arm: {key}")
      if key in seen:
        raise ValueError(f"duplicate returned condition arm: {key}")
      seen.add(key)
      latest[key] = row
    unresolved = sorted(
      {
        identifier
        for identifier in identifiers
        if any(
          not usable_outcome(latest.get((identifier, arm)))
          for arm in ("baseline", "policy")
        )
      }
    )
    history.append(
      {"attempt": attempt, "requested_rows": len(pending), "unresolved": unresolved}
    )
    if not unresolved:
      break
    pending = [row for row in rows if row["condition_id"] in unresolved]
  return [latest[key] for key in expected if key in latest], history, unresolved
