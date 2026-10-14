"""Preflight the authority gate before mjlab allocates training environments."""

from __future__ import annotations

import sys

import mjlab  # noqa: F401
from mjlab.scripts.train import main as train_main

from mc_mjlab.tasks.locomanip_safety import TASK_ID
from mc_mjlab.tasks.locomanip_safety.runner import (
  authority_report_path,
  validate_authority_report,
)


def main() -> None:
  """Validate the report and delegate the unchanged CLI to mjlab training."""
  if len(sys.argv) < 2 or sys.argv[1] != TASK_ID:
    raise ValueError(f"first argument must be {TASK_ID}")
  validate_authority_report(authority_report_path())
  train_main()


if __name__ == "__main__":
  main()
