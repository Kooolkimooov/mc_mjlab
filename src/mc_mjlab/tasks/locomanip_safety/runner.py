"""Refuse full safety training until authority is measured on its frozen profile."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from mc_mjlab.rl.runner import McRtcResidualOnPolicyRunner
from mc_mjlab.tasks.locomanip_safety.locomanip_safety_env_cfg import SAFETY_CONFIG

AUTHORITY_REPORT_ENV = "MC_MJLAB_SAFETY_AUTHORITY_REPORT"


def validate_authority_report(path: Path, config: Path = SAFETY_CONFIG) -> None:
  """Require near-failure input delivery and physical response on this exact profile."""
  report = json.loads(path.read_text())
  digest = hashlib.sha256(config.read_bytes()).hexdigest()
  if report.get("profile_sha256") != digest:
    raise RuntimeError(
      "authority report was measured with a different controller profile"
    )
  if not report.get("all_channels_delivered") or not report.get(
    "measurable_near_failure"
  ):
    raise RuntimeError(
      "authority report does not demonstrate useful near-failure force input"
    )
  if report.get("encoder_bias") is not False:
    raise RuntimeError("authority report must declare its encoder-bias setting")


class LocomanipSafetyRunner(McRtcResidualOnPolicyRunner):
  """Gate a new training run on verified reactive force authority."""

  def setup_task_hooks(self, env: object, train_cfg: dict, log_dir: str | None) -> None:
    """Validate authority before the runner can begin its first update."""
    del env, train_cfg
    if log_dir is None:
      return
    report = os.environ.get(AUTHORITY_REPORT_ENV)
    if not report:
      raise RuntimeError(f"set {AUTHORITY_REPORT_ENV} to the measured authority report")
    validate_authority_report(Path(report))
