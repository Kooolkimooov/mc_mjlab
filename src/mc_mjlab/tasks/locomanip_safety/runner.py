"""Refuse full safety training until authority is measured on its frozen profile."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from mc_mjlab.bridge.config import get_controller_name
from mc_mjlab.rl.runner import McRtcResidualOnPolicyRunner
from mc_mjlab.tasks.locomanip_safety.locomanip_safety_env_cfg import (
  FAILING_MASS_KG,
  PASSING_MASS_KG,
  SAFETY_CONFIG,
)
from mc_mjlab.utils.controller_install import (
  controller_config_paths,
  user_controller_config_paths,
)

AUTHORITY_REPORT_ENV = "MC_MJLAB_SAFETY_AUTHORITY_REPORT"


def controller_inputs_sha256(config: Path) -> str:
  """Bind a probe to installed and user controller settings behind the selector."""
  name = get_controller_name(config)
  paths = [
    Path.home() / ".config/mc_rtc/mc_rtc.yaml",
    *controller_config_paths(name),
    *user_controller_config_paths(name),
  ]
  digest = hashlib.sha256()
  for path in paths:
    digest.update(str(path).encode())
    digest.update(b"\0")
    digest.update(path.read_bytes() if path.is_file() else b"<absent>")
    digest.update(b"\0")
  return digest.hexdigest()


def validate_authority_report(path: Path, config: Path = SAFETY_CONFIG) -> bytes:
  """Require near-failure input delivery and physical response on this exact profile."""
  raw = path.read_bytes()
  report = json.loads(raw)
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
  if type(report.get("encoder_bias")) is not bool:
    raise RuntimeError("authority report must declare encoder_bias as true or false")
  passing = PASSING_MASS_KG
  failing = FAILING_MASS_KG
  if (report.get("passing_mass_kg"), report.get("failing_mass_kg")) != (
    passing,
    failing,
  ):
    raise RuntimeError(
      f"authority report must measure the {passing:g}/{failing:g} kg transition"
    )
  if report.get("controller_inputs_sha256") != controller_inputs_sha256(config):
    raise RuntimeError("authority report used different installed controller inputs")
  return raw


def authority_report_path() -> Path:
  """Find the report before allocating training controllers."""
  report = os.environ.get(AUTHORITY_REPORT_ENV)
  if not report:
    raise RuntimeError(f"set {AUTHORITY_REPORT_ENV} to the measured authority report")
  return Path(report)


class LocomanipSafetyRunner(McRtcResidualOnPolicyRunner):
  """Gate a new training run on verified reactive force authority."""

  def setup_task_hooks(self, env: object, train_cfg: dict, log_dir: str | None) -> None:
    """Validate authority before the runner can begin its first update."""
    del env, train_cfg
    if log_dir is None:
      return
    self._authority_report = validate_authority_report(authority_report_path())

  def materialize_task_inputs(self, log_dir: Path) -> None:
    """Copy the exact admitting report into the training run."""
    log_dir.joinpath("authority_gate.json").write_bytes(self._authority_report)

  def checkpoint_task_state(self) -> dict:
    """Embed the admitting report and its byte hash in each checkpoint."""
    raw = self._authority_report
    return {
      "authority_gate_report": {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "report": json.loads(raw),
      }
    }

  def restore_task_state(self, infos: dict) -> None:
    """Keep a resumed run bound to its original authority measurement."""
    saved = infos.get("authority_gate_report")
    digest = hashlib.sha256(self._authority_report).hexdigest()
    if saved is None or saved.get("sha256") != digest:
      raise RuntimeError("checkpoint authority report differs from this run")
