"""Expose motion intervention alongside the existing physical task outcomes."""

from __future__ import annotations

from dataclasses import replace

from mc_mjlab.tasks.locomanip.evaluation import LOCOMANIP_EVALUATION

LOCOMANIP_MOTION_EVALUATION = replace(
  LOCOMANIP_EVALUATION,
  metrics=(*LOCOMANIP_EVALUATION.metrics, "motion_scale", "motion_holding"),
)
