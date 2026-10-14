"""Causal Hold-gated force probes with per-episode trigger state."""

from __future__ import annotations

from typing import Any

import torch

from evaluation.locomanip_authority import conditions

SCHEDULES = ("constant", "pulse", "reactive")


def reactive_conditions(repetitions: int = 3) -> list[dict[str, Any]]:
  """Repeat every signed channel excitation and zero control for every schedule."""
  return [
    {**row, "schedule": schedule, "repeat": repeat}
    for repeat in range(repetitions)
    for schedule in SCHEDULES
    for row in conditions("feedback", [0.1, 0.5, 1.0], "all")
  ]


class ForceSchedule:
  """Turn pre-action lag measurements into bounded force requests."""

  def __init__(self, schedules: list[str], device: str) -> None:
    if any(name not in SCHEDULES for name in schedules):
      raise ValueError("unknown force schedule")
    self.kind = torch.tensor([SCHEDULES.index(s) for s in schedules], device=device)
    self.trigger_step = torch.full(
      (len(schedules),), -1, dtype=torch.int64, device=device
    )
    self.trigger_time = torch.full(
      (len(schedules),), -1.0, dtype=torch.float64, device=device
    )
    self.duration = torch.zeros(len(schedules), dtype=torch.float64, device=device)
    self.active = torch.zeros(len(schedules), dtype=torch.bool, device=device)

  def update(
    self, time: torch.Tensor, lag: torch.Tensor, holding: torch.Tensor, dt: float
  ) -> torch.Tensor:
    """Apply one-second pulses or 0.05/0.03 m hysteresis using current inputs only."""
    step = torch.round(time.to(torch.float64) / dt).to(torch.int64)
    holding = holding.bool() & torch.isfinite(lag)
    crossed = holding & (lag > 0.05)
    first = (self.trigger_step < 0) & (crossed | (holding & (self.kind == 0)))
    self.trigger_step[first] = step[first]
    self.trigger_time[first] = step[first].to(torch.float64) * dt
    pulse_steps = round(1.0 / dt)
    pulse = (self.trigger_step >= 0) & (step - self.trigger_step < pulse_steps)
    reactive = crossed | (self.active & (lag >= 0.03))
    self.active = holding & torch.where(
      self.kind == 0, True, torch.where(self.kind == 1, pulse, reactive)
    )
    self.duration += self.active.to(torch.float64) * dt
    return self.active.clone()

  def reset(self, env_ids: torch.Tensor | slice) -> None:
    """Clear only the episodes being reset."""
    self.trigger_step[env_ids] = -1
    self.trigger_time[env_ids] = -1.0
    self.duration[env_ids] = 0.0
    self.active[env_ids] = False
