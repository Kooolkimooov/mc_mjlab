"""Terminal printout of one environment's residual, for interactive play."""

from __future__ import annotations

import os

import torch


class ResidualPrinter:
  """Throttled one-line printout of environment zero's executed residual."""

  def __init__(
    self,
    every: int,
    names: list[str],
    limit: list[float] | None,
    unit: str,
  ) -> None:
    override = os.environ.get("MC_MJLAB_PRINT_RESIDUAL")
    self.every = max(0, int(override) if override is not None else every)
    self._names = names
    self._limit = limit
    self._unit = unit
    self._countdown = 0
    self._header_pending = True
    self._pending = False

  def request(self) -> None:
    """Mark the current policy step as one whose residual should be printed."""
    if self.every:
      self._pending = True

  def emit(self, executed_physical: torch.Tensor) -> None:
    """Print the requested line once the residual for that step is known."""
    if not self._pending:
      return
    self._pending = False
    if self._countdown:
      self._countdown -= 1
      return
    self._countdown = self.every - 1

    values = executed_physical[0].detach().cpu().tolist()
    if self._header_pending:
      # Lazily, on the first line: a header printed at construction would be
      # buried under mc_rtc's own startup logging long before the first frame.
      unit = f" [{self._unit}]" if self._unit else ""
      print(f"[residual] env 0, every {self.every} policy step(s){unit}; * = clipped")
      print("[residual] " + " ".join(f"{n:>6s} " for n in self._names) + "   |r|")
      self._header_pending = False

    limit = self._limit
    cells = [
      f"{v:+.3f}" + ("*" if limit is not None and abs(v) >= 0.999 * limit[j] else " ")
      for j, v in enumerate(values)
    ]
    norm = sum(v * v for v in values) ** 0.5
    # Flushed: Python block-buffers into a pipe while spdlog writes to fd 1, so
    # unflushed the two interleave wrongly. docs/coupling.md#console-output
    print("[residual] " + " ".join(cells) + f" {norm:6.3f}", flush=True)
