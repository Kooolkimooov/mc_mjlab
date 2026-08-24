"""Pure tensor helpers for residual feasibility enforcement."""

from __future__ import annotations

import torch


def project_residual(
  nominal: torch.Tensor,
  residual: torch.Tensor,
  lower: torch.Tensor,
  upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Project a residual around a separately clamped nominal target."""
  safe_nominal = nominal.clamp(lower, upper)
  target = (nominal + residual).clamp(lower, upper)
  executed = target - safe_nominal
  projected = ~torch.isclose(executed, residual, atol=1e-7, rtol=1e-5)
  return target, executed, projected
