"""An rsl_rl actor whose mean head starts at zero, so the residual starts at zero."""

from __future__ import annotations

import torch
from rsl_rl.models.mlp_model import MLPModel


class ZeroInitMLPModel(MLPModel):
  """MLPModel with the final layer's mean rows zeroed after construction."""

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    zero_mean_head(self)


def zero_mean_head(model: MLPModel) -> None:
  """Zero the mean rows of a model's output layer, leaving any std rows alone."""
  if model.distribution is None:
    return
  layers = [m for m in model.mlp if isinstance(m, torch.nn.Linear)]
  # Heteroscedastic heads stack mean over std in the same layer; scalar-std heads
  # carry std in a separate Parameter and own every row. Slicing suits both.
  mean_rows = model.distribution.output_dim
  with torch.no_grad():
    layers[-1].weight[:mean_rows].zero_()
    layers[-1].bias[:mean_rows].zero_()


def mean_head_magnitude(model: MLPModel, obs) -> float:
  """Peak absolute mean action over a batch, for asserting the head really is zero."""
  with torch.no_grad():
    return float(model(obs, stochastic_output=False).abs().max())
