"""Payload mixtures preserve pseudo-inertia consistency and reset boundaries."""

from __future__ import annotations

import math
from types import SimpleNamespace as NS
from typing import Any, cast

import pytest
import torch
from mjlab.envs.mdp import dr
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mc_mjlab.tasks.locomanip.mdp import payload_mixture


def test_mixture_keeps_model_fields_and_recompute_contract() -> None:
  assert (
    payload_mixture.transition_payload.model_fields  # ty: ignore[unresolved-attribute]
    == dr.pseudo_inertia.model_fields  # ty: ignore[unresolved-attribute]
  )
  assert (
    payload_mixture.transition_payload.recompute  # ty: ignore[unresolved-attribute]
    == dr.pseudo_inertia.recompute  # ty: ignore[unresolved-attribute]
  )
  assert payload_mixture.mass_bands(537.5) == (
    (462.5, 612.5),
    (10, 462.5),
    (612.5, 1000),
  )
  assert payload_mixture.mass_bands(950)[0] == (875, 1000)
  with pytest.raises(ValueError):
    payload_mixture.mass_bands(float("nan"))
  assert payload_mixture.mass_bands(20)[1] == (10, 10)


def test_reset_partitions_ids_and_calls_density_randomization_once_per_band(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls = []

  def pseudo(env: Any, ids: torch.Tensor, **kwargs: Any) -> None:
    calls.append((ids.clone(), kwargs))

  monkeypatch.setattr(dr, "pseudo_inertia", pseudo)
  monkeypatch.setattr(
    torch, "rand", lambda *a, **kw: torch.tensor([0.1, 0.59, 0.6, 0.79, 0.8, 0.99])
  )
  env = cast(Any, NS(num_envs=20, device="cpu"))
  ids = torch.tensor([1, 4, 7, 8, 12, 17])
  asset = SceneEntityCfg("cart", body_names=("Body",))
  payload_mixture.transition_payload(env, ids, 537.5, 10, asset)
  assert [selected.tolist() for selected, _ in calls] == [[1, 4], [7, 8], [12, 17]]
  for (_, params), band in zip(calls, payload_mixture.mass_bands(537.5), strict=True):
    assert params["asset_cfg"] is asset
    assert [
      10 * math.exp(2 * value) for value in params["alpha_range"]
    ] == pytest.approx(band)
    assert set(params) == {"alpha_range", "asset_cfg"}
  calls.clear()
  payload_mixture.transition_payload(
    env, torch.tensor([], dtype=torch.long), 537.5, 10, asset
  )
  assert calls == []


def test_degenerate_lower_band_redistributes_samples(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  calls = []
  monkeypatch.setattr(dr, "pseudo_inertia", lambda env, ids, **kw: calls.append(ids))
  monkeypatch.setattr(torch, "rand", lambda *a, **kw: torch.tensor([0.1, 0.9]))
  env = cast(Any, NS(num_envs=2, device="cpu"))
  asset = SceneEntityCfg("cart", body_names=("Body",))
  payload_mixture.transition_payload(env, torch.tensor([0, 1]), 20, 10, asset)
  assert [ids.tolist() for ids in calls] == [[0], [1]]
