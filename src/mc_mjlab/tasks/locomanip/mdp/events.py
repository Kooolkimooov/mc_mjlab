"""The per-episode draw that makes this a payload task: the cart's own mass."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields

from mc_mjlab.tasks.locomanip.mdp import accessors

if TYPE_CHECKING:
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.scene_entity_config import SceneEntityCfg


def mass_alpha_range(
  mass_range_kg: tuple[float, float], nominal_mass_kg: float
) -> tuple[float, float]:
  """Convert a mass range to `pseudo_inertia`'s log scale, where mass is e^(2a)."""
  low, high = mass_range_kg
  if low <= 0.0 or high < low:
    raise ValueError(f"invalid mass range {mass_range_kg}")
  return (
    0.5 * math.log(low / nominal_mass_kg),
    0.5 * math.log(high / nominal_mass_kg),
  )


# `set_const_fixed`, not `pseudo_inertia`'s own `set_const`: every level above it
# blanks every sensor in the scene above four environments, so the reference
# weights that level would recompute are rescaled below instead.
# docs/locomanip.md#cart_mass_range_kg
@requires_model_fields(
  "body_mass",
  "body_ipos",
  "body_inertia",
  "body_iquat",
  "dof_invweight0",
  "body_invweight0",
  recompute=RecomputeLevel.set_const_fixed,
)
def randomize_object_mass(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  mass_range_kg: tuple[float, float],
  nominal_mass_kg: float,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Draw the simulated payload log-uniformly, scaling mass and inertia together."""
  dr.pseudo_inertia(
    env,
    env_ids,
    alpha_range=mass_alpha_range(mass_range_kg, nominal_mass_kg),
    asset_cfg=asset_cfg,
  )
  _rescale_reference_weights(env, asset_cfg)


def _rescale_reference_weights(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> None:
  """Divide the object's own reference weights by the mass scale it just drew."""
  body = int(env.sim.mj_model.body(f"{asset_cfg.name}/{accessors.OBJECT_BODY}").id)
  dof = int(env.sim.mj_model.body_dofadr[body])
  count = int(env.sim.mj_model.body_dofnum[body])

  default_mass = env.sim.get_default_field("body_mass")[body]
  scale = (env.sim.model.body_mass[:, body] / default_mass).unsqueeze(-1)

  # An isolated free body's reference weights are exactly inverse in its uniform
  # density scale, which is what `pseudo_inertia` applies.
  dof_default = env.sim.get_default_field("dof_invweight0")[dof : dof + count]
  env.sim.model.dof_invweight0[:, dof : dof + count] = dof_default / scale
  body_default = env.sim.get_default_field("body_invweight0")[body]
  env.sim.model.body_invweight0[:, body] = body_default / scale
