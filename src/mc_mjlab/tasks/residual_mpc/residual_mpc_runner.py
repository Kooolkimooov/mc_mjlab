"""Runner that refuses checkpoints written under different action semantics."""

from mc_mjlab.residual_mpc import ACTION_SEMANTICS_VERSION
from mc_mjlab.tasks.residual_balance.residual_balance_runner import (
  ResidualBalanceOnPolicyRunner,
)


class ResidualMpcOnPolicyRunner(ResidualBalanceOnPolicyRunner):
  """Adds a hard action-semantics gate to the provenance-aware runner."""

  SEMANTICS_KEY = "action_semantics_version"

  def save(self, path: str, infos=None) -> None:
    """Stamp the action-semantics version into every checkpoint."""
    super().save(path, {**(infos or {}), self.SEMANTICS_KEY: ACTION_SEMANTICS_VERSION})

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Reject a checkpoint whose blending equation differs from the current one."""
    infos = super().load(path, load_cfg, strict, map_location)
    saved = (infos or {}).get(self.SEMANTICS_KEY)
    # Shapes are unchanged across the versions, so an actor-only load would
    # otherwise succeed and be scored under the wrong equation.
    # docs/residual-mpc.md#paper_torque_blend
    if saved != ACTION_SEMANTICS_VERSION:
      raise RuntimeError(
        f"Checkpoint {path} was written under action semantics "
        f"{saved!r}, but this task builds {ACTION_SEMANTICS_VERSION}. Version 1 "
        "referenced the controller target where eq (23) requires the default "
        "posture, so its actions mean something different. Retrain, or check "
        "out the commit that produced it."
      )
    return infos
