"""Runner lifecycle shared by every residual task: provenance and contracts."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase
from mc_mjlab.rl.controller_provenance import (
  collect_controller_provenance,
  materialize_provenance,
  provenance_signature,
)
from mc_mjlab.rl.effective_training_manifest import (
  build_effective_training_manifest,
  curriculum_runtime_snapshot,
  materialize_effective_training_manifest,
  synchronize_resumed_curriculum,
  validate_effective_training_manifest,
)
from mc_mjlab.rl.ppo_diagnostics import ppo_diagnostics


class McRtcResidualOnPolicyRunner(MjlabOnPolicyRunner):
  """Persist and validate the base-controller files used by a residual run."""

  PROVENANCE_KEY = "base_controller_provenance"
  MANIFEST_KEY = "effective_training_manifest"
  CURRICULUM_KEY = "curriculum_runtime"

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    super().__init__(env, train_cfg, log_dir, device)
    self._configure_actor_update_mask(env)
    self._controller_provenance = collect_controller_provenance(env)
    self._effective_manifest = build_effective_training_manifest(env, train_cfg)

    self.last_ppo_diagnostics: dict[str, float] = {}
    # rsl_rl's `learn()` offers no per-iteration hook, so the one logging call it
    # makes is where the diagnostics attach. A task wrapping it afterwards runs
    # outside this one and reads `last_ppo_diagnostics` rather than recomputing.
    # docs/ppo.md#training-diagnostics
    self.logger.log = self._log_with_diagnostics(  # ty: ignore[invalid-assignment]
      self.logger.log
    )
    self.setup_task_hooks(env, train_cfg, log_dir)

    if log_dir is not None and int(os.environ.get("RANK", "0")) == 0:
      materialize_provenance(self._controller_provenance, Path(log_dir))
      materialize_effective_training_manifest(self._effective_manifest, Path(log_dir))
      self.materialize_task_inputs(Path(log_dir))

  def save(self, path: str, infos: dict | None = None) -> None:
    """Embed controller inputs in every checkpoint as well as the run directory."""
    infos = {
      **(infos or {}),
      self.PROVENANCE_KEY: self._controller_provenance,
      self.MANIFEST_KEY: self._effective_manifest,
      self.CURRICULUM_KEY: curriculum_runtime_snapshot(self.env),
      **self.checkpoint_task_state(),
    }
    super().save(path, infos)
    self.checkpoint_saved(path)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Reject a checkpoint when its recorded base controller is different."""
    infos = super().load(path, load_cfg, strict, map_location)
    checkpoint_infos = infos or {}

    saved_provenance = checkpoint_infos.get(self.PROVENANCE_KEY)
    if saved_provenance is None:
      print(f"[mc_mjlab] checkpoint {path} predates base-controller provenance")
    elif provenance_signature(saved_provenance) != provenance_signature(
      self._controller_provenance
    ):
      raise RuntimeError(
        "Checkpoint base-controller configuration differs from the active "
        "configuration. Restore the YAML/PD files embedded under infos/"
        f"{self.PROVENANCE_KEY} before loading {path}."
      )

    saved_manifest = checkpoint_infos.get(self.MANIFEST_KEY)
    if saved_manifest is None:
      print(f"[mc_mjlab] checkpoint {path} predates effective-config manifests")
    else:
      validate_effective_training_manifest(
        saved_manifest,
        self._effective_manifest,
        full_resume=load_cfg is None,
      )

    if load_cfg is None:
      synchronize_resumed_curriculum(self.env, checkpoint_infos)
      self.restore_task_state(checkpoint_infos)
      synchronized = curriculum_runtime_snapshot(self.env)
      saved_curriculum = checkpoint_infos.get(self.CURRICULUM_KEY)
      if saved_curriculum is not None and saved_curriculum != synchronized:
        print(
          "[mc_mjlab] curriculum targets were realigned to the restored "
          "common_step_counter"
        )

    return infos

  def setup_task_hooks(
    self, env: RslRlVecEnvWrapper, train_cfg: dict, log_dir: str | None
  ) -> None:
    """Attach whatever the task adds to the shared lifecycle."""

  def materialize_task_inputs(self, log_dir: Path) -> None:
    """Write the task's own run-directory records, on rank zero only."""

  def checkpoint_task_state(self) -> dict:
    """Extra ``infos`` entries the task stamps into every checkpoint."""
    return {}

  def checkpoint_saved(self, path: str) -> None:
    """React to a checkpoint the runner has just written."""

  def restore_task_state(self, infos: dict) -> None:
    """Restore the task's own state during a full resume."""

  def _log_with_diagnostics(self, log: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap the logger so every iteration records the PPO diagnostics first."""

    def logging_call(*args: Any, **kwargs: Any) -> Any:
      iteration = kwargs.get("it", args[0] if args else None)
      self.last_ppo_diagnostics = ppo_diagnostics(self.alg)

      writer = self.logger.writer
      if writer is not None and iteration is not None:
        for name, value in self.last_ppo_diagnostics.items():
          writer.add_scalar(f"Diagnostics/{name}", value, iteration)

      return log(*args, **kwargs)

    return logging_call

  def _configure_actor_update_mask(self, env: RslRlVecEnvWrapper) -> None:
    """Connect PPO actor updates to the authority applied by the action term."""
    configure = getattr(self.alg, "set_actor_update_mask_source", None)
    if not callable(configure):
      return

    action = env.unwrapped.action_manager.get_term("mc_rtc_residual")
    if not isinstance(action, McRtcResidualActionBase):
      raise TypeError("residual runner requires an mc_rtc residual action")
    configure(lambda: action.actor_update_gate)
