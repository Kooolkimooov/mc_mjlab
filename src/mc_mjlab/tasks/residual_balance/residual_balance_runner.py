"""Runner that adds the balance task's budget, curriculum and watchdog hooks."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from mjlab.rl import RslRlVecEnvWrapper

from mc_mjlab.rl.runner import McRtcResidualOnPolicyRunner
from mc_mjlab.tasks.residual_balance import qualification_sidecar
from mc_mjlab.tasks.residual_balance.achievement_curriculum import (
  AchievementCurriculumBridge,
)
from mc_mjlab.tasks.residual_balance.residual_balance_diagnostics import (
  ppo_diagnostics,
  training_budget,
)
from mc_mjlab.tasks.residual_balance.training_watchdog import (
  RunnerWatchdogBridge,
  WatchdogStop,
)


class ResidualBalanceOnPolicyRunner(McRtcResidualOnPolicyRunner):
  """Train the balance task under its achievement curriculum and health watchdog."""

  BUDGET_KEY = "training_budget"
  ACHIEVEMENT_KEY = "achievement_curriculum"
  WATCHDOG_KEY = "training_watchdog"

  def setup_task_hooks(
    self, env: RslRlVecEnvWrapper, train_cfg: dict, log_dir: str | None
  ) -> None:
    """Attach the iteration budget, the achievement curriculum and the watchdog."""
    self._training_budget = training_budget(env.num_envs, train_cfg)
    self._achievement = AchievementCurriculumBridge(self, log_dir)
    self._watchdog = RunnerWatchdogBridge(self, log_dir)
    # rsl_rl's `learn()` offers no per-iteration hook, so the one logging call it
    # makes is where the diagnostics attach. docs/ppo.md#training-diagnostics
    self.logger.log = self._log_with_diagnostics(  # ty: ignore[invalid-assignment]
      self.logger.log
    )

  def materialize_task_inputs(self, log_dir: Path) -> None:
    """Write the derived iteration budget beside the other run records."""
    (log_dir / "training_budget.json").write_text(
      json.dumps(self._training_budget, indent=2) + "\n"
    )

  def checkpoint_task_state(self) -> dict:
    """Stamp the budget, the watchdog configuration and the curriculum stage."""
    achievement = self._achievement.snapshot()
    return {
      self.BUDGET_KEY: self._training_budget,
      self.WATCHDOG_KEY: self._watchdog.as_config(),
      **({self.ACHIEVEMENT_KEY: achievement} if achievement is not None else {}),
    }

  def checkpoint_saved(self, path: str) -> None:
    """Let the watchdog count the checkpoints it may later roll back to."""
    self._watchdog.checkpoint_saved(path)

  def restore_task_state(self, infos: dict) -> None:
    """Put the achievement curriculum back on the stage the checkpoint holds."""
    self._achievement.restore(infos.get(self.ACHIEVEMENT_KEY))

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    """Run training while publishing an unambiguous terminal lifecycle state."""
    try:
      super().learn(num_learning_iterations, init_at_random_ep_len)
    except WatchdogStop as error:
      print(f"[mc_mjlab] {error}")
      self.logger.stop_logging_writer()
    except BaseException as error:
      self._watchdog.failed(error)
      raise
    else:
      self._watchdog.completed()

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Reject a full resume from a checkpoint no sweep has qualified."""
    infos = super().load(path, load_cfg, strict, map_location)
    # A checkpoint taken mid-transient contaminates everything resumed from it.
    # docs/hard-constraints.md#c4-the-only-valid-resume-point-is-a-checkpoint-whose-sweep-exists
    qualification_sidecar.enforce(path, full_resume=load_cfg is None)
    return infos

  def _log_with_diagnostics(self, log: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap the logger so every iteration also records the PPO diagnostics."""

    def logging_call(*args: Any, **kwargs: Any) -> Any:
      writer = self.logger.writer
      iteration = kwargs.get("it", args[0] if args else None)
      diagnostics = ppo_diagnostics(self.alg)
      if writer is not None and iteration is not None:
        for name, value in diagnostics.items():
          writer.add_scalar(f"Diagnostics/{name}", value, iteration)
        for name, value in self._achievement.iteration(iteration).items():
          writer.add_scalar(f"Curriculum/Achievement/{name}", value, iteration)
      elif iteration is not None:
        self._achievement.iteration(iteration)
      episode_extras = list(self.logger.ep_extras)
      result = log(*args, **kwargs)
      if iteration is not None:
        self._watchdog.iteration(
          iteration,
          diagnostics,
          episode_extras,
          kwargs.get("collect_time", 0.0),
          kwargs.get("learn_time", 0.0),
        )
      return result

    return logging_call
