"""Environment lifecycle and pre-reset episode snapshots."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from mjlab.envs import ManagerBasedRlEnv

from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnvCfg


@dataclass
class EpisodeSnapshot:
  """Completed episode buffers copied before manager reset."""

  lengths: list[int]
  rewards: dict[str, torch.Tensor]
  terminations: dict[str, list[int]]


@contextmanager
def managed_env(cfg: ManagerBasedRlEnvCfg, device: str) -> Iterator[ManagerBasedRlEnv]:
  """Close every controller before releasing the evaluation environment."""
  with ExitStack() as cleanup:
    env = ManagerBasedRlEnv(cfg, device=device)
    cleanup.callback(env.close)
    for name in env.action_manager.active_terms:
      term = env.action_manager.get_term(name)
      if isinstance(term, McRtcResidualActionBase):
        cleanup.callback(term.close)
    yield env


def episode_snapshot(env: ManagerBasedRlEnv, done: torch.Tensor) -> EpisodeSnapshot:
  """Capture completed lengths, reward sums and all termination labels."""
  return EpisodeSnapshot(
    lengths=env.episode_length_buf[done].tolist(),
    rewards={
      name: env.reward_manager._episode_sums[name][done]
      for name in env.reward_manager.active_terms
    },
    terminations={
      name: env.termination_manager.get_term(name)[done].tolist()
      for name in env.termination_manager.active_terms
    },
  )


def metric_snapshot(
  env: ManagerBasedRlEnv, done: torch.Tensor
) -> dict[str, list[float]]:
  """Read true episode metric reductions before reset clears their buffers."""
  manager = env.metrics_manager
  counts = manager._step_count[done].float().clamp(min=1.0)
  output: dict[str, list[float]] = {}
  for index, name in enumerate(manager.active_terms):
    reduce = manager._term_cfgs[index].reduce
    if reduce == "max":
      values = manager._episode_max[name][done]
    elif reduce == "last":
      values = manager._step_values[done, index]
    else:
      values = manager._episode_sums[name][done] / counts
    output[name] = values.tolist()
  return output


def reset_done(env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> None:
  """Recycle envs without appending an extra observation-history frame."""
  # Public reset appends a history frame for every environment, including survivors.
  env._reset_idx(env_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()
