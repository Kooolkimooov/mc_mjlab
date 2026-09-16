"""Evaluation lifecycle, sampling and reset contracts."""

from __future__ import annotations

from types import SimpleNamespace as NS
from typing import cast

import pytest
import torch
from evaluation import rollout
from evaluation.comparison import Arm, ComparisonEpisode
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.managers.metrics_manager import MetricsTermCfg


def test_episode_buffers_survive_reset_without_advancing_history() -> None:
  """A finished row is captured before reset without touching survivor history."""
  order: list[str] = []
  lengths = torch.tensor([5, 11])
  rewards = torch.tensor([2.0, 9.0])
  labels = {
    "fell_over": torch.tensor([False, False]),
    "controller_worker_failed": torch.tensor([True, False]),
  }
  history = torch.arange(8).reshape(2, 4)
  before = history.clone()

  def reset(ids: torch.Tensor) -> None:
    order.append("reset")
    lengths[ids] = 0
    rewards[ids] = 0
    for flags in labels.values():
      flags[ids] = False

  env = cast(
    ManagerBasedRlEnv,
    NS(
      episode_length_buf=lengths,
      reward_manager=NS(active_terms=["reward"], _episode_sums={"reward": rewards}),
      termination_manager=NS(active_terms=list(labels), get_term=labels.__getitem__),
      _reset_idx=reset,
      reset=lambda: pytest.fail("public reset corrupts history"),
      observation_manager=NS(compute=lambda: pytest.fail("history advanced")),
      scene=NS(write_data_to_sim=lambda: order.append("write")),
      sim=NS(forward=lambda: order.append("forward")),
    ),
  )
  done = torch.tensor([0])
  snapshot = rollout.episode_snapshot(env, done)
  rollout.reset_done(env, done)
  assert order == ["reset", "write", "forward"]
  assert snapshot.lengths == [5]
  assert snapshot.rewards["reward"].tolist() == [2.0]
  assert snapshot.terminations["controller_worker_failed"] == [True]
  assert snapshot.terminations["fell_over"] == [False]
  assert lengths.tolist() == [0, 11]
  assert rewards.tolist() == [0.0, 9.0]
  torch.testing.assert_close(history, before)


def test_metric_reductions_are_preserved() -> None:
  """Mean, last and maximum metrics retain their separate reduction semantics."""
  manager = NS(
    active_terms=["mean", "last", "max"],
    _term_cfgs=[
      MetricsTermCfg(func=lambda _: None, reduce=mode)
      for mode in ("mean", "last", "max")
    ],
    _step_count=torch.tensor([4, 0]),
    _episode_sums={"mean": torch.tensor([12.0, 0.0])},
    _step_values=torch.tensor([[0.0, 7.0, 0.0], [0.0, 8.0, 0.0]]),
    _episode_max={"max": torch.tensor([9.0, 10.0])},
  )
  env = cast(ManagerBasedRlEnv, NS(metrics_manager=manager))
  assert rollout.metric_snapshot(env, torch.tensor([0, 1])) == {
    "mean": [3.0, 0.0],
    "last": [7.0, 8.0],
    "max": [9.0, 10.0],
  }


@pytest.mark.parametrize("fail_close", [False, True])
def test_controllers_close_before_env_on_failure(
  monkeypatch: pytest.MonkeyPatch, fail_close: bool
) -> None:
  """Failed evaluation and controller cleanup both release every resource."""
  order: list[str] = []

  class Controller:
    """Record one cleanup operation."""

    def __init__(self, name: str) -> None:
      self.name = name

    def close(self) -> None:
      order.append(self.name)
      if fail_close and self.name == "b":
        raise RuntimeError("close failed")

  controllers = {name: Controller(name) for name in ("a", "b")}
  env = NS(
    action_manager=NS(active_terms=list(controllers), get_term=controllers.__getitem__),
    close=lambda: order.append("env"),
  )
  monkeypatch.setattr(rollout, "ManagerBasedRlEnv", lambda *a, **kw: env)
  monkeypatch.setattr(rollout, "McRtcResidualActionBase", Controller)
  with pytest.raises(
    RuntimeError, match="close failed" if fail_close else "load failed"
  ):
    with rollout.managed_env(cast(ManagerBasedRlEnvCfg, None), "cpu"):
      raise RuntimeError("load failed")
  assert order == ["b", "a", "env"]


def test_incomplete_environment_is_not_dropped_from_comparison() -> None:
  """An unfinished survivor prevents a biased completed-episodes comparison."""
  arm = Arm(label="baseline", env_ids=(0, 1))
  arm.episodes = [ComparisonEpisode(0, 0, 5, {"fell_over": 1}, {})]
  assert arm.trimmed() == ([], 0)
  arm.episodes.append(ComparisonEpisode(1, 0, 20, {"time_out": 1}, {}))
  assert arm.trimmed() == (arm.episodes, 1)
