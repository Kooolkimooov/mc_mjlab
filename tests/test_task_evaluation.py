"""Per-task evaluation declarations and the comparison sections they drive."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
from compare_to_baseline import _report_task, _resolve_evaluation
from conftest import requires_controller
from evaluation.comparison import ComparisonEpisode, by_stratum
from mjlab.envs import ManagerBasedRlEnv

import mc_mjlab.tasks  # noqa: F401
from mc_mjlab.tasks.evaluation import (
  EMPTY,
  TaskEvaluation,
  evaluation_for,
  stratum_labels,
  stratum_of,
)
from mc_mjlab.tasks.locomanip import RESIDUAL_TASK_IDS
from mc_mjlab.tasks.locomanip.evaluation import LOCOMANIP_EVALUATION
from mc_mjlab.tasks.locomanip.profiles import HRP5P

pytestmark = requires_controller(HRP5P.mc_rtc_yaml)


def episode(nth: int, **metrics: float) -> ComparisonEpisode:
  """A comparison record carrying only the metrics a test cares about."""
  return ComparisonEpisode(
    env_id=0, nth=nth, length=100, terms={}, rewards={}, metrics=metrics
  )


def test_a_task_cannot_declare_a_verdict_it_does_not_measure() -> None:
  """Verify the spec rejects a success or strata key outside its own metrics."""
  with pytest.raises(ValueError):
    TaskEvaluation(metrics=("a",), success="b")
  with pytest.raises(ValueError):
    TaskEvaluation(metrics=("a",), stratify="b")
  with pytest.raises(ValueError):
    TaskEvaluation(metrics=("a",), stratify="a", strata=(2.0, 1.0))


def test_locomanip_declares_physical_success_and_cart_mass() -> None:
  """Verify the registered task resolves to its own declaration."""
  assert evaluation_for(RESIDUAL_TASK_IDS["HRP5P"]) is LOCOMANIP_EVALUATION
  assert LOCOMANIP_EVALUATION.success == "task_success"
  assert LOCOMANIP_EVALUATION.stratify == "cart_mass"
  assert evaluation_for("Mjlab-Cartpole-Balance") is EMPTY


def test_strata_cover_everything_above_the_last_edge() -> None:
  """Verify bucketing and its labels agree, including the open top bucket."""
  strata = (10.0, 100.0)
  assert stratum_of(5.0, strata) == 0
  assert stratum_of(10.0, strata) == 0
  assert stratum_of(50.0, strata) == 1
  assert stratum_of(500.0, strata) == 2
  assert stratum_labels(strata) == ("<=10", "10-100", ">100")
  assert len(stratum_labels(strata)) == len(strata) + 1


def test_episodes_bucket_by_their_own_metric() -> None:
  """Verify episodes land in the stratum their metric value belongs to."""
  episodes = [episode(0, cart_mass=5.0), episode(1, cart_mass=50.0)]
  buckets = by_stratum(episodes, "cart_mass", (10.0, 100.0))
  assert list(buckets) == [0, 1]
  assert buckets[0][0].metrics["cart_mass"] == 5.0


def test_a_missing_metric_narrows_the_declaration() -> None:
  """Verify an env without the declared metrics loses the verdict, not the run."""
  env = cast(
    ManagerBasedRlEnv,
    SimpleNamespace(
      metrics_manager=SimpleNamespace(active_terms=["zmp_error", "cart_mass"])
    ),
  )
  narrowed, names = _resolve_evaluation(RESIDUAL_TASK_IDS["HRP5P"], env)
  assert names == ["zmp_error", "cart_mass"]
  assert narrowed.success is None
  assert narrowed.stratify == "cart_mass"
  assert narrowed.strata == LOCOMANIP_EVALUATION.strata


def test_the_task_section_reports_the_strata_that_have_episodes(
  capsys: pytest.CaptureFixture[str],
) -> None:
  """Verify the report prints the metric table, the verdict and the occupied strata."""
  baseline = [
    episode(0, task_complete=0.0, cart_mass=5.0),
    episode(1, task_complete=0.0, cart_mass=500.0),
  ]
  policy = [
    episode(0, task_complete=1.0, cart_mass=5.0),
    episode(1, task_complete=1.0, cart_mass=500.0),
  ]
  evaluation = TaskEvaluation(
    metrics=("task_complete", "cart_mass"),
    success="task_complete",
    stratify="cart_mass",
    strata=(10.0, 100.0),
  )
  _report_task(baseline, policy, evaluation, evaluation.metrics)
  printed = capsys.readouterr().out
  assert "task_complete rate" in printed
  assert "task_complete by cart_mass" in printed
  labels = stratum_labels(evaluation.strata)
  assert labels[0] in printed and labels[2] in printed
  # An empty stratum is dropped rather than printed as a row of NaNs.
  assert labels[1] not in printed
