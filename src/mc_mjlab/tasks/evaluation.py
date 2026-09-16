"""What each task wants measured when one of its checkpoints is scored."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskEvaluation:
  """The task-specific half of a comparison: its metrics, its verdict, its strata."""

  metrics: tuple[str, ...] = ()
  """Metric manager terms to record per episode, beside reward and terminations."""

  success: str | None = None
  """Metric that is 1.0 on a completed episode; needs ``reduce="last"``."""

  stratify: str | None = None
  """Metric whose per-episode value buckets the report."""

  strata: tuple[float, ...] = field(default_factory=tuple)
  """Upper edges of those buckets, ascending; values above the last form one more."""

  def __post_init__(self) -> None:
    for name in (self.success, self.stratify):
      if name is not None and name not in self.metrics:
        raise ValueError(f"{name!r} must be one of this task's metrics")
    if list(self.strata) != sorted(self.strata):
      raise ValueError(f"strata edges must ascend: {self.strata}")
    if self.strata and self.stratify is None:
      raise ValueError("strata edges without a metric to bucket")


EMPTY = TaskEvaluation()

_REGISTRY: dict[str, TaskEvaluation] = {}


def register_evaluation(task_id: str, evaluation: TaskEvaluation) -> None:
  """Declare how this task's checkpoints are scored, keyed by its registered id."""
  _REGISTRY[task_id] = evaluation


def evaluation_for(task_id: str | None) -> TaskEvaluation:
  """The task's declaration, or an empty one when it has not made one."""
  if task_id is None:
    return EMPTY
  return _REGISTRY.get(task_id, EMPTY)


def stratum_of(value: float, strata: tuple[float, ...]) -> int:
  """Index of the bucket a value falls in, the last one being everything above."""
  for index, edge in enumerate(strata):
    if value <= edge:
      return index
  return len(strata)


def stratum_labels(strata: tuple[float, ...]) -> tuple[str, ...]:
  """Human-readable ranges for the buckets, in the same order."""
  labels = []
  low = None
  for edge in strata:
    labels.append(f"<={edge:g}" if low is None else f"{low:g}-{edge:g}")
    low = edge
  labels.append(f">{low:g}" if low is not None else "all")
  return tuple(labels)
