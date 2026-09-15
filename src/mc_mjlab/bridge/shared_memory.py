"""Shared-memory blocks the trainer and its controller workers swap rows in."""

from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory

import numpy as np

ITEM_SIZE = np.dtype(np.float64).itemsize


@dataclass
class ShmHandle:
  """A shared-memory block and its numpy view, kept alive together."""

  shm: SharedMemory
  arr: np.ndarray = field(repr=False)

  def close(self) -> None:
    self.arr = None  # type: ignore[assignment]
    self.shm.close()

  def unlink(self) -> None:
    """Close and remove the block; only its creator may call this."""
    self.close()
    self.shm.unlink()


def attach_shm(name: str, shape: tuple[int, int]) -> ShmHandle:
  """Attach to an existing shared block, untracked: the trainer unlinks it."""
  shm = SharedMemory(name=name, track=False)
  arr = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
  return ShmHandle(shm, arr)


def create_shm(shape: tuple[int, int]) -> ShmHandle:
  """Create a zeroed block; the creator owns it and alone unlinks it."""
  shm = SharedMemory(create=True, size=ITEM_SIZE * shape[0] * shape[1])
  arr = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
  arr[:] = 0.0
  return ShmHandle(shm, arr)


def row_window(
  handle: ShmHandle, first_row: int, num_rows: int
) -> tuple[str, int, int]:
  """The ``(name, offset, size)`` a worker maps to see rows ``first_row..+num_rows``."""
  # Bytes, and the mirror of utils::shared_memory::map_worker_io's own row math.
  rows, width = handle.arr.shape
  if num_rows <= 0 or first_row < 0 or first_row + num_rows > rows:
    raise ValueError(
      f"rows [{first_row}, {first_row + num_rows}) exceed the block's {rows}"
    )
  return handle.shm.name, ITEM_SIZE * first_row * width, ITEM_SIZE * num_rows * width
