"""The worker side maps, by name and row window, the blocks the trainer creates."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from utils.shared_memory import create_shm, row_window

BINARY = Path(
  os.environ.get(
    "MC_RTC_SHARED_MEMORY_TEST",
    Path(__file__).resolve().parents[1] / "build" / "test_shared_memory",
  )
)

pytestmark = pytest.mark.skipif(not BINARY.exists(), reason=f"{BINARY} is not built")


def test_worker_views_the_trainers_rows() -> None:
  widths = subprocess.run(
    [BINARY, "--widths"], capture_output=True, text=True, check=True
  ).stdout.split()
  in_width, out_width = (int(width) for width in widths)
  inputs = create_shm((4, in_width))
  outputs = create_shm((4, out_width))
  try:
    inputs.arr[:, 0] = [1.0, 2.0, 3.0, 4.0]
    window = [*row_window(inputs, 1, 2), *row_window(outputs, 1, 2)]
    subprocess.run([BINARY, "--view", *(str(field) for field in window)], check=True)
    # Only the two rows in the window are the worker's to write.
    assert list(outputs.arr[:, 0]) == [0.0, 3.0, 4.0, 0.0]
  finally:
    inputs.unlink()
    outputs.unlink()


def test_row_window_rejects_rows_outside_the_block() -> None:
  block = create_shm((2, 3))
  try:
    assert row_window(block, 1, 1) == (block.shm.name, 24, 24)
    for first_row, num_rows in ((0, 0), (0, 3), (2, 1), (-1, 1)):
      with pytest.raises(ValueError):
        row_window(block, first_row, num_rows)
  finally:
    block.unlink()
