"""Main-side transport for the per-env mc_rtc controllers.

Owns the worker processes (or the single in-process ``ControllerHost``), their
pipes and the two shared-memory I/O blocks, and exposes a small dispatch API to
the action term:

    pool = ControllerPool(...)      # spawns workers (non-blocking)
    metadata = pool.await_ready()   # block until controllers are constructed
    pool.configure(layout)          # allocate shm, hand the layout to the hosts
    pool.reset_envs(indices)        # synchronous controller reset
    pool.dispatch_controller_step(indices)     # async controller step (worker path)
    idx = pool.collect()            # await the outstanding step

``in_np``/``out_np`` are the shared input/output arrays (available after
``configure``); the action writes inputs and reads outputs directly on the hot
path. Torch- and mjlab-free, like the worker-side ``ControllerHost``.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import weakref
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from mc_mjlab.actions.mc_rtc_controller_host import (
  ControllerHost,
  HostMetadata,
  IoLayout,
  worker_main,
)


def _shutdown_workers(
  procs: list[mp.process.BaseProcess],
  conns: list[Connection],
  shms: list[SharedMemory],
) -> None:
  """Stop workers and release shared blocks (best-effort).

  Runs via ``weakref.finalize``; must not reference the pool.
  """
  for proc, conn in zip(procs, conns, strict=True):
    try:
      if proc.is_alive():
        conn.send(("stop", None))
    except (BrokenPipeError, OSError):
      pass
  for proc in procs:
    proc.join(timeout=2.0)
    if proc.is_alive():
      proc.terminate()
      proc.join(timeout=1.0)
  for conn in conns:
    try:
      conn.close()
    except OSError:
      pass
  for shm in shms:
    try:
      shm.close()
      shm.unlink()
    except (FileNotFoundError, OSError):
      pass


class ControllerPool:
  """Owns the controller workers/host, their pipes and the shared I/O blocks.

  One async step is outstanding at a time: ``dispatch_controller_step`` sends without
  blocking and ``collect`` awaits it, so each worker holds at most one command.
  """

  # Assigned in `configure`; the action reads/writes these on the hot path.
  in_np: np.ndarray
  out_np: np.ndarray

  def __init__(
    self,
    config_path: str,
    num_envs: int,
    target_names: Sequence[str],
    num_workers: int | None = None,
    use_worker_processes: bool = True,
  ):
    self._config_path = config_path
    self._num_envs = num_envs
    self._target_names = list(target_names)
    self._use_worker_processes = use_worker_processes

    self._procs: list[mp.process.BaseProcess] = []
    self._conns: list[Connection] = []
    self._shms: list[SharedMemory] = []
    self._worker_env_ids: list[list[int]] = []
    self._worker_of = np.empty(num_envs, dtype=np.intp)
    self._host: ControllerHost | None = None
    self._thread_pool: ThreadPoolExecutor | None = None
    self._t0 = 0.0

    # The single outstanding async step (worker path).
    self._inflight_workers: list[int] = []
    self._dispatched_indices: list[int] = []

    if use_worker_processes:
      self._start_workers(num_workers)
    else:
      n = max(1, num_workers or 1)
      if n > 1:
        self._thread_pool = ThreadPoolExecutor(max_workers=n)

  # ---- Construction. ----

  def _start_workers(self, num_workers: int | None) -> None:
    """Spawn the worker processes (non-blocking); metadata is awaited later."""
    n = min(self._num_envs, max(1, num_workers or ((os.cpu_count() or 4) - 2)))
    print(
      f"[mc_rtc] constructing {self._num_envs} controllers across "
      f"{n} worker processes..."
    )
    self._t0 = time.perf_counter()
    splits = np.array_split(np.arange(self._num_envs), n)
    self._worker_env_ids = [s.tolist() for s in splits]
    # Forkserver so workers skip re-importing the launch script (spawn would
    # pull torch/mjlab into each worker; ~20% slower startup). Any non-empty
    # preload keeps the server from importing __main__; it must not pull in
    # numpy or the bindings, since a forked server must stay single-threaded
    # and numpy's import starts OpenBLAS threads. The env var makes worker
    # numpy skip its thread pool too (workers do no BLAS).
    ctx = mp.get_context("forkserver")
    ctx.set_forkserver_preload(["mc_mjlab"])
    saved_blas = os.environ.get("OPENBLAS_NUM_THREADS")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
      for w, env_ids in enumerate(self._worker_env_ids):
        self._worker_of[env_ids] = w
        parent, child = ctx.Pipe()
        proc = ctx.Process(
          target=worker_main,
          args=(child, self._config_path, env_ids, self._target_names),
          daemon=True,
        )
        proc.start()
        child.close()
        self._procs.append(proc)
        self._conns.append(parent)
    finally:
      if saved_blas is None:
        del os.environ["OPENBLAS_NUM_THREADS"]
      else:
        os.environ["OPENBLAS_NUM_THREADS"] = saved_blas
    # Registered here so cleanup runs even if the owner's construction raises;
    # the lists are captured by reference, covering the shm blocks below.
    self._finalizer = weakref.finalize(
      self, _shutdown_workers, self._procs, self._conns, self._shms
    )

  def await_ready(self) -> HostMetadata:
    """Block until the controllers are constructed and return their metadata."""
    if not self._use_worker_processes:
      self._host = ControllerHost(
        self._config_path, range(self._num_envs), self._target_names
      )
      return self._host.metadata()
    metadata: HostMetadata | None = None
    for w in range(len(self._conns)):
      tag, payload = self._recv(w)
      if tag == "error":
        self.close()
        if "ImportError" in payload:
          raise ImportError(f"mc_rtc worker failed to start:\n{payload}")
        raise RuntimeError(f"mc_rtc worker failed to start:\n{payload}")
      metadata = payload
    assert metadata is not None
    print(f"[mc_rtc] controllers ready in {time.perf_counter() - self._t0:.1f}s")
    return metadata

  def configure(self, layout: IoLayout) -> None:
    """Allocate the I/O blocks and hand the layout to the hosts."""
    in_shape = (self._num_envs, layout.in_width)
    out_shape = (self._num_envs, layout.out_width)
    if self._use_worker_processes:
      in_shm = SharedMemory(create=True, size=8 * in_shape[0] * in_shape[1])
      out_shm = SharedMemory(create=True, size=8 * out_shape[0] * out_shape[1])
      # in place: the finalizer holds this list
      self._shms += [in_shm, out_shm]
      self.in_np = np.ndarray(in_shape, dtype=np.float64, buffer=in_shm.buf)
      self.out_np = np.ndarray(out_shape, dtype=np.float64, buffer=out_shm.buf)
      self.in_np[:] = 0.0
      self.out_np[:] = 0.0
      for conn in self._conns:
        conn.send(
          ("configure", (layout, in_shm.name, out_shm.name, in_shape, out_shape))
        )
      self._await_ok("configure")
      # All workers attached: unlink now. POSIX keeps the memory alive until
      # the last mapping closes, so nothing leaks even on SIGKILL.
      in_shm.unlink()
      out_shm.unlink()
    else:
      self.in_np = np.zeros(in_shape, dtype=np.float64)
      self.out_np = np.zeros(out_shape, dtype=np.float64)
      assert self._host is not None
      self._host.configure(layout)

  # ---- Dispatch. ----

  def reset_envs(self, env_indices: list[int]) -> None:
    """Reset the given envs' controllers and wait for completion."""
    if not env_indices:
      return
    if self._host is not None:
      self._host.reset_envs(env_indices, self.in_np)
      return
    workers = self._worker_of[env_indices]
    active_workers = []
    for w in np.unique(workers):
      worker_env_indices = [env_indices[k] for k in np.flatnonzero(workers == w)]
      self._conns[w].send(("reset", worker_env_indices))
      active_workers.append(int(w))
    self._await_ok("reset", active_workers)

  def dispatch_controller_step(self, run_indices: list[int]) -> None:
    """Issue a controller step for ``run_indices`` without blocking.

    Worker path: send the per-worker batches and return; the replies are
    awaited by ``collect``. In-process path: no async is possible, so step now
    (results land in ``out_np``) and let ``collect`` read them next period,
    keeping the one-period-behind timing identical to the worker path.
    """
    if not run_indices:
      return
    if self._host is not None:
      if self._thread_pool is not None and len(run_indices) > 1:
        list(
          self._thread_pool.map(
            partial(self._host.step_env, in_arr=self.in_np, out_arr=self.out_np),
            run_indices,
          )
        )
      else:
        self._host.step_envs(run_indices, self.in_np, self.out_np)
      self._dispatched_indices = run_indices
      return
    workers = self._worker_of[run_indices]
    for w in np.unique(workers):
      worker_env_indices = [run_indices[k] for k in np.flatnonzero(workers == w)]
      self._conns[w].send(("step", worker_env_indices))
      self._inflight_workers.append(int(w))
    self._dispatched_indices = run_indices

  def collect(self) -> list[int] | None:
    """Await the outstanding step; return its env indices (outputs now in
    ``out_np``), or ``None`` if nothing was in flight.

    The await guarantees the workers have finished reading ``in_np`` and writing
    ``out_np``, so it must run before either block is reused.
    """
    if not self._dispatched_indices:
      return None
    if self._inflight_workers:
      self._await_ok("step", self._inflight_workers)
      self._inflight_workers = []
    env_indices = self._dispatched_indices
    self._dispatched_indices = []
    return env_indices

  def close(self) -> None:
    """Stop the workers and release the shared blocks."""
    _shutdown_workers(self._procs, self._conns, self._shms)
    self._procs, self._conns, self._shms = [], [], []

  # ---- Pipe helpers. ----

  def _recv(self, w: int) -> tuple[str, object]:
    """Receive one message from worker ``w``, watching for a dead process."""
    conn, proc = self._conns[w], self._procs[w]
    try:
      while not conn.poll(timeout=1.0):
        if not proc.is_alive():
          break
      else:
        return conn.recv()
    except (EOFError, ConnectionResetError, BrokenPipeError):
      pass
    raise RuntimeError(
      f"mc_rtc worker {w} died (exit code {proc.exitcode}); it hosts envs "
      f"{self._worker_env_ids[w][0]}..{self._worker_env_ids[w][-1]}"
    )

  def _await_ok(self, what: str, workers: list[int] | None = None) -> None:
    """Collect one reply per worker; raise with the worker traceback on error."""
    for w in workers if workers is not None else range(len(self._conns)):
      tag, payload = self._recv(w)
      if tag != "ok":
        raise RuntimeError(f"mc_rtc worker {w} failed during {what}:\n{payload}")
