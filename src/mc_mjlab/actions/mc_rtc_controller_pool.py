"""Main-side transport for the per-env mc_rtc controllers."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import time
import weakref
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from typing import Literal

import numpy as np

from mc_mjlab.actions.mc_rtc_controller_host import (
  STATUS_WORKER_FAILED,
  ControllerHost,
  HostMetadata,
  IoLayout,
  redirect_output_to_devnull,
  worker_main,
)

_forensics_armed = False


def _arm_freeze_forensics() -> None:
  """Make a wedged run diagnosable: ``kill -USR1 <pid>`` dumps thread stacks."""
  global _forensics_armed
  if _forensics_armed:
    return
  import faulthandler
  import signal

  faulthandler.enable()
  if hasattr(signal, "SIGUSR1"):  # POSIX only
    faulthandler.register(signal.SIGUSR1, all_threads=True)
  _forensics_armed = True


def _shutdown_workers(
  procs: list[mp.process.BaseProcess],
  conns: list[Connection],
  shms: list[SharedMemory],
) -> None:
  """Stop workers and release shared blocks (best-effort)."""
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
  """Owns the controller workers/host, their pipes and the shared I/O blocks."""

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
    console_output: Literal["none", "single", "all"] = "none",
    timeout_s: float = 60.0,
  ):
    if console_output not in ("none", "single", "all"):
      raise ValueError(
        f"console_output must be 'none', 'single' or 'all', got {console_output!r}"
      )
    self._config_path = config_path
    self._num_envs = num_envs
    self._target_names = list(target_names)
    self._use_worker_processes = use_worker_processes
    self._console_output = console_output
    # Per-command budget. A step is milliseconds and a reset not much more, so
    # this only fires on a genuinely stuck worker; construction gets its own,
    # much larger budget in `await_ready`.
    self.timeout_s = timeout_s

    self._procs: list[mp.process.BaseProcess] = []
    self._conns: list[Connection] = []
    self._shms: list[SharedMemory] = []
    self._worker_env_ids: list[list[int]] = []
    self._worker_of = np.empty(num_envs, dtype=np.intp)
    self._host: ControllerHost | None = None
    self._thread_pool: ThreadPoolExecutor | None = None
    self._t0 = 0.0
    # Respawn state: the forkserver context, per-worker suppress flags and the
    # configure payload, so _revive_worker can rebuild a worker from scratch.
    self._mp_ctx: mp.context.ForkServerContext | None = None
    self._suppress_of: list[bool] = []
    self._layout: IoLayout | None = None
    self._configure_payload: tuple | None = None

    # The single outstanding async step (worker path).
    self._inflight_workers: list[int] = []
    self._dispatched_indices: list[int] = []

    if use_worker_processes:
      _arm_freeze_forensics()
      self._start_workers(num_workers)
    else:
      n = max(1, num_workers or 1)
      if n > 1:
        self._thread_pool = ThreadPoolExecutor(max_workers=n)

  # ---- Construction. ----

  def _start_workers(self, num_workers: int | None) -> None:
    """Spawn the worker processes (non-blocking); metadata is awaited later."""
    budget = min(self._num_envs, max(1, num_workers or ((os.cpu_count() or 4) - 2)))
    if self._console_output == "single" and self._num_envs > 1:
      # Env 0 (the one allowed to print) gets a dedicated worker so every
      # other worker can silence its fds wholesale at startup.
      rest = np.array_split(
        np.arange(1, self._num_envs), min(self._num_envs - 1, max(1, budget - 1))
      )
      splits = [np.array([0]), *rest]
    else:
      splits = np.array_split(np.arange(self._num_envs), budget)
    self._worker_env_ids = [s.tolist() for s in splits]
    print(
      f"[mc_rtc] constructing {self._num_envs} controllers across "
      f"{len(self._worker_env_ids)} worker processes..."
    )
    self._t0 = time.perf_counter()
    # Forkserver so workers skip re-importing the launch script (spawn would
    # pull torch/mjlab into each worker; ~20% slower startup). Any non-empty
    # preload keeps the server from importing __main__; it must not pull in
    # numpy or the bindings, since a forked server must stay single-threaded
    # and numpy's import starts OpenBLAS threads. The env var makes worker
    # numpy skip its thread pool too (workers do no BLAS).
    self._mp_ctx = mp.get_context("forkserver")
    self._mp_ctx.set_forkserver_preload(["mc_mjlab"])
    self._suppress_of = [
      self._console_output == "none"
      or (self._console_output == "single" and 0 not in env_ids)
      for env_ids in self._worker_env_ids
    ]
    # The env var must be set before the first start(): that launches the
    # forkserver, whose environment every worker (respawns included) inherits.
    saved_blas = os.environ.get("OPENBLAS_NUM_THREADS")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
      for w, env_ids in enumerate(self._worker_env_ids):
        self._worker_of[env_ids] = w
        self._spawn_worker(w)
    finally:
      if saved_blas is None:
        del os.environ["OPENBLAS_NUM_THREADS"]
      else:
        os.environ["OPENBLAS_NUM_THREADS"] = saved_blas
    # Registered here so cleanup runs even if the owner's construction raises;
    # the lists are captured by reference, covering the shm blocks below and
    # any respawned workers (revival mutates the list slots in place).
    self._finalizer = weakref.finalize(
      self, _shutdown_workers, self._procs, self._conns, self._shms
    )

  def _spawn_worker(self, w: int) -> None:
    """Start worker ``w``'s process and pipe, filling its slot in place."""
    ctx = self._mp_ctx
    assert ctx is not None
    parent, child = ctx.Pipe()
    proc = ctx.Process(
      target=worker_main,
      args=(
        child,
        self._config_path,
        self._worker_env_ids[w],
        self._target_names,
        self._suppress_of[w],
        # Workers are forked from the forkserver, not from us, so the
        # daemon flag alone cannot reap them if we die abruptly.
        os.getpid(),
      ),
      daemon=True,
    )
    proc.start()
    child.close()
    if w < len(self._procs):
      self._procs[w] = proc
      self._conns[w] = parent
    else:
      self._procs.append(proc)
      self._conns.append(parent)

  def _revive_worker(self, w: int, why: str) -> None:
    """Replace a dead or wedged worker with a fresh process."""
    env_ids = self._worker_env_ids[w]
    print(
      f"[mc_rtc] worker {w} (envs {env_ids[0]}..{env_ids[-1]}) {why}; "
      f"respawning it and rebuilding its controllers...",
      flush=True,
    )
    t0 = time.perf_counter()
    proc = self._procs[w]
    if proc.is_alive():
      proc.terminate()
      proc.join(timeout=2.0)
      if proc.is_alive():
        proc.kill()
        proc.join(timeout=2.0)
    try:
      self._conns[w].close()
    except OSError:
      pass
    self._spawn_worker(w)
    tag, payload = self._recv(
      w, timeout=max(300.0, 30.0 * len(env_ids)), what="respawn"
    )
    if tag == "error":
      raise RuntimeError(f"mc_rtc worker {w} failed to respawn:\n{payload}")
    if self._configure_payload is not None:
      self._conns[w].send(("configure", self._configure_payload))
      tag, payload = self._recv(w, timeout=self.timeout_s, what="respawn configure")
      if tag != "ok":
        raise RuntimeError(f"mc_rtc worker {w} failed to reconfigure:\n{payload}")
    print(
      f"[mc_rtc] worker {w} respawned in {time.perf_counter() - t0:.1f}s", flush=True
    )

  def _allowed_output_envs(self) -> tuple[int, ...] | None:
    """In-process suppression set (``None`` = every env may print)."""
    if self._console_output == "all":
      return None
    return (0,) if self._console_output == "single" else ()

  def await_ready(self) -> HostMetadata:
    """Block until the controllers are constructed and return their metadata."""
    if not self._use_worker_processes:
      self._host = ControllerHost(
        self._config_path,
        range(self._num_envs),
        self._target_names,
        allowed_output_envs=self._allowed_output_envs(),
      )
      return self._host.metadata()
    metadata: HostMetadata | None = None
    # Construction is ~570 ms per controller and serial within a worker, so it
    # needs a far larger budget than a step; scale it with the biggest share.
    biggest_share = max(len(ids) for ids in self._worker_env_ids)
    build_timeout = max(300.0, 30.0 * biggest_share)
    for w in range(len(self._conns)):
      tag, payload = self._recv(w, timeout=build_timeout)
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
      # Kept named (unlinked in close/finalize, resource_tracker as backstop):
      # a respawned worker must be able to re-attach by name.
      self._layout = layout
      self._configure_payload = (
        layout,
        in_shm.name,
        out_shm.name,
        in_shape,
        out_shape,
      )
      for conn in self._conns:
        conn.send(("configure", self._configure_payload))
      self._await_ok("configure")
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
    by_worker: dict[int, list[int]] = {}
    for w in np.unique(workers):
      by_worker[int(w)] = [env_indices[k] for k in np.flatnonzero(workers == w)]
    active_workers = []
    for w, ids in by_worker.items():
      try:
        self._conns[w].send(("reset", ids))
      except (BrokenPipeError, OSError):
        self._revive_worker(w, "died (pipe closed) before reset")
        self._conns[w].send(("reset", ids))
      active_workers.append(w)
    revived = self._await_ok("reset", active_workers, revive=True)
    if revived:
      for w in revived:
        self._conns[w].send(("reset", by_worker[w]))
      self._await_ok("reset retry", revived)

  def dispatch_controller_step(self, run_indices: list[int]) -> None:
    """Issue a controller step for ``run_indices`` without blocking."""
    if not run_indices:
      return
    if self._host is not None:
      if self._thread_pool is not None and len(run_indices) > 1:
        # fd redirection is process-global, so per-env guards cannot run under
        # threads: silence the whole batch in "none" mode; "single" is only
        # honored serially (the host guards per env in step_envs).
        guard = (
          redirect_output_to_devnull()
          if self._console_output == "none"
          else contextlib.nullcontext()
        )
        with guard:
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
      try:
        self._conns[w].send(("step", worker_env_indices))
      except (BrokenPipeError, OSError):
        self._revive_worker(int(w), "died (pipe closed) before step")
        self._mark_failed(worker_env_indices)
        continue
      self._inflight_workers.append(int(w))
    self._dispatched_indices = run_indices

  def collect(self) -> list[int] | None:
    """Await the outstanding step; return its env indices, or ``None`` if idle."""
    if not self._dispatched_indices:
      return None
    if self._inflight_workers:
      revived = self._await_ok("step", self._inflight_workers, revive=True)
      self._inflight_workers = []
      for w in revived:
        self._mark_failed(self._worker_env_ids[w])
    env_indices = self._dispatched_indices
    self._dispatched_indices = []
    return env_indices

  def _mark_failed(self, env_indices: list[int]) -> None:
    """Report the given envs lost with their worker, via the output status column."""
    assert self._layout is not None
    self.out_np[np.asarray(env_indices, dtype=np.intp), self._layout.status_off] = (
      STATUS_WORKER_FAILED
    )

  def close(self) -> None:
    """Stop the workers and release the shared blocks."""
    _shutdown_workers(self._procs, self._conns, self._shms)
    self._procs, self._conns, self._shms = [], [], []

  # ---- Pipe helpers. ----

  def _recv(
    self, w: int, timeout: float | None = None, what: str = "startup"
  ) -> tuple[str, object]:
    """Receive one message from worker ``w``, watching for a dead or wedged one."""
    conn, proc = self._conns[w], self._procs[w]
    deadline = None if timeout is None else time.monotonic() + timeout
    timed_out = False
    try:
      while not conn.poll(timeout=1.0):
        if not proc.is_alive():
          break
        if deadline is not None and time.monotonic() >= deadline:
          timed_out = True
          break
      else:
        return conn.recv()
    except (EOFError, ConnectionResetError, BrokenPipeError):
      pass
    env_ids = self._worker_env_ids[w]
    if timed_out:
      raise TimeoutError(
        f"mc_rtc worker {w} went unresponsive for {timeout:.0f}s during {what}; "
        f"it hosts envs {env_ids[0]}..{env_ids[-1]}. The controller is most "
        f"likely wedged inside mc_rtc ({what}). Re-run with "
        f"MC_MJLAB_WORKER_LOG_DIR=<dir> for per-worker logs with faulthandler "
        f"enabled."
      )
    raise RuntimeError(
      f"mc_rtc worker {w} died (exit code {proc.exitcode}) during {what}; it "
      f"hosts envs {env_ids[0]}..{env_ids[-1]}"
    )

  def _await_ok(
    self,
    what: str,
    workers: list[int] | None = None,
    timeout: float | None = None,
    revive: bool = False,
  ) -> list[int]:
    """Collect one reply per worker; raise with the worker traceback on error."""
    revived: list[int] = []
    for w in workers if workers is not None else range(len(self._conns)):
      try:
        tag, payload = self._recv(
          w, timeout=timeout if timeout is not None else self.timeout_s, what=what
        )
      except (TimeoutError, RuntimeError) as exc:
        if not revive:
          raise
        self._revive_worker(w, str(exc).split(". ")[0].splitlines()[0])
        revived.append(w)
        continue
      if tag != "ok":
        raise RuntimeError(f"mc_rtc worker {w} failed during {what}:\n{payload}")
    return revived
