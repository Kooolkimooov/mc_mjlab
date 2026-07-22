"""Residual joint-position action terms backed by per-env mc_rtc controllers.

Controllers live in workers (see ``mc_rtc_host``) because construction, reset
and the Cython marshalling are all GIL-bound or non-thread-safe. I/O flows
through two shared-memory blocks; commands go over pipes.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.envs.mdp.actions.actions import BaseAction, BaseActionCfg
from mjlab.utils.lab_api.string import resolve_matching_names

from mc_mjlab.actions.mc_rtc_controller_host import (
  ControllerHost,
  HostMetadata,
  IoLayout,
  worker_main,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _shutdown_workers(
  procs: list[mp.process.BaseProcess],
  conns: list[Connection],
  shms: list[SharedMemory],
) -> None:
  """Stop workers and release shared blocks (best-effort).

  Runs via ``weakref.finalize``; must not reference the action term.
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


@dataclass(kw_only=True)
class McRtcResidualJointPositionActionCfg(BaseActionCfg):
  """Configuration for mc_rtc residual joint position control."""

  mc_rtc_config_path: str
  """Path to the mc_rtc configuration file."""

  mc_rtc_robot_name: str = "jvrc1"
  """Name of the robot in mc_rtc."""

  frameskip: int = 1
  """Physics substeps per controller step (e.g. 5ms control / 1ms physics -> 5)."""

  num_workers: int | None = None
  """Worker process count; ``None`` = ``min(num_envs, cpu_count - 2)``.

  With ``use_worker_processes=False``: thread count of the in-process pool."""

  use_worker_processes: bool = True
  """When False, host controllers in-process (serial or threaded) -- mainly for
  debugging, since only ``run()`` releases the GIL."""

  pd_gains_path: str | None = None
  """Optional mc_mujoco ``PDgains_sim.dat`` (one ``kp kd`` row per refJointOrder
  joint) overriding the entity's PD gains. Without the real gains a walking
  controller's trajectory is not tracked and the robot falls."""

  feed_wrenches: bool = True
  """Feed matching MuJoCo ``<force>``/``<torque>`` sensor pairs to the controller
  via ``setWrenches``. The stabilizer is a force-feedback loop; without wrenches
  it runs open-loop and falls when stepping."""

  residual_actuator_names: tuple[str, ...] | None = None
  """Actuator names (regex) receiving the RL residual; ``None`` = all controlled
  joints. Non-matched joints track the raw mc_rtc output."""

  use_controller_reset: bool = True
  """Reset via ``MCGlobalController.reset()`` (mc_mujoco parity). Requires the
  locally patched mc_rtc (stock fsm::Controller segfaults on destruction, see
  the GUI/StateBuilder fix). When False, resets re-run ``init()``, which raises
  for plugins that register datastore entries."""

  def build(self, env: ManagerBasedRlEnv) -> "McRtcResidualJointPositionAction":
    return McRtcResidualJointPositionAction(self, env)


class McRtcResidualJointPositionAction(BaseAction):
  """mc_rtc action term. Evaluates mc_rtc controllers in workers and adds RL residual."""

  cfg: McRtcResidualJointPositionActionCfg

  def __init__(self, cfg: McRtcResidualJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    self.cfg = cfg
    self._mc_rtc_robot_name = cfg.mc_rtc_robot_name

    # Scale/offset/clip were resolved against the full target list by the base
    # class; slice their columns down to the residual subset.
    self._residual_ids: torch.Tensor | None = None
    if cfg.residual_actuator_names is not None:
      ids, _ = resolve_matching_names(cfg.residual_actuator_names, self._target_names)
      self._residual_ids = torch.tensor(ids, device=self.device, dtype=torch.long)
      self._action_dim = len(ids)
      self._raw_actions = torch.zeros(
        self.num_envs, self._action_dim, device=self.device
      )
      self._processed_actions = torch.zeros_like(self._raw_actions)
      if isinstance(self._scale, torch.Tensor):
        self._scale = self._scale[:, ids]
      if isinstance(self._offset, torch.Tensor):
        self._offset = self._offset[:, ids]
      if cfg.clip is not None:
        self._clip = self._clip[:, ids]
      self._residual_full = torch.zeros(
        self.num_envs, len(self._target_names), device=self.device
      )

    self._steps_since_run = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    # Workers launch first so construction overlaps the model lookups below.
    self._procs: list[mp.process.BaseProcess] = []
    self._conns: list[Connection] = []
    self._shms: list[SharedMemory] = []
    self._host: ControllerHost | None = None
    self._pool: ThreadPoolExecutor | None = None
    if cfg.use_worker_processes:
      num_workers = min(
        self.num_envs, max(1, cfg.num_workers or ((os.cpu_count() or 4) - 2))
      )
      print(
        f"[mc_rtc] constructing {self.num_envs} controllers across "
        f"{num_workers} worker processes..."
      )
      t0 = time.perf_counter()
      splits = np.array_split(np.arange(self.num_envs), num_workers)
      self._worker_env_ids = [s.tolist() for s in splits]
      self._worker_of = np.empty(self.num_envs, dtype=np.intp)
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
            args=(child, cfg.mc_rtc_config_path, env_ids, list(self._target_names)),
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
      # Registered before setup so cleanup runs even if __init__ raises; the
      # lists are captured by reference, covering the shm blocks created below.
      self._finalizer = weakref.finalize(
        self, _shutdown_workers, self._procs, self._conns, self._shms
      )
    else:
      num_workers = max(1, cfg.num_workers or 1)
      if num_workers > 1:
        self._pool = ThreadPoolExecutor(max_workers=num_workers)

    # ---- Model-side lookups (overlap with worker construction). ----

    # Model joint names carry the entity prefix (e.g. "robot/RCY").
    mj_model = self._env.sim.mj_model
    dof_by_name = {
      mj_model.joint(j).name: int(mj_model.jnt_dofadr[j]) for j in range(mj_model.njnt)
    }
    self._target_dof_adr: list[int] = []
    entity_prefix = ""
    for name in self._target_names:
      match = next(
        ((n, d) for n, d in dof_by_name.items() if n == name or n.endswith("/" + name)),
        None,
      )
      if match is None:
        raise ValueError(f"Target joint '{name}' not found in the MuJoCo model.")
      self._target_dof_adr.append(match[1])
      entity_prefix = match[0].removesuffix(name)

    # Controllers expect env-local coordinates (mc_mujoco robots live at the
    # world origin); feeding offset positions confuses the observers.
    self._env_origins_np = self._env.scene.env_origins.cpu().numpy()

    # Free (root) joint addresses; -1 when fixed-base.
    self._root_qpos_adr = -1
    self._root_dof_adr = -1
    for j in range(mj_model.njnt):
      jnt = mj_model.joint(j)
      if jnt.type[0] == mujoco.mjtJoint.mjJNT_FREE and jnt.name.startswith(
        entity_prefix
      ):
        self._root_qpos_adr = int(mj_model.jnt_qposadr[j])
        self._root_dof_adr = int(mj_model.jnt_dofadr[j])
        break

    self._target_ids_np = self._target_ids.cpu().numpy()

    # ---- Controller-side metadata (waits for worker construction). ----
    if cfg.use_worker_processes:
      metadata: HostMetadata | None = None
      for w in range(len(self._conns)):
        tag, payload = self._recv(w)
        if tag == "error":
          self._teardown()
          if "ImportError" in payload:
            raise ImportError(f"mc_rtc worker failed to start:\n{payload}")
          raise RuntimeError(f"mc_rtc worker failed to start:\n{payload}")
        metadata = payload
      assert metadata is not None
      print(f"[mc_rtc] controllers ready in {time.perf_counter() - t0:.1f}s")
    else:
      self._host = ControllerHost(
        cfg.mc_rtc_config_path, range(self.num_envs), list(self._target_names)
      )
      metadata = self._host.metadata()

    self._mc_ref_joint_order = list(metadata.ref_joint_order)

    # Applied here, not in the robot cfg: refJointOrder is only known now.
    if cfg.pd_gains_path is not None:
      self._apply_reference_pd_gains(cfg.pd_gains_path)

    # Named routing (mc_mujoco parity: raw base state to "FloatingBase", IMU
    # readings to the other body sensors) needs the extended binding's
    # name-keyed setters; the singular fallback only reaches bodySensors[0].
    self._use_named_routing = metadata.has_named_setters and self._root_qpos_adr >= 0
    use_reset = cfg.use_controller_reset and metadata.has_reset

    # Sensor names follow mc_mujoco ("<sensor>_gyro") with mjlab ("imu_gyro")
    # as fallback.
    imu_sensors: list[tuple[str, int, int]] = []
    for bs_name in metadata.body_sensor_names:
      if not bs_name or bs_name == "FloatingBase":
        continue
      g_adr = self._sensor_adr(f"{bs_name}_gyro")
      if g_adr < 0:
        g_adr = self._sensor_adr("imu_gyro")
      a_adr = self._sensor_adr(f"{bs_name}_accelerometer")
      if a_adr < 0:
        a_adr = self._sensor_adr("imu_accelerometer")
      if g_adr >= 0 or a_adr >= 0:
        imu_sensors.append((bs_name, g_adr, a_adr))

    if imu_sensors:
      self._gyro_adr = imu_sensors[0][1]
      self._accel_adr = imu_sensors[0][2]
    else:
      self._gyro_adr = self._sensor_adr("imu_gyro")
      self._accel_adr = self._sensor_adr("imu_accelerometer")

    # mc_rtc force sensor "X" maps to model sensors "<prefix>X_fsensor"/"_tsensor".
    wrench_sensors: list[tuple[str, int, int]] = []
    if cfg.feed_wrenches:
      adr_by_name = {
        mj_model.sensor(i).name: int(mj_model.sensor(i).adr[0])
        for i in range(mj_model.nsensor)
      }
      for name in metadata.force_sensor_names:
        f_adr = next(
          (a for n, a in adr_by_name.items() if n.endswith(name + "_fsensor")), None
        )
        t_adr = next(
          (a for n, a in adr_by_name.items() if n.endswith(name + "_tsensor")), None
        )
        if f_adr is not None and t_adr is not None:
          wrench_sensors.append((name, f_adr, t_adr))
      print(
        f"[mc_rtc] feeding {len(wrench_sensors)} force sensor(s): "
        f"{[n for n, _, _ in wrench_sensors]}"
      )

    # ---- Shared I/O blocks and layout. ----
    layout = IoLayout(
      num_targets=len(self._target_names),
      named_routing=self._use_named_routing,
      has_floating_base_sensor="FloatingBase" in metadata.body_sensor_names,
      use_reset=use_reset,
      feed_accel_fallback=self._accel_adr >= 0,
      imu=tuple((n, g >= 0, a >= 0) for n, g, a in imu_sensors),
      wrenches=tuple(n for n, _, _ in wrench_sensors),
    )
    self._layout = layout

    in_shape = (self.num_envs, layout.in_width)
    out_shape = (self.num_envs, layout.out_width)
    if cfg.use_worker_processes:
      in_shm = SharedMemory(create=True, size=8 * in_shape[0] * in_shape[1])
      out_shm = SharedMemory(create=True, size=8 * out_shape[0] * out_shape[1])
      # in place: the finalizer holds this list
      self._shms += [in_shm, out_shm]
      self._in_np = np.ndarray(in_shape, dtype=np.float64, buffer=in_shm.buf)
      self._out_np = np.ndarray(out_shape, dtype=np.float64, buffer=out_shm.buf)
      self._in_np[:] = 0.0
      self._out_np[:] = 0.0
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
      self._in_np = np.zeros(in_shape, dtype=np.float64)
      self._out_np = np.zeros(out_shape, dtype=np.float64)
      assert self._host is not None
      self._host.configure(layout)

    # Gather columns for a single fancy-indexed sensordata copy per step;
    # missing readings (adr < 0) keep their zeroed columns.
    src_cols: list[int] = []
    dst_cols: list[int] = []
    for i, (_, g_adr, a_adr) in enumerate(imu_sensors):
      off = layout.imu_off + 6 * i
      if g_adr >= 0:
        src_cols += [g_adr, g_adr + 1, g_adr + 2]
        dst_cols += [off, off + 1, off + 2]
      if a_adr >= 0:
        src_cols += [a_adr, a_adr + 1, a_adr + 2]
        dst_cols += [off + 3, off + 4, off + 5]
    for i, (_, f_adr, t_adr) in enumerate(wrench_sensors):
      off = layout.wrench_off + 6 * i
      src_cols += [f_adr, f_adr + 1, f_adr + 2, t_adr, t_adr + 1, t_adr + 2]
      dst_cols += [off, off + 1, off + 2, off + 3, off + 4, off + 5]
    self._sens_src_cols = np.array(src_cols, dtype=np.intp)
    self._sens_dst_cols = np.array(dst_cols, dtype=np.intp)

    # Prev/next controller outputs; targets interpolate between them across the
    # frameskip substeps (mc_mujoco ramps q_ref/alpha_ref instead of stepping).
    # Sized to the full controlled-joint set, not the residual subset.
    num_targets = len(self._target_names)
    self._prev_pos = torch.zeros(self.num_envs, num_targets, device=self.device)
    self._next_pos = torch.zeros(self.num_envs, num_targets, device=self.device)
    self._prev_vel = torch.zeros(self.num_envs, num_targets, device=self.device)
    self._next_vel = torch.zeros(self.num_envs, num_targets, device=self.device)

  # ---- Worker plumbing. ----

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

  def _teardown(self) -> None:
    _shutdown_workers(self._procs, self._conns, self._shms)
    self._procs, self._conns, self._shms = [], [], []

  # ---- Setup helpers. ----

  def _sensor_adr(self, suffix: str) -> int:
    """sensordata offset of the model sensor whose name ends with ``suffix``."""
    mj_model = self._env.sim.mj_model
    for i in range(mj_model.nsensor):
      if mj_model.sensor(i).name.endswith(suffix):
        return int(mj_model.sensor(i).adr[0])
    return -1

  def _apply_reference_pd_gains(self, path: str) -> None:
    """Set the entity's PD gains from an mc_mujoco ``PDgains_sim.dat``.

    Rows pair with ``refJointOrder``; joints missing from the file or the
    reduced model keep their configured gains.
    """
    with open(path) as f:
      rows = [line.split() for line in f if line.strip()]
    gains = {
      name: (float(row[0]), float(row[1]))
      for name, row in zip(self._mc_ref_joint_order, rows, strict=False)
      if len(row) >= 2
    }

    matched = 0
    for act in self._entity.actuators:
      stiffness = getattr(act, "stiffness", None)
      damping = getattr(act, "damping", None)
      if stiffness is None or damping is None:
        continue
      for j, name in enumerate(act.target_names):
        if name in gains:
          kp, kd = gains[name]
          stiffness[:, j] = kp
          damping[:, j] = kd
          matched += 1
    print(
      f"[mc_rtc] applied reference PD gains from {path} to {matched} joints "
      f"({len(gains)} in file, {len(self._target_names)} controlled)."
    )

  # ---- Per-step input assembly. ----

  @staticmethod
  def _ang_vel_world_to_body(quat_wxyz: np.ndarray, omega_w: np.ndarray) -> np.ndarray:
    """Rotate world-frame angular velocities into the base (gyro) frame, batched.

    mc_rtc's BodySensor expects body-frame angular velocity; mjlab measures it
    in world. ``omega_body = R(q)^T omega_w``.
    """
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    rt = np.stack(
      [
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y),
        2 * (x * y - w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z + w * x),
        2 * (x * z + w * y),
        2 * (y * z - w * x),
        1 - 2 * (x * x + y * y),
      ],
      axis=-1,
    ).reshape(-1, 3, 3)
    return np.einsum("nij,nj->ni", rt, omega_w)

  def _fill_joint_columns(self) -> None:
    """Write encoder/velocity/torque columns of the input block (all envs)."""
    T = self._layout.num_targets
    current_pos = self._entity.data.joint_pos.cpu().numpy()
    current_vel = self._entity.data.joint_vel.cpu().numpy()
    self._in_np[:, 0:T] = current_pos[:, self._target_ids_np]
    self._in_np[:, T : 2 * T] = current_vel[:, self._target_ids_np]
    self._in_np[:, 2 * T : 3 * T] = self._env.sim.data.qfrc_actuator.cpu().numpy()[
      :, self._target_dof_adr
    ]

  def _fill_root_and_sensor_columns(self) -> None:
    """Write the root block and the IMU/wrench columns of the input block."""
    layout = self._layout
    ro = layout.root_off
    in_np = self._in_np

    if self._use_named_routing:
      qadr, dadr = self._root_qpos_adr, self._root_dof_adr
      in_np[:, ro : ro + 7] = self._env.sim.data.qpos.cpu().numpy()[:, qadr : qadr + 7]
      in_np[:, ro : ro + 3] -= self._env_origins_np
      in_np[:, ro + 7 : ro + 13] = self._env.sim.data.qvel.cpu().numpy()[
        :, dadr : dadr + 6
      ]
      in_np[:, ro + 13 : ro + 16] = self._env.sim.data.qacc.cpu().numpy()[
        :, dadr : dadr + 3
      ]
    else:
      root_quat = self._entity.data.root_link_quat_w.cpu().numpy()
      in_np[:, ro : ro + 3] = (
        self._entity.data.root_link_pos_w.cpu().numpy() - self._env_origins_np
      )
      in_np[:, ro + 3 : ro + 7] = root_quat
      in_np[:, ro + 7 : ro + 10] = self._entity.data.root_link_lin_vel_w.cpu().numpy()
      # Prefer the real gyro (filled below); fall back to rotating world omega.
      if self._gyro_adr < 0:
        in_np[:, ro + 10 : ro + 13] = self._ang_vel_world_to_body(
          root_quat, self._entity.data.root_link_ang_vel_w.cpu().numpy()
        )

    if len(self._sens_src_cols) or (
      not self._use_named_routing and (self._gyro_adr >= 0 or self._accel_adr >= 0)
    ):
      sensordata = self._env.sim.data.sensordata.cpu().numpy()
      if len(self._sens_src_cols):
        in_np[:, self._sens_dst_cols] = sensordata[:, self._sens_src_cols]
      if not self._use_named_routing:
        if self._gyro_adr >= 0:
          g = self._gyro_adr
          in_np[:, ro + 10 : ro + 13] = sensordata[:, g : g + 3]
        if self._accel_adr >= 0:
          a = self._accel_adr
          in_np[:, ro + 13 : ro + 16] = sensordata[:, a : a + 3]

  def _dispatch(self, cmd: str, env_indices: list[int]) -> None:
    """Run ``cmd`` for the given envs on their hosts and wait for completion."""
    if not env_indices:
      return
    if self._host is not None:
      if cmd == "step":
        if self._pool is not None and len(env_indices) > 1:
          list(
            self._pool.map(
              partial(self._host.step_env, in_arr=self._in_np, out_arr=self._out_np),
              env_indices,
            )
          )
        else:
          self._host.step_envs(env_indices, self._in_np, self._out_np)
      else:
        self._host.reset_envs(env_indices, self._in_np)
      return
    workers = self._worker_of[env_indices]
    active = []
    for w in np.unique(workers):
      ids = [env_indices[k] for k in np.flatnonzero(workers == w)]
      self._conns[w].send((cmd, ids))
      active.append(int(w))
    self._await_ok(cmd, active)

  # ---- ActionTerm API. ----

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids=env_ids)

    if env_ids is None:
      env_ids = slice(None)

    if isinstance(env_ids, slice):
      env_indices = list(range(self.num_envs))[env_ids]
    else:
      env_indices = env_ids.tolist()

    # Hosts read the encoder columns and the first 7 root columns (pos + quat
    # wxyz, same prefix in both routing modes).
    T = self._layout.num_targets
    ro = self._layout.root_off
    self._in_np[:, 0:T] = self._entity.data.joint_pos.cpu().numpy()[
      :, self._target_ids_np
    ]
    if self._root_qpos_adr >= 0:
      adr = self._root_qpos_adr
      self._in_np[:, ro : ro + 7] = self._env.sim.data.qpos.cpu().numpy()[
        :, adr : adr + 7
      ]
    else:
      self._in_np[:, ro : ro + 3] = self._entity.data.root_link_pos_w.cpu().numpy()
      self._in_np[:, ro + 3 : ro + 7] = self._entity.data.root_link_quat_w.cpu().numpy()
    self._in_np[:, ro : ro + 3] -= self._env_origins_np

    self._dispatch("reset", env_indices)
    self._steps_since_run[env_indices] = 0

    # Seed interpolation from the current stance so the first control period
    # does not ramp from zero.
    idx_t = torch.tensor(env_indices, device=self.device, dtype=torch.long)
    stance = self._entity.data.joint_pos[:, self._target_ids]
    self._prev_pos[idx_t] = stance[idx_t]
    self._next_pos[idx_t] = stance[idx_t]
    self._prev_vel[idx_t] = 0.0
    self._next_vel[idx_t] = 0.0

  def apply_actions(self) -> None:
    interp_idx = self._steps_since_run % self.cfg.frameskip
    run_envs = interp_idx == 0

    run_indices = run_envs.nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(run_indices, int):
      run_indices = [run_indices]

    if run_indices:
      self._fill_joint_columns()
      self._fill_root_and_sensor_columns()
      self._dispatch("step", run_indices)

      T = self._layout.num_targets
      out = self._out_np[run_indices]
      index = torch.tensor(run_indices, device=self.device, dtype=torch.long)
      pos = torch.tensor(out[:, 0:T], dtype=self._next_pos.dtype)
      vel = torch.tensor(out[:, T : 2 * T], dtype=self._next_vel.dtype)
      self._prev_pos[index] = self._next_pos[index]
      self._prev_vel[index] = self._next_vel[index]
      self._next_pos[index] = pos.to(self.device)
      self._next_vel[index] = vel.to(self.device)

    # coef=1 on the last substep gives the full new target, matching mc_mujoco.
    coef = ((interp_idx + 1).float() / self.cfg.frameskip).unsqueeze(-1)
    interp_pos = self._prev_pos + coef * (self._next_pos - self._prev_pos)
    interp_vel = self._prev_vel + coef * (self._next_vel - self._prev_vel)

    self._steps_since_run += 1

    # Scatter a restricted residual into the full target width (non-matched
    # joints get 0, i.e. pure mc_rtc tracking); velocity gets no residual.
    residual = self._processed_actions
    if self._residual_ids is not None:
      self._residual_full.zero_()
      self._residual_full[:, self._residual_ids] = residual
      residual = self._residual_full
    target = interp_pos + residual
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)
    self._entity.set_joint_velocity_target(interp_vel, joint_ids=self._target_ids)
