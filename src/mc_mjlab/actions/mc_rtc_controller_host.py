"""Worker-side mc_rtc controller host.

Owns everything that touches the mc_rtc bindings; imports only numpy, the
stdlib and the bindings -- never torch or mjlab, so it stays light in worker
processes. I/O flows through two ``IoLayout``-shaped shared-memory blocks;
commands travel over a pipe per worker, whose send/recv also orders the
shared-memory writes. The same ``ControllerHost`` serves the in-process path
(``use_worker_processes=False``).
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import tempfile
import time
import traceback
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory

import numpy as np

# Out-of-tree bindings; the host raises ImportError on construction if absent.
try:
  import eigen
  import mc_control
  import sva
except ImportError:
  mc_control = None
  sva = None
  eigen = None


@contextlib.contextmanager
def suppress_mc_rtc_output() -> Iterator[None]:
  """Silence mc_rtc's terminal logging for the duration of the block.

  The spdlog loggers write from C++ and have no config switch, so fds 1/2
  themselves must be redirected (a ``sys.stdout`` swap catches nothing). On
  error the captured text is replayed to the real stderr. The trailing
  flush-drain sleep makes this a cold-path tool (construction, reset); the
  step path uses ``redirect_output_to_devnull``.
  """
  sys.stdout.flush()
  sys.stderr.flush()
  saved_out, saved_err = os.dup(1), os.dup(2)
  capture = tempfile.TemporaryFile()
  try:
    os.dup2(capture.fileno(), 1)
    os.dup2(capture.fileno(), 2)
    try:
      yield
    finally:
      # Let spdlog's async flush thread drain before restoring the fds.
      time.sleep(0.05)
      sys.stdout.flush()
      sys.stderr.flush()
      os.dup2(saved_out, 1)
      os.dup2(saved_err, 2)
  except BaseException:
    capture.seek(0)
    os.write(saved_err, capture.read())
    raise
  finally:
    os.close(saved_out)
    os.close(saved_err)
    capture.close()


@contextlib.contextmanager
def redirect_output_to_devnull() -> Iterator[None]:
  """Discard fds 1/2 for the duration of the block, cheaply.

  No capture, no replay and no flush-drain sleep, so it is safe on the step
  path; the price is that a line spdlog flushes asynchronously right after
  the block can slip through.
  """
  sys.stdout.flush()
  sys.stderr.flush()
  saved_out, saved_err = os.dup(1), os.dup(2)
  devnull = os.open(os.devnull, os.O_WRONLY)
  try:
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
      yield
    finally:
      sys.stdout.flush()
      sys.stderr.flush()
      os.dup2(saved_out, 1)
      os.dup2(saved_err, 2)
  finally:
    os.close(devnull)
    os.close(saved_out)
    os.close(saved_err)


class _IterItemsDict(dict):
  """``setWrenches`` iterates with Python-2 ``.iteritems()``; alias it."""

  def iteritems(self):
    return iter(self.items())


@dataclass(frozen=True)
class HostMetadata:
  """What the trainer needs to know about the mc_rtc robot, probed worker-side."""

  ref_joint_order: tuple[str, ...]
  body_sensor_names: tuple[str, ...]
  force_sensor_names: tuple[str, ...]
  has_named_setters: bool
  has_reset: bool


@dataclass(frozen=True)
class IoLayout:
  """Column layout of the shared input/output blocks (one row per env).

  Input row::

    [0, T)          target-joint positions (encoders)
    [T, 2T)         target-joint velocities
    [2T, 3T)        target-joint torques (qfrc_actuator)
    [3T, 3T+16)     root block; the first 7 are always pos(3) + quat wxyz(4):
                      named routing:   qpos7, qvel6, qacc3
                      singular routing: pos3, quat4, linvel3, omega_body3, accel3
    [imu_off, ...)  6 per IMU body sensor: gyro(3), accel(3)
    [wrench_off, ..) 6 per force sensor: force(3), torque(3) as MuJoCo reads them

  Output row: target q ``[0, T)`` and alpha ``[T, 2T)`` for the target joints.
  """

  num_targets: int
  named_routing: bool
  has_floating_base_sensor: bool
  use_reset: bool
  # Singular routing only: the accel slot carries a real accelerometer reading.
  feed_accel_fallback: bool
  # (body sensor name, has_gyro, has_accel) per IMU, in input-block order.
  imu: tuple[tuple[str, bool, bool], ...] = ()
  # Force sensor names, in input-block order.
  wrenches: tuple[str, ...] = ()

  @property
  def root_off(self) -> int:
    return 3 * self.num_targets

  @property
  def imu_off(self) -> int:
    return self.root_off + 16

  @property
  def wrench_off(self) -> int:
    return self.imu_off + 6 * len(self.imu)

  @property
  def in_width(self) -> int:
    return self.wrench_off + 6 * len(self.wrenches)

  @property
  def out_width(self) -> int:
    return 2 * self.num_targets


@dataclass
class _ShmHandle:
  """A shared-memory block and its numpy view, kept alive together."""

  shm: SharedMemory
  arr: np.ndarray = field(repr=False)

  def close(self) -> None:
    self.arr = None  # type: ignore[assignment]
    self.shm.close()


def attach_shm(name: str, shape: tuple[int, int]) -> _ShmHandle:
  """Attach to an existing shared block. ``track=False``: the trainer owns and
  unlinks it; tracking here would double-unlink."""
  shm = SharedMemory(name=name, track=False)
  arr = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)
  return _ShmHandle(shm, arr)


class ControllerHost:
  """Owns a slice of MCGlobalControllers and steps/resets them from I/O rows.

  ``env_ids`` maps global env indices to local controllers; all I/O goes
  through ``IoLayout``-shaped arrays. ``allowed_output_envs`` lists the global
  env ids whose controllers may write to the console; ``None`` (the worker
  path, where silencing is fd-level for the whole process) suppresses nothing.
  """

  def __init__(
    self,
    config_path: str,
    env_ids: Sequence[int],
    target_names: Sequence[str],
    allowed_output_envs: Sequence[int] | None = None,
  ):
    if mc_control is None:
      raise ImportError(
        "mc_control, mc_rbdyn, sva, eigen modules are required for mc_rtc integration."
      )
    self._env_ids = list(env_ids)
    self._local_of = {env_id: k for k, env_id in enumerate(self._env_ids)}
    self._target_names = list(target_names)
    self._allowed_output = (
      None if allowed_output_envs is None else frozenset(allowed_output_envs)
    )

    self._controllers = []
    for env_id in self._env_ids:
      with self._output_guard(env_id):
        self._controllers.append(mc_control.MCGlobalController(config_path))
    # mc_mujoco calls init() exactly once per controller; later resets go
    # through MCGlobalController::reset().
    self._initialized = [False] * len(self._controllers)

    robot = self._controllers[0].robot()
    rn = robot.name()
    self._robot_key: bytes = rn if isinstance(rn, bytes) else rn.encode()

    # jointIndexByName on a missing joint throws a C++ std::out_of_range that
    # terminates the process (it cannot be caught from Python), so always
    # probe with hasJoint first.
    def joint_index(name: str) -> int:
      return robot.jointIndexByName(name) if robot.hasJoint(name) else -1

    # refJointOrder includes joints mjlab does not simulate; those keep their
    # default-stance values.
    self._ref_joint_order = [
      j.decode() if isinstance(j, bytes) else j
      for j in robot.module().ref_joint_order()
    ]
    stance_q = robot.mbc.q
    self._default_encoders = np.array(
      [
        stance_q[joint_index(name)][0]
        if joint_index(name) != -1 and len(stance_q[joint_index(name)]) > 0
        else 0.0
        for name in self._ref_joint_order
      ],
      dtype=np.float64,
    )

    target_to_ref = [
      self._ref_joint_order.index(n) if n in self._ref_joint_order else -1
      for n in self._target_names
    ]
    self._valid_k = np.array(
      [k for k, r in enumerate(target_to_ref) if r != -1], dtype=np.intp
    )
    self._valid_ref = np.array([r for r in target_to_ref if r != -1], dtype=np.intp)
    self._target_mbc_indices = [joint_index(name) for name in self._target_names]
    if all(i == -1 for i in self._target_mbc_indices):
      robot_name = self._robot_key.decode()
      raise RuntimeError(
        f"none of the mjlab entity's joints exist on the mc_rtc robot "
        f"'{robot_name}': the entity does not match the config's MainRobot."
      )

    body_sensor_names: list[str] = []
    try:
      body_sensor_names = [
        s.name().decode() if isinstance(s.name(), bytes) else s.name()
        for s in robot.bodySensors()
      ]
    except AttributeError:
      # Older binding without bodySensors(): probe conventional names.
      for probe in ("FloatingBase", "Accelerometer"):
        if robot.hasBodySensor(probe):
          body_sensor_names.append(probe)

    # The binding exposes no forceSensors() enumeration; probe known names.
    force_sensor_names = [
      name
      for name in (
        "RightFootForceSensor",
        "LeftFootForceSensor",
        "RightHandForceSensor",
        "LeftHandForceSensor",
      )
      if robot.hasForceSensor(name)
    ]

    self._metadata = HostMetadata(
      ref_joint_order=tuple(self._ref_joint_order),
      body_sensor_names=tuple(body_sensor_names),
      force_sensor_names=tuple(force_sensor_names),
      has_named_setters=hasattr(self._controllers[0], "setSensorPositions"),
      has_reset=hasattr(self._controllers[0], "reset"),
    )
    self._layout: IoLayout | None = None
    self._zero_base = np.zeros(len(self._ref_joint_order), dtype=np.float64)

  def metadata(self) -> HostMetadata:
    return self._metadata

  def configure(self, layout: IoLayout) -> None:
    """Fix the I/O layout and precompute the per-step name keys."""
    self._layout = layout
    self._imu_keys = [name.encode() for name, _, _ in layout.imu]
    self._wrench_keys = [name.encode() for name in layout.wrenches]

  def _output_guard(
    self, env_id: int, hot: bool = False
  ) -> contextlib.AbstractContextManager[None]:
    """Suppression wrapper for one env's controller call.

    A no-op when the env may print; otherwise capture-and-replay on cold
    paths and a plain fd discard on the step path (``hot``). ``step_env``
    itself stays guard-free: fd redirection is process-global, so the
    threaded in-process pool must guard whole batches instead.
    """
    if self._allowed_output is None or env_id in self._allowed_output:
      return contextlib.nullcontext()
    return redirect_output_to_devnull() if hot else suppress_mc_rtc_output()

  def _expand(self, values: np.ndarray, base: np.ndarray) -> list[float]:
    """Expand target-joint values into refJointOrder on top of ``base``."""
    full = base.copy()
    full[self._valid_ref] = values[self._valid_k]
    return full.tolist()

  def reset_envs(self, env_ids: Sequence[int], in_arr: np.ndarray) -> None:
    """init() (first time) or reset() the controllers for the given envs.

    Reads the encoder columns and the first 7 root columns (pos + quat wxyz)
    of each env's input row.
    """
    layout = self._layout
    assert layout is not None
    T = layout.num_targets
    ro = layout.root_off
    for env_id in env_ids:
      with self._output_guard(env_id):
        local = self._local_of[env_id]
        controller = self._controllers[local]
        row = in_arr[env_id]
        encoders = self._expand(row[0:T], self._default_encoders)
        pos = row[ro : ro + 3]
        quat = row[ro + 3 : ro + 7]

        if layout.use_reset and self._initialized[local]:
          # reset() takes the inverse of the MuJoCo world<-body quaternion
          # (the same convention init() applies internally to its 7-array).
          q = eigen.Quaterniond(
            float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
          )
          pose = sva.PTransformd(
            q.inverse(), eigen.Vector3d(float(pos[0]), float(pos[1]), float(pos[2]))
          )
          controller.reset({self._robot_key: encoders}, {self._robot_key: pose})
        else:
          controller.setEncoderValues(encoders)
          # init() attitude is [qw, qx, qy, qz, tx, ty, tz].
          init_attitude = [
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3]),
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
          ]
          controller.init(encoders, init_attitude)
          self._initialized[local] = True

        controller.running = True

  def step_env(self, env_id: int, in_arr: np.ndarray, out_arr: np.ndarray) -> None:
    """Feed one env's input row to its controller, run it, write q/alpha out."""
    layout = self._layout
    assert layout is not None
    T = layout.num_targets
    ro = layout.root_off
    controller = self._controllers[self._local_of[env_id]]
    row = in_arr[env_id]

    controller.setEncoderValues(self._expand(row[0:T], self._default_encoders))
    controller.setEncoderVelocities(self._expand(row[T : 2 * T], self._zero_base))

    if layout.named_routing:
      # mc_mujoco routing: raw free-joint state to "FloatingBase", MuJoCo
      # gyro/accelerometer readings to each IMU body sensor.
      if layout.has_floating_base_sensor:
        fb = b"FloatingBase"
        q = eigen.Quaterniond(
          float(row[ro + 3]), float(row[ro + 4]), float(row[ro + 5]), float(row[ro + 6])
        )
        controller.setSensorPositions(
          {fb: eigen.Vector3d(float(row[ro]), float(row[ro + 1]), float(row[ro + 2]))}
        )
        controller.setSensorOrientations({fb: q.inverse()})
        controller.setSensorLinearVelocities(
          {
            fb: eigen.Vector3d(
              float(row[ro + 7]), float(row[ro + 8]), float(row[ro + 9])
            )
          }
        )
        controller.setSensorAngularVelocities(
          {
            fb: eigen.Vector3d(
              float(row[ro + 10]), float(row[ro + 11]), float(row[ro + 12])
            )
          }
        )
        controller.setSensorLinearAccelerations(
          {
            fb: eigen.Vector3d(
              float(row[ro + 13]), float(row[ro + 14]), float(row[ro + 15])
            )
          }
        )
      gyros = {}
      accels = {}
      for i, (key, (_, has_gyro, has_accel)) in enumerate(
        zip(self._imu_keys, layout.imu, strict=True)
      ):
        off = layout.imu_off + 6 * i
        if has_gyro:
          gyros[key] = eigen.Vector3d(
            float(row[off]), float(row[off + 1]), float(row[off + 2])
          )
        if has_accel:
          accels[key] = eigen.Vector3d(
            float(row[off + 3]), float(row[off + 4]), float(row[off + 5])
          )
      if gyros:
        controller.setSensorAngularVelocities(gyros)
      if accels:
        controller.setSensorLinearAccelerations(accels)
    else:
      # Singular-setter fallback (older binding): everything lands on
      # bodySensors[0]. The trainer already computed omega in the base frame.
      q = eigen.Quaterniond(
        float(row[ro + 3]), float(row[ro + 4]), float(row[ro + 5]), float(row[ro + 6])
      )
      controller.setSensorPosition(
        eigen.Vector3d(float(row[ro]), float(row[ro + 1]), float(row[ro + 2]))
      )
      controller.setSensorOrientation(q.inverse())
      controller.setSensorLinearVelocity(
        eigen.Vector3d(float(row[ro + 7]), float(row[ro + 8]), float(row[ro + 9]))
      )
      controller.setSensorAngularVelocity(
        eigen.Vector3d(float(row[ro + 10]), float(row[ro + 11]), float(row[ro + 12]))
      )
      if layout.feed_accel_fallback:
        controller.setSensorLinearAcceleration(
          eigen.Vector3d(float(row[ro + 13]), float(row[ro + 14]), float(row[ro + 15]))
        )

    # MuJoCo reports the force on the sensor site; mc_rtc wants the reaction
    # on the robot, hence the negation (mc_mujoco's `fs *= -1`).
    if self._wrench_keys:
      wrenches = _IterItemsDict()
      for i, key in enumerate(self._wrench_keys):
        off = layout.wrench_off + 6 * i
        wrenches[key] = sva.ForceVecd(
          [-float(row[off + 3]), -float(row[off + 4]), -float(row[off + 5])],
          [-float(row[off]), -float(row[off + 1]), -float(row[off + 2])],
        )
      controller.setWrenches(wrenches)

    controller.setJointTorques(self._expand(row[2 * T : 3 * T], self._zero_base))

    controller.run()

    mbc = controller.robot().mbc
    mbc_q, mbc_alpha = mbc.q, mbc.alpha
    out_row = out_arr[env_id]
    for k, j in enumerate(self._target_mbc_indices):
      if j != -1 and len(mbc_q[j]) > 0:
        out_row[k] = mbc_q[j][0]
        out_row[T + k] = mbc_alpha[j][0]
      else:
        out_row[k] = 0.0
        out_row[T + k] = 0.0

  def step_envs(
    self, env_ids: Sequence[int], in_arr: np.ndarray, out_arr: np.ndarray
  ) -> None:
    if self._allowed_output is None:
      for env_id in env_ids:
        self.step_env(env_id, in_arr, out_arr)
      return
    for env_id in env_ids:
      with self._output_guard(env_id, hot=True):
        self.step_env(env_id, in_arr, out_arr)


def worker_main(
  conn: Connection,
  config_path: str,
  env_ids: Sequence[int],
  target_names: Sequence[str],
  suppress_output: bool = False,
) -> None:
  """Entry point of a controller worker process.

  Messages are ``(tag, payload)`` tuples. Startup sends ``("meta",
  HostMetadata)`` or ``("error", tb)``; then ``configure``/``step``/``reset``
  commands each reply ``("ok", None)`` or ``("error", tb)`` (the worker stays
  alive); ``stop`` replies and exits.

  ``suppress_output`` silences the whole process by pointing fds 1/2 at a
  capture file once at startup (zero per-step cost); error replies carry the
  captured tail so mc_rtc's own error text is not lost. The
  ``MC_MJLAB_WORKER_LOG_DIR`` debug hook takes precedence: output then goes
  to a per-worker log file instead.
  """
  # Ctrl+C hits the whole foreground process group; shutdown is coordinated by
  # the trainer instead ("stop", pipe EOF, or the daemon flag).
  signal.signal(signal.SIGINT, signal.SIG_IGN)
  capture = None
  log_dir = os.environ.get("MC_MJLAB_WORKER_LOG_DIR")
  if log_dir:
    fd = os.open(
      os.path.join(log_dir, f"worker-{os.getpid()}.log"),
      os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
      0o644,
    )
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    import faulthandler

    faulthandler.enable()
  elif suppress_output:
    # spdlog writes from C++, so only an fd-level redirect silences it. A
    # capture file rather than /dev/null so error replies can attach mc_rtc's
    # own error text; reply_ok truncates it to keep it from growing.
    capture = tempfile.TemporaryFile()
    os.dup2(capture.fileno(), 1)
    os.dup2(capture.fileno(), 2)

  def error_payload() -> str:
    tb = traceback.format_exc()
    if capture is None:
      return tb
    try:
      capture.seek(0)
      tail = capture.read()[-8192:].decode(errors="replace")
      capture.seek(0)
      capture.truncate()
    except OSError:
      return tb
    if not tail.strip():
      return tb
    return f"{tb}\n--- captured mc_rtc output (tail) ---\n{tail}"

  def reply_ok() -> None:
    conn.send(("ok", None))
    if capture is not None:
      try:
        capture.seek(0)
        capture.truncate()
      except OSError:
        pass

  host: ControllerHost | None = None
  in_h: _ShmHandle | None = None
  out_h: _ShmHandle | None = None
  try:
    try:
      host = ControllerHost(config_path, env_ids, target_names)
    except BaseException:
      conn.send(("error", error_payload()))
      return
    conn.send(("meta", host.metadata()))

    while True:
      cmd, payload = conn.recv()
      try:
        if cmd == "configure":
          layout, in_name, out_name, in_shape, out_shape = payload
          in_h = attach_shm(in_name, in_shape)
          out_h = attach_shm(out_name, out_shape)
          host.configure(layout)
          reply_ok()
        elif cmd == "step":
          assert in_h is not None and out_h is not None
          host.step_envs(payload, in_h.arr, out_h.arr)
          reply_ok()
        elif cmd == "reset":
          assert in_h is not None
          host.reset_envs(payload, in_h.arr)
          reply_ok()
        elif cmd == "stop":
          conn.send(("ok", None))
          return
        else:
          conn.send(("error", f"unknown command {cmd!r}"))
      except BaseException:
        conn.send(("error", error_payload()))
  except (EOFError, BrokenPipeError, ConnectionResetError, KeyboardInterrupt):
    pass  # trainer went away: exit quietly
  finally:
    if in_h is not None:
      in_h.close()
    if out_h is not None:
      out_h.close()
    if capture is not None:
      capture.close()
