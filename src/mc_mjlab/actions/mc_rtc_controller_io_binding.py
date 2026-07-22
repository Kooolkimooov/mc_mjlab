"""Sim <-> mc_rtc input wiring.

Translates mjlab sim state into the mc_rtc input block and owns the ``IoLayout``
that describes the shared blocks. At construction it introspects the MuJoCo model
(target->dof addresses, the root free-joint, IMU/force sensor addresses), decides
the sensor routing and builds the layout; per step it fills the input block from
``entity.data``/``sim.data``. This is the mjlab-coupled counterpart to the
torch-free ``ControllerPool`` (transport) and ``ControllerHost`` (worker compute).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mc_mjlab.actions.mc_rtc_controller_host import HostMetadata, IoLayout

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def apply_reference_pd_gains(
  entity,
  ref_joint_order: Sequence[str],
  target_names: Sequence[str],
  path: str,
) -> None:
  """Set the entity's PD gains from an mc_mujoco ``PDgains_sim.dat``.

  Rows pair with ``ref_joint_order``; joints missing from the file or the reduced
  model keep their configured gains.
  """
  with open(path) as f:
    rows = [line.split() for line in f if line.strip()]
  gains = {
    name: (float(row[0]), float(row[1]))
    for name, row in zip(ref_joint_order, rows, strict=False)
    if len(row) >= 2
  }

  matched = 0
  for act in entity.actuators:
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
    f"({len(gains)} in file, {len(target_names)} controlled)."
  )


class ControllerIoBinding:
  """Owns the ``IoLayout`` and fills the mc_rtc input block from the sim.

  Resolves all model addresses and the sensor routing once at construction;
  ``fill_controller_input``/``reset_controller_input`` write the shared input
  array each step/reset.
  """

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    entity,
    target_names: Sequence[str],
    target_ids: torch.Tensor,
    metadata: HostMetadata,
    use_controller_reset: bool,
    output_channels: Sequence[str],
  ):
    self._env = env
    self._entity = entity
    self._target_names = list(target_names)
    self._target_ids_np = target_ids.cpu().numpy()
    self._device = target_ids.device
    self._output_channels = tuple(output_channels)
    self._num_targets = len(self._target_names)

    mj_model = env.sim.mj_model

    # Model joint names carry the entity prefix (e.g. "robot/RCY"); resolve each
    # target to its dof address and recover that prefix.
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
    self._env_origins_np = env.scene.env_origins.cpu().numpy()

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

    # Named routing (mc_mujoco parity: raw base state to "FloatingBase", IMU
    # readings to the other body sensors) needs the extended binding's
    # name-keyed setters; the singular fallback only reaches bodySensors[0].
    self._use_named_routing = metadata.has_named_setters and self._root_qpos_adr >= 0
    use_reset = use_controller_reset and metadata.has_reset

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

    self.layout = IoLayout(
      num_targets=len(self._target_names),
      named_routing=self._use_named_routing,
      has_floating_base_sensor="FloatingBase" in metadata.body_sensor_names,
      use_reset=use_reset,
      feed_accel_fallback=self._accel_adr >= 0,
      imu=tuple((n, g >= 0, a >= 0) for n, g, a in imu_sensors),
      wrenches=tuple(n for n, _, _ in wrench_sensors),
    )

    # Gather columns for a single fancy-indexed sensordata copy per step;
    # missing readings (adr < 0) keep their zeroed columns.
    src_cols: list[int] = []
    dst_cols: list[int] = []
    for i, (_, g_adr, a_adr) in enumerate(imu_sensors):
      off = self.layout.imu_off + 6 * i
      if g_adr >= 0:
        src_cols += [g_adr, g_adr + 1, g_adr + 2]
        dst_cols += [off, off + 1, off + 2]
      if a_adr >= 0:
        src_cols += [a_adr, a_adr + 1, a_adr + 2]
        dst_cols += [off + 3, off + 4, off + 5]
    for i, (_, f_adr, t_adr) in enumerate(wrench_sensors):
      off = self.layout.wrench_off + 6 * i
      src_cols += [f_adr, f_adr + 1, f_adr + 2, t_adr, t_adr + 1, t_adr + 2]
      dst_cols += [off, off + 1, off + 2, off + 3, off + 4, off + 5]
    self._sens_src_cols = np.array(src_cols, dtype=np.intp)
    self._sens_dst_cols = np.array(dst_cols, dtype=np.intp)

  def _sensor_adr(self, suffix: str) -> int:
    """sensordata offset of the model sensor whose name ends with ``suffix``."""
    mj_model = self._env.sim.mj_model
    for i in range(mj_model.nsensor):
      if mj_model.sensor(i).name.endswith(suffix):
        return int(mj_model.sensor(i).adr[0])
    return -1

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

  def fill_controller_input(self, in_np: np.ndarray) -> None:
    """Write every input column (encoders/velocities/torques, root, sensors)."""
    self._fill_joint_columns(in_np)
    self._fill_root_and_sensor_columns(in_np)

  def reset_controller_input(self, in_np: np.ndarray) -> None:
    """Write the encoder columns and the first 7 root columns (pos + quat wxyz,
    same prefix in both routing modes) that the hosts read on reset/init."""
    T = self.layout.num_targets
    ro = self.layout.root_off
    in_np[:, 0:T] = self._entity.data.joint_pos.cpu().numpy()[:, self._target_ids_np]
    if self._root_qpos_adr >= 0:
      adr = self._root_qpos_adr
      in_np[:, ro : ro + 7] = self._env.sim.data.qpos.cpu().numpy()[:, adr : adr + 7]
    else:
      in_np[:, ro : ro + 3] = self._entity.data.root_link_pos_w.cpu().numpy()
      in_np[:, ro + 3 : ro + 7] = self._entity.data.root_link_quat_w.cpu().numpy()
    in_np[:, ro : ro + 3] -= self._env_origins_np

  def read_controller_output(
    self, out_np: np.ndarray, env_indices: list[int]
  ) -> dict[str, torch.Tensor]:
    """Unpack the controller outputs for ``env_indices`` into sim-ready tensors.

    Slices the output block per channel (``output_channels`` order) and moves
    each to the sim device, ready to feed the interpolation buffers -- the
    mc_rtc -> sim counterpart of ``fill_controller_input``. The default dtype matches
    the action's interpolation buffers (both created without an explicit dtype).
    """
    T = self._num_targets
    rows = out_np[env_indices]
    dtype = torch.get_default_dtype()
    return {
      c: torch.tensor(rows[:, i * T : (i + 1) * T], dtype=dtype, device=self._device)
      for i, c in enumerate(self._output_channels)
    }

  def _fill_joint_columns(self, in_np: np.ndarray) -> None:
    """Write encoder/velocity/torque columns of the input block (all envs)."""
    T = self.layout.num_targets
    current_pos = self._entity.data.joint_pos.cpu().numpy()
    current_vel = self._entity.data.joint_vel.cpu().numpy()
    in_np[:, 0:T] = current_pos[:, self._target_ids_np]
    in_np[:, T : 2 * T] = current_vel[:, self._target_ids_np]
    in_np[:, 2 * T : 3 * T] = self._env.sim.data.qfrc_actuator.cpu().numpy()[
      :, self._target_dof_adr
    ]

  def _fill_root_and_sensor_columns(self, in_np: np.ndarray) -> None:
    """Write the root block and the IMU/wrench columns of the input block."""
    ro = self.layout.root_off

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
