"""Simulation state routing into the native controller layout."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_from_angle_axis, quat_mul

import mc_rtc_interface as native
from mc_mjlab.robots import robot_module as robots

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_TRIPLE = 3
_SENSOR_STRIDE = 6
_FLOATING_BASE = "FloatingBase"
_BODY_SENSOR_CHANNELS = (("_gyro", "imu_gyro"), ("_accelerometer", "imu_accelerometer"))
_FORCE_SENSOR_CHANNELS = ("_fsensor", "_tsensor")


def _f64(values: torch.Tensor) -> torch.Tensor:
  """Widen to the block's dtype; an indexed write will not cast for us."""
  return values.to(torch.float64)


def _text(name: object) -> str:
  """Decode an mc_rbdyn name that may arrive as bytes."""
  return name.decode() if isinstance(name, (bytes, bytearray)) else str(name)


def _quat_from_rotvec(rotvec: torch.Tensor) -> torch.Tensor:
  """Unit wxyz quaternion from a rotation vector; a zero vector gives identity."""
  rotvec = _f64(rotvec)
  return quat_from_angle_axis(torch.linalg.vector_norm(rotvec, dim=1), rotvec)


def _compose_small_rotation(quat: torch.Tensor, rotvec: torch.Tensor) -> torch.Tensor:
  """Rotate `quat` (wxyz) by a body-frame rotation vector; adding would break it."""
  # Right-multiply: the offset is body-frame. mjlab's quat_box_plus is left-handed.
  return quat_mul(quat, _quat_from_rotvec(rotvec))


def _rotate_by_rotvec(vec: torch.Tensor, rotvec: torch.Tensor) -> torch.Tensor:
  """Rotate row vectors by a per-row rotation vector."""
  return quat_apply(_quat_from_rotvec(rotvec), vec)


class SimControllerBridge:
  """Resolve reference-order joints and named sensors once, then transfer batches."""

  def __init__(
    self,
    env: ManagerBasedRlEnv,
    entity: Entity,
    target_names: list[str],
    target_ids: torch.Tensor,
    robot_name: str,
    channels: tuple[str, ...],
    entity_name: str,
  ) -> None:
    self._env = env
    self._entity = entity
    self._device = target_ids.device
    self._output_channels = channels
    self._prefix = entity_name + "/"
    self.layout = native.IoLayout()

    self._resolve_joint_columns(robot_name, entity, target_names)
    self._resolve_root_joint()
    self._resolve_sensor_routes(robot_name)
    self._clear_state_feedback_offsets()
    self._alloc_device_buffers(env.num_envs)

  # Simulation to controller.

  def fill_controller_input(self, rows: np.ndarray) -> None:
    """Write biased encoders, measured effort, local root pose and raw sensors."""
    block = self._input_block
    self._fill_joint_columns(block)
    self._fill_root_columns(block)
    self._fill_sensor_columns(block)
    self._apply_state_feedback_offsets(block)

    # Last: every step above reads the root rotation as mjlab wxyz, native wants xyzw.
    ro = self.layout.input.root_offset()
    block[:, ro + 3 : ro + 7] = block[:, self._quat_xyzw_t]

    self._host_view(rows, "input")[:, : self._input_width].copy_(block)

  def _fill_joint_columns(self, block: torch.Tensor) -> None:
    """Scatter encoders, velocities and measured effort into reference order."""
    layout = self.layout.input
    count = len(layout.joint_order)
    entity = self._entity.data

    # The reference-order stance was written once; only simulated slots vary.
    block[:, layout.q_offset() + self._ref_cols_t] = _f64(
      entity.joint_pos_biased[:, self._sim_cols_t]
    )
    for offset, data in (
      (layout.qd_offset(), entity.joint_vel[:, self._sim_cols_t]),
      (layout.tau_offset(), self._env.sim.data.qfrc_actuator[:, self._dof_cols_t]),
    ):
      block[:, offset : offset + count] = 0.0
      block[:, offset + self._ref_cols_t] = _f64(data)

    for offset, feedback in (
      (layout.q_offset(), self._feedback_offset),
      (layout.qd_offset(), self._joint_velocity_offset),
    ):
      if feedback is not None:
        block[:, offset + self._target_cols_t] += _f64(feedback)

  def _fill_root_columns(self, block: torch.Tensor) -> None:
    """Write the root pose and linear velocity, local to this env's origin."""
    layout = self.layout.input
    ro = layout.root_offset()

    if self._root_qpos_adr >= 0:
      qa, da = self._root_qpos_adr, self._root_dof_adr
      block[:, ro : ro + 7] = self._env.sim.data.qpos[:, qa : qa + 7]
      block[:, ro + 7 : ro + 10] = self._env.sim.data.qvel[:, da : da + 3]
    else:
      data = self._entity.data
      block[:, ro : ro + 3] = data.root_link_pos_w
      block[:, ro + 3 : ro + 7] = data.root_link_quat_w
      block[:, ro + 7 : ro + 10] = data.root_link_lin_vel_w

    block[:, ro : ro + 3] -= self._env.scene.env_origins

  def _fill_sensor_columns(self, block: torch.Tensor) -> None:
    """Scatter the routed sensor triples, synthesizing FloatingBase from the root."""
    layout = self.layout.input

    block[:, layout.body_sensors_offset() : self._input_width] = 0.0
    if self._sens_src_cols:
      block[:, self._sens_dst_t] = _f64(
        self._env.sim.data.sensordata[:, self._sens_src_t]
      )

    if _FLOATING_BASE in layout.body_sensors and self._root_dof_adr >= 0:
      index = layout.body_sensors.index(_FLOATING_BASE)
      off = layout.body_sensors_offset() + _SENSOR_STRIDE * index
      da = self._root_dof_adr
      block[:, off : off + 3] = self._env.sim.data.qvel[:, da + 3 : da + 6]
      block[:, off + 3 : off + 6] = self._env.sim.data.qacc[:, da : da + 3]

  def _apply_state_feedback_offsets(self, block: torch.Tensor) -> None:
    """Apply feedback while the root quaternion is still wxyz."""
    layout = self.layout.input
    ro = layout.root_offset()

    if self._wrench_offset is not None:
      off = layout.force_sensors_offset()
      width = _SENSOR_STRIDE * len(layout.force_sensors)
      block[:, off : off + width] += self._wrench_offset

    if self._root_translation_offset is not None:
      block[:, ro : ro + 3] += self._root_translation_offset

    if self._root_rotation_offset is None:
      return

    # The body sensors read in the root frame, so they turn the opposite way.
    for i, name in enumerate(layout.body_sensors):
      if name == _FLOATING_BASE:
        continue
      for j in (0, _TRIPLE):
        off = layout.body_sensors_offset() + _SENSOR_STRIDE * i + j
        block[:, off : off + _TRIPLE] = _rotate_by_rotvec(
          block[:, off : off + _TRIPLE], -self._root_rotation_offset
        )
    block[:, ro + 3 : ro + 7] = _compose_small_rotation(
      block[:, ro + 3 : ro + 7], self._root_rotation_offset
    )

  # Controller to simulation.

  def upload_controller_output(self, rows: np.ndarray) -> torch.Tensor:
    """Copy the whole output block to the device; every gather then runs there."""
    if self._output_block is None or self._output_block.shape != rows.shape:
      self._output_block = torch.empty(
        rows.shape, dtype=torch.float64, device=self._device
      )
    self._output_block.copy_(self._host_view(rows, "output"))
    return self._output_block

  def read_controller_output(self, block: torch.Tensor) -> dict[str, torch.Tensor]:
    """Gather action joints and expose native qd as the public alpha channel."""
    layout = self.layout.output
    offsets = {
      "q": layout.q_offset(),
      "alpha": layout.qd_offset(),
      "tau": layout.tau_offset(),
    }
    dtype = torch.get_default_dtype()
    return {
      c: block[:, offsets[c] + self._target_cols_t].to(dtype)
      for c in self._output_channels
    }

  # State feedback: what the controller is told, not what the simulation measured.

  def set_feedback_offset(self, offset: torch.Tensor | None) -> None:
    """Bias the joint positions the controller sees, in target order."""
    self._feedback_offset = offset

  def set_joint_velocity_offset(self, offset: torch.Tensor | None) -> None:
    """Bias the joint velocities the controller sees, in target order."""
    self._joint_velocity_offset = offset

  def set_wrench_offset(self, offset: torch.Tensor | None) -> None:
    """Bias every force-sensor wrench the controller sees."""
    self._wrench_offset = offset

  def set_root_pose_offset(
    self, translation: torch.Tensor | None, rotation: torch.Tensor | None
  ) -> None:
    """Bias the root pose the controller sees, rotation as a body-frame rotvec."""
    self._root_translation_offset = translation
    self._root_rotation_offset = rotation

  def release_views(self) -> None:
    """Drop the shared-block views so the blocks can be unlinked."""
    self._views = {}

  # Setup: resolve every index once, so a control period is pure gather/scatter.

  def _resolve_joint_columns(
    self, robot_name: str, entity: Entity, target_names: list[str]
  ) -> None:
    """Map simulated joints onto reference order; unsimulated slots keep the stance."""
    order = robots.get_ref_joint_order(robot_name)
    self.layout.set_joint_order(list(order))
    self.target_columns = np.array([order.index(n) for n in target_names])

    stance = robots.get_default_joint_positions(robot_name, drop_zeros=False)
    self._stance = np.array([stance.get(n, 0.0) for n in order])

    names = list(entity.joint_names)
    self._sim_columns = np.array([i for i, n in enumerate(names) if n in order])
    self._ref_columns = np.array([order.index(names[i]) for i in self._sim_columns])

    model = self._env.sim.mj_model
    self._dof_columns = [
      int(model.joint(self._prefix + names[i]).dofadr[0]) for i in self._sim_columns
    ]

  def _resolve_root_joint(self) -> None:
    """Locate this entity's free joint; -1 falls back to the entity's own root data."""
    model = self._env.sim.mj_model
    self._root_qpos_adr = self._root_dof_adr = -1

    for i in range(model.njnt):
      joint = model.joint(i)
      if not joint.name.startswith(self._prefix):
        continue
      if joint.type[0] == mujoco.mjtJoint.mjJNT_FREE:
        self._root_qpos_adr = int(model.jnt_qposadr[i])
        self._root_dof_adr = int(model.jnt_dofadr[i])
        break

  def _resolve_sensor_routes(self, robot_name: str) -> None:
    """Route each named MuJoCo sensor triple to its slot in the sensor block."""
    module = robots.get_robot_module(robot_name)
    # mc_rbdyn sensor names come back as bytes; the native setter wants str.
    self.layout.input.body_sensors = [_text(s.name()) for s in module.bodySensors()]
    self.layout.input.force_sensors = [_text(s.name()) for s in module.forceSensors()]

    layout = self.layout.input
    self._sens_src_cols = []
    self._sens_dst_cols = []

    for i, name in enumerate(layout.body_sensors):
      # FloatingBase has no MuJoCo sensor; it is synthesized from qvel/qacc instead.
      if name == _FLOATING_BASE:
        continue
      for j, (suffix, fallback) in enumerate(_BODY_SENSOR_CHANNELS):
        adr = self._sensor_adr(name + suffix)
        if adr < 0:
          adr = self._sensor_adr(fallback)
        self._route(
          adr, layout.body_sensors_offset() + _SENSOR_STRIDE * i + _TRIPLE * j
        )

    for i, name in enumerate(layout.force_sensors):
      for j, suffix in enumerate(_FORCE_SENSOR_CHANNELS):
        self._route(
          self._sensor_adr(name + suffix),
          layout.force_sensors_offset() + _SENSOR_STRIDE * i + _TRIPLE * j,
        )

  def _sensor_adr(self, suffix: str) -> int:
    """Find a sensor within this entity's namespace, or -1 when it has none."""
    try:
      return int(self._env.sim.mj_model.sensor(self._prefix + suffix).adr[0])
    except KeyError:
      return -1

  def _route(self, source: int, destination: int) -> None:
    """Leave absent sensor readings zero and route available triples."""
    if source >= 0:
      self._sens_src_cols.extend(range(source, source + _TRIPLE))
      self._sens_dst_cols.extend(range(destination, destination + _TRIPLE))

  def _clear_state_feedback_offsets(self) -> None:
    """Start every modality neutral; the action sets only the ones it owns."""
    self._feedback_offset = None
    self._joint_velocity_offset = None
    self._root_translation_offset = None
    self._root_rotation_offset = None
    self._wrench_offset = None

  def _alloc_device_buffers(self, num_envs: int) -> None:
    """Stage whole I/O blocks on the device so each period costs one copy."""
    layout = self.layout.input
    index = {"dtype": torch.long, "device": self._device}

    self._ref_cols_t = torch.as_tensor(self._ref_columns, **index)
    self._sim_cols_t = torch.as_tensor(self._sim_columns, **index)
    self._dof_cols_t = torch.as_tensor(self._dof_columns, **index)
    self._sens_src_t = torch.as_tensor(self._sens_src_cols, **index)
    self._sens_dst_t = torch.as_tensor(self._sens_dst_cols, **index)
    self._target_cols_t = torch.as_tensor(self.target_columns, **index)

    ro = layout.root_offset()
    self._quat_xyzw_t = torch.as_tensor([ro + 4, ro + 5, ro + 6, ro + 3], **index)

    # The simulation owns the contiguous prefix; datastore, log and reset columns
    # are written by the host and must survive the copy.
    self._input_width = layout.datastore_scalar_offset()
    self._input_block = torch.zeros(
      num_envs, self._input_width, dtype=torch.float64, device=self._device
    )
    self._input_block[:, layout.q_offset() : layout.q_offset() + len(self._stance)] = (
      torch.as_tensor(self._stance, dtype=torch.float64, device=self._device)
    )

    # The task declares these columns and the action sets them on the layout after
    # this constructor, so the output width is only final at the first upload.
    self._output_block = None
    self._views: dict[str, tuple[torch.Tensor, np.ndarray]] = {}

  # Shared memory: the host owns the numpy rows, this class owns the device copy.

  def _host_view(self, rows: np.ndarray, key: str) -> torch.Tensor:
    """Cache the torch view of a shared block; from_numpy must not run per period."""
    cached = self._views.get(key)
    if cached is None or cached[1] is not rows:
      view = torch.from_numpy(rows)
      self._views[key] = (view, rows)
      return view
    return cached[0]
