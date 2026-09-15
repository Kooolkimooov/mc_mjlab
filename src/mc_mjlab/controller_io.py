"""Simulation state routing into the native controller layout."""

from __future__ import annotations

import mujoco
import numpy as np
import torch

import mc_rtc_interface as native
from mc_mjlab.controller_objects import ControllerObjects
from mc_mjlab.robots import mc_rtc_robot_configuration as robots


def _f64(values: torch.Tensor) -> torch.Tensor:
  """Widen to the block's dtype; an indexed write will not cast for us."""
  return values.to(torch.float64)


def _text(name) -> str:
  """Decode an mc_rbdyn name that may arrive as bytes."""
  return name.decode() if isinstance(name, (bytes, bytearray)) else str(name)


class ControllerIoBinding:
  """Resolve reference-order joints and named sensors once, then transfer batches."""

  def __init__(
    self,
    env,
    entity,
    target_names,
    target_ids,
    robot_name,
    channels,
    entity_name,
    controller_objects=None,
  ):
    self._env = env
    self._entity = entity
    self._device = target_ids.device
    self._output_channels = channels
    self.layout = native.IoLayout()
    order = robots.get_ref_joint_order(robot_name)
    self.layout.set_joint_order(list(order))
    self.target_columns = np.array([order.index(n) for n in target_names])
    stance = robots.get_default_joint_positions(robot_name, drop_zeros=False)
    self._stance = np.array([stance.get(n, 0.0) for n in order])
    entity_names = list(entity.joint_names)
    self._sim_columns = np.array([i for i, n in enumerate(entity_names) if n in order])
    self._ref_columns = np.array(
      [order.index(entity_names[i]) for i in self._sim_columns]
    )
    model = env.sim.mj_model
    prefix = entity_name + "/"
    self._prefix = prefix
    self._dof_columns = []
    for i in self._sim_columns:
      name = entity_names[i]
      joint = model.joint(prefix + name)
      self._dof_columns.append(int(joint.dofadr[0]))
    self._root_qpos_adr = self._root_dof_adr = -1
    for i in range(model.njnt):
      joint = model.joint(i)
      if joint.name.startswith(prefix) and joint.type[0] == mujoco.mjtJoint.mjJNT_FREE:
        self._root_qpos_adr = int(model.jnt_qposadr[i])
        self._root_dof_adr = int(model.jnt_dofadr[i])
        break
    module = robots.get_robot_module(robot_name)
    # mc_rbdyn sensor names come back as bytes; the native setter wants str.
    self.layout.input.body_sensors = [_text(s.name()) for s in module.bodySensors()]
    self.layout.input.force_sensors = [_text(s.name()) for s in module.forceSensors()]
    self._objects = ControllerObjects(env, controller_objects or {})
    self.layout.input.objects = list(controller_objects or {})
    self._sens_src_cols = []
    self._sens_dst_cols = []
    for i, name in enumerate(self.layout.input.body_sensors):
      if name == "FloatingBase":
        continue
      for j, (suffix, fallback) in enumerate(
        (("_gyro", "imu_gyro"), ("_accelerometer", "imu_accelerometer"))
      ):
        adr = self._sensor_adr(name + suffix)
        if adr < 0:
          adr = self._sensor_adr(fallback)
        self._route(adr, self.layout.input.body_sensors_offset() + 6 * i + 3 * j)
    for i, name in enumerate(self.layout.input.force_sensors):
      for j, suffix in enumerate(("_fsensor", "_tsensor")):
        self._route(
          self._sensor_adr(name + suffix),
          self.layout.input.force_sensors_offset() + 6 * i + 3 * j,
        )
    self._feedback_offset = None
    self._joint_velocity_offset = None
    self._root_translation_offset = None
    self._root_rotation_offset = None
    self._wrench_offset = None
    self._alloc_device_buffers(env.num_envs)

  def _sensor_adr(self, suffix):
    """Find a sensor within this entity's namespace."""
    model = self._env.sim.mj_model
    for i in range(model.nsensor):
      sensor = model.sensor(i)
      if sensor.name == self._prefix + suffix:
        return int(sensor.adr[0])
    return -1

  def _route(self, source, destination):
    """Leave absent sensor readings zero and route available triples."""
    if source >= 0:
      self._sens_src_cols.extend(range(source, source + 3))
      self._sens_dst_cols.extend(range(destination, destination + 3))

  def _alloc_device_buffers(self, num_envs):
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
    # The output width is only final once the action has declared its datastore
    # getters, which happens after this constructor: allocate on first upload.
    self._output_block = None
    self._views: dict[str, tuple[torch.Tensor, np.ndarray]] = {}

  def _host_view(self, rows, key):
    """Cache the torch view of a shared block; from_numpy must not run per period."""
    view, base = self._views.get(key, (None, None))
    if base is not rows:
      view = torch.from_numpy(rows)
      self._views[key] = (view, rows)
    return view

  def release_views(self) -> None:
    """Drop the shared-block views so the blocks can be unlinked."""
    self._views = {}

  def fill_controller_input(self, rows):
    """Write biased encoders, measured effort, local root pose and raw sensors."""
    layout = self.layout.input
    block = self._input_block
    count = len(layout.joint_order)
    # The reference-order stance was written once; only simulated slots vary.
    entity = self._entity.data
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
    self._fill_root_and_sensor_columns(block)
    self._objects.fill(block, layout.objects_offset())
    self._host_view(rows, "input")[:, : self._input_width].copy_(block)

  def _fill_root_and_sensor_columns(self, block):
    """Keep root xyzw separate from the six-value body-sensor slots."""
    layout = self.layout.input
    ro = layout.root_offset()
    data = self._entity.data
    if self._root_qpos_adr >= 0:
      qa, da = self._root_qpos_adr, self._root_dof_adr
      block[:, ro : ro + 7] = self._env.sim.data.qpos[:, qa : qa + 7]
      block[:, ro + 7 : ro + 10] = self._env.sim.data.qvel[:, da : da + 3]
    else:
      block[:, ro : ro + 3] = data.root_link_pos_w
      block[:, ro + 3 : ro + 7] = data.root_link_quat_w
      block[:, ro + 7 : ro + 10] = data.root_link_lin_vel_w
    block[:, ro : ro + 3] -= self._env.scene.env_origins
    block[:, layout.body_sensors_offset() : self._input_width] = 0.0
    if self._sens_src_cols:
      block[:, self._sens_dst_t] = _f64(
        self._env.sim.data.sensordata[:, self._sens_src_t]
      )
    if "FloatingBase" in layout.body_sensors and self._root_dof_adr >= 0:
      off = layout.body_sensors_offset() + 6 * layout.body_sensors.index("FloatingBase")
      da = self._root_dof_adr
      block[:, off : off + 3] = self._env.sim.data.qvel[:, da + 3 : da + 6]
      block[:, off + 3 : off + 6] = self._env.sim.data.qacc[:, da : da + 3]
    self._apply_state_feedback_offsets(block)
    block[:, ro + 3 : ro + 7] = block[:, self._quat_xyzw_t]

  def _apply_state_feedback_offsets(self, block):
    """Apply feedback before converting the root quaternion to native xyzw."""
    layout = self.layout.input
    ro = layout.root_offset()
    if self._wrench_offset is not None:
      off = layout.force_sensors_offset()
      block[:, off : off + 6 * len(layout.force_sensors)] += self._wrench_offset
    if self._root_translation_offset is not None:
      block[:, ro : ro + 3] += self._root_translation_offset
    if self._root_rotation_offset is not None:
      for i, name in enumerate(layout.body_sensors):
        if name == "FloatingBase":
          continue
        for j in (0, 3):
          off = layout.body_sensors_offset() + 6 * i + j
          block[:, off : off + 3] = self._rotate_by_rotvec(
            block[:, off : off + 3], -self._root_rotation_offset
          )
      block[:, ro + 3 : ro + 7] = self._compose_small_rotation(
        block[:, ro + 3 : ro + 7], self._root_rotation_offset
      )

  def upload_controller_output(self, rows) -> torch.Tensor:
    """Copy the whole output block to the device; every gather then runs there."""
    if self._output_block is None or self._output_block.shape != rows.shape:
      self._output_block = torch.empty(
        rows.shape, dtype=torch.float64, device=self._device
      )
    self._output_block.copy_(self._host_view(rows, "output"))
    return self._output_block

  def read_controller_output(self, block) -> dict[str, torch.Tensor]:
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

  @staticmethod
  def _compose_small_rotation(quat: torch.Tensor, rotvec: torch.Tensor) -> torch.Tensor:
    """Rotate `quat` (wxyz) by a body-frame rotation vector; adding would break it."""
    rotvec = _f64(rotvec)
    angle = torch.linalg.vector_norm(rotvec, dim=1, keepdim=True)
    half = 0.5 * angle
    # sinc-style guard: the axis is arbitrary at zero angle, the term vanishes.
    scale = torch.where(
      angle > 1.0e-9,
      torch.sin(half) / angle.clamp(min=1.0e-9),
      0.5 * torch.ones_like(angle),
    )
    dq = torch.cat([torch.cos(half), rotvec * scale], dim=1)
    w1, x1, y1, z1 = (quat[:, i] for i in range(4))
    w2, x2, y2, z2 = (dq[:, i] for i in range(4))
    out = torch.stack(
      [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
      ],
      dim=1,
    )
    return out / torch.linalg.vector_norm(out, dim=1, keepdim=True).clamp(min=1.0e-12)

  @staticmethod
  def _rotate_by_rotvec(vec: torch.Tensor, rotvec: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation of row vectors by a per-row rotation vector."""
    rotvec = _f64(rotvec)
    angle = torch.linalg.vector_norm(rotvec, dim=1, keepdim=True)
    axis = rotvec / angle.clamp(min=1.0e-12)
    cos, sin = torch.cos(angle), torch.sin(angle)
    cross = torch.linalg.cross(axis, vec, dim=1)
    dot = (axis * vec).sum(dim=1, keepdim=True)
    return vec * cos + cross * sin + axis * dot * (1.0 - cos)
