"""Model-sensor plumbing and the controller-side reads every term shares."""

from __future__ import annotations

from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import mujoco
import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase
from mc_mjlab.actions.walking_reference_action import WalkingReferenceMixin
from mc_mjlab.bridge.controller_datastore import (
  CONTROL_COM,
  CONTROL_COM_VEL,
  PLANNED_ZMP,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.manager_base import ManagerTermBaseCfg


def _residual_term(env: ManagerBasedRlEnv, action_name: str) -> McRtcResidualActionBase:
  term = env.action_manager.get_term(action_name)
  if not isinstance(term, McRtcResidualActionBase):
    raise TypeError(
      f"action term {action_name!r} is expected to be an mc_rtc residual "
      f"action, got {type(term).__name__}"
    )
  return term


def _walking_term(env: ManagerBasedRlEnv, action_name: str) -> WalkingReferenceMixin:
  term = env.action_manager.get_term(action_name)
  if not isinstance(term, WalkingReferenceMixin):
    raise TypeError(
      f"action term {action_name!r} does not drive the walking reference, "
      f"got {type(term).__name__}"
    )
  return term


def _restrict(term: McRtcResidualActionBase, values: torch.Tensor) -> torch.Tensor:
  """Keep only the columns carrying the residual (see ``residual_ids``)."""
  ids = term.residual_ids
  return values if ids is None else values[:, ids]


# mc_mujoco's "<name>_fsensor"/"_tsensor" pair; what the stabilizer sees too.
GROUND_CONTACT_SENSORS = ("LeftFootForceSensor", "RightFootForceSensor")

#: mc_rtc's own gravity constant, matching the host's ZMP formulas.
GRAVITY = 9.81

#: Floor under the CoM height, so a collapsed robot cannot divide omega by ~0.
MIN_COM_HEIGHT = 0.1


def _wrench_sensor(
  mj_model: mujoco.MjModel, suffix: str, sensor_type: int
) -> tuple[int, int]:
  """``(sensordata offset, site id)`` of the model sensor named ``*suffix``."""
  for i in range(mj_model.nsensor):
    sensor = mj_model.sensor(i)
    if sensor.name.endswith(suffix) and int(sensor.type[0]) == sensor_type:
      return int(sensor.adr[0]), int(sensor.objid[0])
  raise ValueError(
    f"the MuJoCo model has no force/torque sensor named '*{suffix}'; the ZMP "
    f"reward needs the mc_mujoco F/T sensor pair on every contact it sums."
  )


def _scalar_sensor_range(
  mj_model: mujoco.MjModel, name: str, dim: int, device: torch.device | str
) -> torch.Tensor:
  """``sensordata`` columns of the ``dim``-wide model sensor named exactly ``name``."""
  # Entity-prefixed in the compiled model ("robot/root_angmom"), bare in the spec.
  for i in range(mj_model.nsensor):
    sensor_name = mj_model.sensor(i).name
    if sensor_name == name or sensor_name.endswith(f"/{name}"):
      adr = int(mj_model.sensor(i).adr[0])
      return torch.arange(adr, adr + dim, device=device, dtype=torch.long)
  raise ValueError(
    f"the MuJoCo model has no sensor named {name!r}; it is added by "
    f"`robots/sensors.add_locomotion_sensors`."
  )


class _ZmpSensors:
  """Sensor plumbing for the measured centre of pressure, resolved once."""

  def __init__(
    self, env: ManagerBasedRlEnv, sensor_names: tuple[str, ...], asset_name: str
  ) -> None:
    mj_model = env.sim.mj_model
    force_cols: list[int] = []
    torque_cols: list[int] = []
    site_ids: list[int] = []
    for name in sensor_names:
      f_adr, site_id = _wrench_sensor(
        mj_model, f"{name}_fsensor", mujoco.mjtSensor.mjSENS_FORCE
      )
      t_adr, _ = _wrench_sensor(
        mj_model, f"{name}_tsensor", mujoco.mjtSensor.mjSENS_TORQUE
      )
      force_cols += [f_adr, f_adr + 1, f_adr + 2]
      torque_cols += [t_adr, t_adr + 1, t_adr + 2]
      site_ids.append(site_id)

    self.force_cols = torch.tensor(force_cols, device=env.device, dtype=torch.long)
    self.torque_cols = torch.tensor(torque_cols, device=env.device, dtype=torch.long)
    self.site_ids = torch.tensor(site_ids, device=env.device, dtype=torch.long)
    self.num_sensors = len(site_ids)
    self.root_body_id = env.scene[asset_name].indexing.root_body_id

    self._cache_key: tuple[object, ...] | None = None
    self._cache: tuple[torch.Tensor, torch.Tensor] | None = None

  def normal_forces(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Vertical contact force under each sensor, ``(num_envs, num_sensors)``."""
    num_envs, k = env.num_envs, self.num_sensors
    data = env.sim.data
    rot = data.site_xmat[:, self.site_ids].reshape(num_envs, k, 3, 3)
    force_s = data.sensordata[:, self.force_cols].reshape(num_envs, k, 3, 1)
    return -(rot @ force_s).squeeze(-1)[:, :, 2]

  def measured_offset(
    self,
    env: ManagerBasedRlEnv,
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(CoM-to-ZMP offset xy, vertical contact force)``, both ``(num_envs, ...)``."""
    # Memoised per step; five terms now read this. docs/reward-shaping.md#_zmpsensors
    key = (env.common_step_counter, min_normal_force, plane_height)
    if self._cache_key == key and self._cache is not None:
      return self._cache
    num_envs, k = env.num_envs, self.num_sensors
    data = env.sim.data

    rot = data.site_xmat[:, self.site_ids].reshape(num_envs, k, 3, 3)
    site_pos = data.site_xpos[:, self.site_ids].reshape(num_envs, k, 3)
    sensordata = data.sensordata
    force_s = sensordata[:, self.force_cols].reshape(num_envs, k, 3, 1)
    torque_s = sensordata[:, self.torque_cols].reshape(num_envs, k, 3, 1)

    force_w = -(rot @ force_s).squeeze(-1)
    torque_w = -(rot @ torque_s).squeeze(-1)

    com = data.subtree_com[:, self.root_body_id]
    lever = site_pos - com.unsqueeze(1)
    force = force_w.sum(dim=1)
    moment = (torque_w + torch.cross(lever, force_w, dim=-1)).sum(dim=1)

    # Ground plane, expressed from the CoM: mc_rbdyn::zmp with n = +z.
    normal_force = force[:, 2]
    height = plane_height - com[:, 2]
    safe_force = normal_force.clamp(min=min_normal_force)
    measured = torch.stack(
      (
        (height * force[:, 0] - moment[:, 1]) / safe_force,
        (moment[:, 0] + height * force[:, 1]) / safe_force,
      ),
      dim=-1,
    )
    self._cache_key, self._cache = key, (measured, normal_force)
    return measured, normal_force

  def offset_error(
    self,
    env: ManagerBasedRlEnv,
    action_name: str,
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(distance from the planned ZMP in metres, vertical contact force)``."""
    measured, normal_force = self.measured_offset(env, min_normal_force, plane_height)
    error = torch.linalg.vector_norm(
      measured - planned_zmp_offset(env, action_name), dim=1
    )
    return error, normal_force

  def dcm_offset(
    self,
    env: ManagerBasedRlEnv,
    action_name: str = "mc_rtc_residual",
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(distance from the *commanded* divergent-component offset, vertical force)``."""
    # LIPM: d(xi)/dt = omega * (xi - CoP), and walking at v needs xi - CoP = v/omega,
    # so the offset is scored against the commanded one, never against zero.
    measured, normal_force = self.measured_offset(env, min_normal_force, plane_height)
    data = env.sim.data
    com = data.subtree_com[:, self.root_body_id]
    com_vel = data.subtree_linvel[:, self.root_body_id]
    commanded = _residual_term(env, action_name).datastore_vector_output(
      CONTROL_COM_VEL
    )
    omega = torch.sqrt(GRAVITY / com[:, 2].clamp(min=MIN_COM_HEIGHT)).unsqueeze(-1)
    offset = (com_vel[:, :2] - commanded[:, :2]) / omega - measured
    return torch.linalg.vector_norm(offset, dim=1), normal_force


#: Per-env ``_ZmpSensors``, keyed weakly so they die with their env.
_ZMP_SENSOR_CACHE: WeakKeyDictionary[
  ManagerBasedRlEnv, dict[tuple[tuple[str, ...], str], _ZmpSensors]
] = WeakKeyDictionary()


def _zmp_sensors(
  env: ManagerBasedRlEnv, sensor_names: tuple[str, ...], asset_name: str
) -> _ZmpSensors:
  """The one :class:`_ZmpSensors` for this env and sensor set."""
  cache = _ZMP_SENSOR_CACHE.setdefault(env, {})
  key = (tuple(sensor_names), asset_name)
  sensors = cache.get(key)
  if sensors is None:
    sensors = cache[key] = _ZmpSensors(env, sensor_names, asset_name)
  return sensors


def planned_zmp_offset(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The controller's own CoM-to-ZMP offset, the target side of the comparison."""
  term = _residual_term(env, action_name)
  return (
    term.datastore_vector_output(PLANNED_ZMP)
    - term.datastore_vector_output(CONTROL_COM)
  )[:, :2]


def foot_load_share(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...] = GROUND_CONTACT_SENSORS,
  asset_name: str = "robot",
  min_normal_force: float = 20.0,
) -> torch.Tensor:
  """Each foot's share of the vertical contact force: the support state, in 0..1."""
  forces = _zmp_sensors(env, sensor_names, asset_name).normal_forces(env).clamp(min=0.0)
  return forces / forces.sum(dim=1, keepdim=True).clamp(min=min_normal_force)


class gait_phase:
  """``(cos, sin)`` of gait phase, inferred from the foot-load phase plane."""

  def __init__(self, cfg: ManagerTermBaseCfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = _zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )
    self._prev = torch.zeros(env.num_envs, device=env.device)
    self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._step = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Forget the phase derivative across episode boundaries."""
    ids = slice(None) if env_ids is None else env_ids
    self._initialized[ids] = False
    self._step[ids] = -1

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    rate_ref: float = 1.0,
    min_normal_force: float = 20.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg  # Resolved at init.
    forces = self._sensors.normal_forces(env).clamp(min=0.0)
    total = forces.sum(dim=1).clamp(min=min_normal_force)
    load = (forces[:, 0] - forces[:, 1]) / total
    # One read per step, however many terms ask: a second call in the same step
    # would difference against itself and report a zero rate.
    fresh = self._step != env.common_step_counter
    valid = fresh & self._initialized
    rate = torch.where(valid, (load - self._prev) / env.step_dt, torch.zeros_like(load))
    self._prev = torch.where(fresh, load, self._prev)
    self._step = torch.where(fresh, env.common_step_counter, self._step)
    self._initialized |= fresh
    plane = torch.stack((load, rate / rate_ref), dim=-1)
    return plane / torch.linalg.vector_norm(plane, dim=-1, keepdim=True).clamp(min=1e-6)


def measured_zmp_offset(
  env: ManagerBasedRlEnv,
  sensor_names: tuple[str, ...] = GROUND_CONTACT_SENSORS,
  asset_name: str = "robot",
) -> torch.Tensor:
  """The *measured* CoM-to-CoP offset, free of the observer drift the actor sees."""
  measured, _ = _zmp_sensors(env, sensor_names, asset_name).measured_offset(env)
  return measured


def encoder_bias(env: ManagerBasedRlEnv, asset_name: str = "robot") -> torch.Tensor:
  """The per-joint encoder bias itself, which the actor can only suffer."""
  return env.scene[asset_name].data.encoder_bias
