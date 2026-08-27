"""MDP terms that talk about the mc_rtc controller itself, not just robot state."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import mujoco
import torch
from mjlab.envs.mdp import events, terminations
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase
from mc_mjlab.robots import mc_rtc_robot_configuration as mc_rtc

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg


def _residual_term(env: ManagerBasedRlEnv, action_name: str) -> McRtcResidualActionBase:
  term = env.action_manager.get_term(action_name)
  if not isinstance(term, McRtcResidualActionBase):
    raise TypeError(
      f"action term {action_name!r} is expected to be an mc_rtc residual "
      f"action, got {type(term).__name__}"
    )
  return term


def controller_failed(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Terminate envs whose mc_rtc controller gave up."""
  return _residual_term(env, action_name).controller_failed


def controller_worker_failed(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """End envs whose controller *process* died -- as a truncation, not a failure."""
  # Correlated across a worker's envs and unrelated to the action, so it must be
  # configured `time_out=True`. docs/coupling.md#worker-failure-is-a-truncation
  return _residual_term(env, action_name).controller_worker_failed


def collapsed(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  limit_angle: float,
  asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
  """Root below ``minimum_height`` while still upright: a crouch, not a topple."""
  # Exclusive with `bad_orientation` on purpose, so the two counters partition the
  # balance failures; their union is what it always was.
  cfg = asset_cfg or SceneEntityCfg("robot")
  low = terminations.root_height_below_minimum(env, minimum_height, cfg)
  return low & ~terminations.bad_orientation(env, limit_angle, cfg)


def action_l2(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Squared magnitude of the normalized residual actually delivered."""
  action = _residual_term(env, action_name).executed_normalized_action
  return torch.sum(torch.square(action), dim=1)


def action_rate_l2(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Squared policy-step change in normalized residual actually delivered."""
  term = _residual_term(env, action_name)
  delta = term.executed_normalized_action - term.previous_executed_normalized_action
  return torch.sum(torch.square(delta), dim=1)


def requested_action_l2(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Squared normalized residual request while recovery authority is nonzero."""
  term = _residual_term(env, action_name)
  cost = torch.sum(torch.square(term.requested_normalized_action), dim=1)
  return cost * (term.last_gate > 0.0)


def requested_action_rate_l2(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Squared normalized request change across consecutive authorized steps."""
  term = _residual_term(env, action_name)
  delta = term.requested_normalized_action - term.previous_requested_normalized_action
  # Burst onset has no delivered predecessor; differencing against zero there
  # charges the onset step the magnitude cost twice. docs/reward-shaping.md
  paired = (term.last_gate > 0.0) & (term.previous_gate > 0.0)
  return torch.sum(torch.square(delta), dim=1) * paired


def _restrict(term: McRtcResidualActionBase, values: torch.Tensor) -> torch.Tensor:
  """Keep only the columns carrying the residual (see ``residual_ids``)."""
  ids = term.residual_ids
  return values if ids is None else values[:, ids]


def executed_action(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The physical residual actually delivered to the actuator target."""
  return _residual_term(env, action_name).executed_physical_action


def walking_reference_velocity(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Executed walking-reference delta in normalized action coordinates."""
  return _residual_term(env, action_name).walking_reference_normalized


def walking_reference_l2(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Squared normalized walking-reference delta delivered this step."""
  action = _residual_term(env, action_name).walking_reference_normalized
  return torch.sum(torch.square(action), dim=1)


def walking_reference_rate_l2(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Squared policy-step change in the normalized walking-reference delta."""
  term = _residual_term(env, action_name)
  delta = term.walking_reference_normalized - term.previous_walking_reference_normalized
  return torch.sum(torch.square(delta), dim=1)


def requested_normalized_action(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The bounded policy request before physical scaling."""
  return _residual_term(env, action_name).requested_normalized_action


def requested_physical_action(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The physical residual requested before gating and feasibility projection."""
  return _residual_term(env, action_name).requested_physical_action


def randomize_current_pd_gains(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  scale_range: tuple[float, float],
  asset_cfg: SceneEntityCfg | None = None,
  action_name: str = "mc_rtc_residual",
) -> None:
  """Scale the active reference PD gains independently per environment and joint."""
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  else:
    env_ids = env_ids.to(env.device)
  term = _residual_term(env, action_name)
  kp = getattr(term, "_kp", None)
  kd = getattr(term, "_kd", None)
  if kp is not None and kd is not None:
    scale = torch.empty(len(env_ids), kp.shape[1], device=env.device).uniform_(
      *scale_range
    )
    kp[env_ids] *= scale
    kd[env_ids] *= scale
    return
  asset = env.scene[(asset_cfg or SceneEntityCfg("robot")).name]
  for actuator in asset.actuators:
    stiffness = getattr(actuator, "stiffness", None)
    damping = getattr(actuator, "damping", None)
    if stiffness is None or damping is None:
      continue
    scale = torch.empty(
      len(env_ids), len(actuator.target_names), device=env.device
    ).uniform_(*scale_range)
    stiffness[env_ids] *= scale
    damping[env_ids] *= scale


def projection_fraction(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Fraction of residual joints changed by feasibility projection this step."""
  return _residual_term(env, action_name).projection_mask.float().mean(dim=1)


def near_bound_fraction(
  env: ManagerBasedRlEnv,
  action_name: str = "mc_rtc_residual",
  threshold: float = 0.99,
) -> torch.Tensor:
  """Fraction of normalized policy requests within ``1-threshold`` of a bound."""
  action = _residual_term(env, action_name).requested_normalized_action
  return (action.abs() >= threshold).float().mean(dim=1)


def gate_mean(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Recovery-conditioned residual authority in 0..1."""
  return _residual_term(env, action_name).last_gate


def detector_score(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Calibrated transparent recovery score before temporal filtering."""
  return _residual_term(env, action_name).detector_score


def recovery_dcm_error_vector(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Signed deployable DCM error used by the recovery detector."""
  return _residual_term(env, action_name).recovery_dcm_error_vector


def inactive_residual_violation(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Peak executed residual where authority is exactly zero."""
  term = _residual_term(env, action_name)
  inactive = term.last_gate == 0.0
  peak = term.executed_physical_action.abs().amax(dim=1)
  return peak * inactive


def controller_position_error(
  env: ManagerBasedRlEnv,
  action_name: str = "mc_rtc_residual",
  biased: bool = True,
) -> torch.Tensor:
  """Controller reference minus encoder-visible or privileged joint position."""
  term = _residual_term(env, action_name)
  asset = env.scene[term.cfg.entity_name]
  position = asset.data.joint_pos_biased if biased else asset.data.joint_pos
  error = term.controller_reference("q") - position[:, term.target_ids]
  return _restrict(term, error)


def controller_reference_velocity(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The controller's joint-velocity reference: where its gait is headed."""
  term = _residual_term(env, action_name)
  return _restrict(term, term.controller_reference("alpha"))


def controller_reference_position(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The controller's joint-position reference: the vector the residual is added to."""
  term = _residual_term(env, action_name)
  return _restrict(term, term.controller_reference("q"))


def controller_planned_zmp_offset(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The controller's planned CoM-to-ZMP offset: where it means to push."""
  return planned_zmp_offset(env, action_name)


def controller_planned_com_velocity(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The CoM velocity the controller's plan calls for."""
  return _residual_term(env, action_name).controller_vector("control_com_vel")


def controller_walking_reference_velocity(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Current ``(vx, vy, yaw_rate)`` command reported by the walking controller."""
  return _residual_term(env, action_name).controller_vector("walking_ref_vel")


def base_progress_tanh(
  env: ManagerBasedRlEnv, speed: float = 0.1, asset_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
  """Reward the robot actually travelling forward, saturating at ``speed``."""
  asset = env.scene[(asset_cfg or SceneEntityCfg("robot")).name]
  forward = asset.data.root_link_lin_vel_b[:, 0].clamp(min=0.0)
  return torch.tanh(forward / speed)


def controller_reference_motion(
  env: ManagerBasedRlEnv, scale: float = 1.0, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Reward the controller still generating a gait, bounded by tanh."""
  alpha = controller_reference_velocity(env, action_name)
  return torch.tanh(torch.linalg.vector_norm(alpha, dim=1) / scale)


# mc_mujoco's "<name>_fsensor"/"_tsensor" pair; what the stabilizer sees too.
GROUND_CONTACT_SENSORS = ("LeftFootForceSensor", "RightFootForceSensor")

#: mc_rtc's own gravity constant, matching the host's ZMP formulas.
GRAVITY = 9.81

#: Floor under the CoM height, so a collapsed robot cannot divide omega by ~0.
MIN_COM_HEIGHT = 0.1


def _wrench_sensor(mj_model, suffix: str, sensor_type: int) -> tuple[int, int]:
  """``(sensordata offset, site id)`` of the model sensor named ``*suffix``."""
  for i in range(mj_model.nsensor):
    sensor = mj_model.sensor(i)
    if sensor.name.endswith(suffix) and int(sensor.type[0]) == sensor_type:
      return int(sensor.adr[0]), int(sensor.objid[0])
  raise ValueError(
    f"the MuJoCo model has no force/torque sensor named '*{suffix}'; the ZMP "
    f"reward needs the mc_mujoco F/T sensor pair on every contact it sums."
  )


def _scalar_sensor_range(mj_model, name: str, dim: int, device) -> torch.Tensor:
  """``sensordata`` columns of the ``dim``-wide model sensor named exactly ``name``."""
  # Entity-prefixed in the compiled model ("robot/root_angmom"), bare in the spec.
  for i in range(mj_model.nsensor):
    sensor_name = mj_model.sensor(i).name
    if sensor_name == name or sensor_name.endswith(f"/{name}"):
      adr = int(mj_model.sensor(i).adr[0])
      return torch.arange(adr, adr + dim, device=device, dtype=torch.long)
  raise ValueError(
    f"the MuJoCo model has no sensor named {name!r}; it is added by "
    f"`robots/additional_sensors_configuration.add_locomotion_sensors`."
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
    commanded = _residual_term(env, action_name).controller_vector("control_com_vel")
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
    term.controller_vector("planned_zmp") - term.controller_vector("control_com")
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

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
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


class zmp_error:
  """Distance from the measured centre of pressure to the planned one, in metres."""

  # Read as `zmp_error / zmp_grounded`; alone it falls when the feet lift.

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = _zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    action_name: str = "mc_rtc_residual",
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg  # Resolved at init.
    error, normal_force = self._sensors.offset_error(
      env, action_name, min_normal_force, plane_height
    )
    return error * (normal_force >= min_normal_force)


class zmp_grounded:
  """Share of steps whose feet carry enough load for a centre of pressure."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = _zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    action_name: str = "mc_rtc_residual",
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg  # Resolved at init.
    _, normal_force = self._sensors.offset_error(
      env, action_name, min_normal_force, plane_height
    )
    return (normal_force >= min_normal_force).float()


class zmp_tracking:
  """Reward the measured ZMP sitting where the mc_rtc controller wants it."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = _zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    action_name: str = "mc_rtc_residual",
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg  # Resolved at init.
    error, normal_force = self._sensors.offset_error(
      env, action_name, min_normal_force, plane_height
    )
    return torch.exp(-torch.square(error / std)) * (normal_force >= min_normal_force)


class com_velocity_tracking:
  """Reward the robot's CoM moving the way the controller is commanding."""

  # Vertical scored apart from the horizontal pair, or crouch-collapse hides.

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self._root_body_id = env.scene[cfg.params["asset_cfg"].name].indexing.root_body_id

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    std_vertical: float,
    asset_cfg: SceneEntityCfg,
    action_name: str = "mc_rtc_residual",
  ) -> torch.Tensor:
    del asset_cfg  # Resolved at init.
    term = _residual_term(env, action_name)
    error = env.sim.data.subtree_linvel[:, self._root_body_id] - term.controller_vector(
      "control_com_vel"
    )
    horizontal = torch.linalg.vector_norm(error[:, :2], dim=1)
    vertical = error[:, 2].abs()
    return 0.5 * (
      torch.exp(-torch.square(horizontal / std))
      + torch.exp(-torch.square(vertical / std_vertical))
    )


class com_velocity_error:
  """Distance from the controller's commanded CoM velocity, m/s."""

  # The one term negative in every comparison: keep it as the canary for a policy
  # fighting the plan. docs/reward-shaping.md#com_velocity_error

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    self._root_body_id = env.scene[cfg.params["asset_cfg"].name].indexing.root_body_id

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    action_name: str = "mc_rtc_residual",
  ) -> torch.Tensor:
    del asset_cfg  # Resolved at init.
    term = _residual_term(env, action_name)
    error = env.sim.data.subtree_linvel[:, self._root_body_id] - term.controller_vector(
      "control_com_vel"
    )
    return torch.linalg.vector_norm(error, dim=1)


class dcm_error:
  """Distance from the divergent component of motion to the centre of pressure."""

  # Read as `dcm_error / zmp_grounded`, for the same reason `zmp_error` is.

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = _zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    action_name: str = "mc_rtc_residual",
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg  # Resolved at init.
    error, normal_force = self._sensors.dcm_offset(
      env, action_name, min_normal_force, plane_height
    )
    return error * (normal_force >= min_normal_force)


class dcm_stability:
  """Reward the robot not diverging, which is what mc_rtc's plan cannot buy itself."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = _zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    action_name: str = "mc_rtc_residual",
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg  # Resolved at init.
    error, normal_force = self._sensors.dcm_offset(
      env, action_name, min_normal_force, plane_height
    )
    return torch.exp(-torch.square(error / std)) * (normal_force >= min_normal_force)


class angular_momentum_l2:
  """Penalise centroidal angular momentum, which the stabilizer QP does not regulate."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    del cfg
    self._adr = _scalar_sensor_range(env.sim.mj_model, "root_angmom", 3, env.device)

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    momentum = env.sim.data.sensordata[:, self._adr]
    return torch.sum(torch.square(momentum), dim=1)


class foot_slip:
  """Penalise a loaded sole sliding, a contact violation the QP's model cannot see."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = _zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )
    self._adr = torch.cat(
      [
        _scalar_sensor_range(env.sim.mj_model, name, 3, env.device)
        for name in cfg.params["velocimeter_names"]
      ]
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    velocimeter_names: tuple[str, ...],
    min_normal_force: float = 20.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg, velocimeter_names  # Resolved at init.
    num_feet = self._adr.numel() // 3
    velocity = env.sim.data.sensordata[:, self._adr].reshape(-1, num_feet, 3)
    loaded = self._sensors.normal_forces(env) >= min_normal_force
    tangential = torch.sum(torch.square(velocity[:, :, :2]), dim=2)
    return torch.sum(tangential * loaded, dim=1)


class torque_margin:
  """Penalise peak joint torque past the robot's own hardware limit."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    term = _residual_term(env, cfg.params.get("action_name", "mc_rtc_residual"))
    ids = term.residual_ids
    cols = list(range(len(term.target_names))) if ids is None else ids.tolist()
    limits = mc_rtc.get_effort_limits(term.cfg.mc_rtc_robot_name)
    missing = [term.target_names[i] for i in cols if term.target_names[i] not in limits]
    if missing:
      raise KeyError(
        f"the mc_rtc RobotModule reports no torque limit for {missing}; "
        f"`torque_margin` cannot bound a joint it has no limit for."
      )
    self._cols = torch.tensor(cols, device=env.device, dtype=torch.long)
    self._limits = torch.tensor(
      [limits[term.target_names[i]] for i in cols], device=env.device
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    soft_ratio: float = 1.0,
    action_name: str = "mc_rtc_residual",
    warmup_steps: int = 25,
  ) -> torch.Tensor:
    term = _residual_term(env, action_name)
    # Fold in this step's own read: apply_actions trails sim.step by one substep,
    # so the accumulator alone misses the last of the decimation window.
    peak = torch.maximum(
      term.consume_torque_peak(),
      env.scene[term.cfg.entity_name].data.qfrc_actuator[:, term.target_ids].abs(),
    )[:, self._cols]
    over = torch.relu(peak / self._limits - soft_ratio)
    # The reset teleport drives a substep transient of 22x the limit that no policy
    # -rate sample sees and the residual did not cause. docs/reward-shaping.md
    settled = env.episode_length_buf >= warmup_steps
    return torch.sum(torch.log1p(over), dim=1) * settled


class max_effort_ratio:
  """Maximum residual-joint actuator effort divided by its hardware limit."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    term = _residual_term(env, cfg.params.get("action_name", "mc_rtc_residual"))
    ids = term.residual_ids
    cols = list(range(len(term.target_names))) if ids is None else ids.tolist()
    limits = mc_rtc.get_effort_limits(term.cfg.mc_rtc_robot_name)
    self._cols = torch.tensor(cols, device=env.device, dtype=torch.long)
    self._limits = torch.tensor(
      [limits[term.target_names[i]] for i in cols], device=env.device
    )

  def __call__(
    self, env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
  ) -> torch.Tensor:
    term = _residual_term(env, action_name)
    effort = env.scene[term.cfg.entity_name].data.qfrc_actuator[:, term.target_ids]
    return (effort[:, self._cols].abs() / self._limits).amax(dim=1)


class recorded_disturbance:
  """Shared per-environment record for disturbance-aware terms."""

  #: Monotone counter, so this reads as "no push yet" for any run length.
  NEVER = -(1 << 30)

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    del cfg
    self.last_push_step = torch.full(
      (env.num_envs,), self.NEVER, dtype=torch.long, device=env.device
    )
    self.last_push_vel = torch.zeros((env.num_envs, 3), device=env.device)
    self.enabled = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)

  def disable(self, env_ids: torch.Tensor) -> None:
    """Suppress scheduled pushes for selected calibration environments."""
    self.enabled[env_ids] = False


class push_and_record(recorded_disturbance):
  """``push_by_setting_velocity``, plus a record of when it last fired."""

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg | None = None,
    warmup_s: float = 0.0,
    planar_speed: float | None = None,
  ) -> None:
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    ids = ids[self.enabled[ids]]
    if ids.numel() == 0:
      return
    if warmup_s > 0.0:
      # Suppress, do not reschedule: `EventManager` re-samples the countdown
      # whenever this fires. docs/difficulty.md#warmup_s
      ids = ids[env.episode_length_buf[ids] * env.step_dt >= warmup_s]
      if ids.numel() == 0:
        return
    asset = env.scene[(asset_cfg or SceneEntityCfg("robot")).name]
    # Mirrors `events.push_by_setting_velocity`, sampling here so the delta can be
    # recorded: `root_link_vel_w` comes from `cvel`, which MuJoCo does not
    # recompute until the next forward, so a before/after difference reads zero.
    vel_w = asset.data.root_link_vel_w[ids]
    if planar_speed is None:
      delta = events._sample_se3_range(velocity_range, vel_w.shape, str(env.device))
    else:
      angle = 2.0 * torch.pi * torch.rand(len(ids), device=env.device)
      delta = torch.zeros_like(vel_w)
      delta[:, 0] = planar_speed * torch.cos(angle)
      delta[:, 1] = planar_speed * torch.sin(angle)
    asset.write_root_link_velocity_to_sim(vel_w + delta, env_ids=ids)
    self.last_push_vel[ids] = quat_apply_inverse(
      asset.data.root_link_quat_w[ids], delta[:, :3]
    )
    self.last_push_step[ids] = env.common_step_counter


class finite_impulse_curriculum(recorded_disturbance):
  """Apply finite, mass-scaled torso impulses with a global-step curriculum."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self._env = env
    self.asset = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]
    self.interval_range_s = cfg.params["interval_range_s"]
    self.warmup_s = cfg.params["warmup_s"]
    self.force = torch.zeros(env.num_envs, 1, 3, device=env.device)
    self.torque = torch.zeros_like(self.force)
    self.remaining = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    self.next_push_step = torch.zeros_like(self.remaining)

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Clear active wrenches and schedule the first post-warmup impulse."""
    env = self._env
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    self._write_zeros(ids)
    self.last_push_step[ids] = self.NEVER
    self.last_push_vel[ids] = 0.0
    warmup = round(self.warmup_s / env.step_dt)
    span = max(
      1, round((self.interval_range_s[1] - self.interval_range_s[0]) / env.step_dt)
    )
    self.next_push_step[ids] = warmup + torch.randint(
      0, span + 1, (len(ids),), device=env.device
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    interval_range_s: tuple[float, float],
    warmup_s: float,
    duration_range_s: tuple[float, float],
    height_range_m: tuple[float, float],
    stages: tuple[tuple[int, tuple[float, float]], ...],
    enabled: bool = True,
    asset_cfg: SceneEntityCfg | None = None,
    rehearsal_weights: tuple[tuple[float, ...], ...] | None = None,
    initial_stage: int = 0,
    bands: tuple[tuple[float, float], ...] | None = None,
    band_weights: tuple[float, ...] | None = None,
  ) -> None:
    """Expire the current wrench and trigger any due curriculum impulse."""
    del (
      env_ids,
      interval_range_s,
      warmup_s,
      asset_cfg,
      rehearsal_weights,
      initial_stage,
      bands,
      band_weights,
    )
    active = self.remaining > 0
    self.remaining[active] -= 1
    expired = active & (self.remaining == 0)
    if bool(expired.any()):
      self._write_zeros(expired.nonzero(as_tuple=False).flatten())
    if not enabled:
      return
    due = self._due(env)
    ids = due.nonzero(as_tuple=False).flatten()
    if ids.numel() == 0:
      return
    self._trigger(env, ids, duration_range_s, height_range_m, stages)
    low, high = self.interval_range_s
    low_steps = round(low / env.step_dt)
    high_steps = round(high / env.step_dt)
    self.next_push_step[ids] = env.episode_length_buf[ids] + torch.randint(
      low_steps, high_steps + 1, (len(ids),), device=env.device
    )

  def _due(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Return environments whose next scheduled impulse has arrived."""
    return self.enabled & (env.episode_length_buf >= self.next_push_step)

  def _trigger(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    duration_range_s: tuple[float, float],
    height_range_m: tuple[float, float],
    stages: tuple[tuple[int, tuple[float, float]], ...],
  ) -> None:
    """Sample and install one force-equivalent planar velocity change."""
    velocity_range = stages[0][1]
    for step, candidate in stages:
      if env.common_step_counter >= step:
        velocity_range = candidate
    count = len(env_ids)
    angle = 2.0 * torch.pi * torch.rand(count, device=env.device)
    speed = velocity_range[0] + (velocity_range[1] - velocity_range[0]) * torch.rand(
      count, device=env.device
    )
    delta_b = torch.zeros(count, 3, device=env.device)
    delta_b[:, 0] = speed * torch.cos(angle)
    delta_b[:, 1] = speed * torch.sin(angle)
    quat = self.asset.data.root_link_quat_w[env_ids]
    delta_w = quat_apply(quat, delta_b)
    min_steps = math.ceil(duration_range_s[0] / env.step_dt)
    max_steps = math.floor(duration_range_s[1] / env.step_dt)
    duration_steps = torch.randint(
      min_steps, max_steps + 1, (count,), device=env.device
    )
    duration = duration_steps * env.step_dt
    body_ids = self.asset.indexing.body_ids
    mass = env.sim.model.body_mass[env_ids][:, body_ids].sum(dim=1)
    force = mass.unsqueeze(-1) * delta_w / duration.unsqueeze(-1)
    height = height_range_m[0] + (height_range_m[1] - height_range_m[0]) * torch.rand(
      count, device=env.device
    )
    offset_b = torch.zeros_like(force)
    offset_b[:, 2] = height
    torque = torch.cross(quat_apply(quat, offset_b), force, dim=1)
    self.force[env_ids, 0] = force
    self.torque[env_ids, 0] = torque
    self.remaining[env_ids] = duration_steps
    self.asset.write_external_wrench_to_sim(
      self.force[env_ids], self.torque[env_ids], env_ids=env_ids, body_ids=[0]
    )
    self.last_push_vel[env_ids] = delta_b
    self.last_push_step[env_ids] = env.common_step_counter

  def _write_zeros(self, env_ids: torch.Tensor) -> None:
    """Remove external wrenches for selected environments."""
    if env_ids.numel() == 0:
      return
    zeros = torch.zeros(len(env_ids), 1, 3, device=env_ids.device)
    self.asset.write_external_wrench_to_sim(zeros, zeros, env_ids=env_ids, body_ids=[0])
    self.force[env_ids] = 0.0
    self.torque[env_ids] = 0.0
    self.remaining[env_ids] = 0


def interpolated_impulse_range(
  step: int,
  stages: tuple[tuple[int, tuple[float, float]], ...],
) -> tuple[float, float]:
  """Linearly interpolate an impulse range between ordered curriculum stages."""
  if not stages:
    raise ValueError("impulse curriculum requires at least one stage")
  previous_step, previous_range = stages[0]
  for next_step, next_range in stages[1:]:
    if next_step <= previous_step:
      raise ValueError("impulse curriculum stages must have increasing steps")
    if step < next_step:
      if step <= previous_step:
        return previous_range
      fraction = (step - previous_step) / (next_step - previous_step)
      return (
        previous_range[0] + fraction * (next_range[0] - previous_range[0]),
        previous_range[1] + fraction * (next_range[1] - previous_range[1]),
      )
    previous_step, previous_range = next_step, next_range
  return previous_range


class gradual_finite_impulse_curriculum(finite_impulse_curriculum):
  """Apply the finite impulse curriculum with linear stage interpolation."""

  def _trigger(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    duration_range_s: tuple[float, float],
    height_range_m: tuple[float, float],
    stages: tuple[tuple[int, tuple[float, float]], ...],
  ) -> None:
    """Sample an impulse from the range interpolated at the current step."""
    velocity_range = interpolated_impulse_range(env.common_step_counter, stages)
    super()._trigger(
      env,
      env_ids,
      duration_range_s,
      height_range_m,
      ((0, velocity_range),),
    )


class stratified_finite_impulse_curriculum(finite_impulse_curriculum):
  """Draw each reset cohort from a stationary standing-plus-band mixture."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.bands = tuple(tuple(band) for band in cfg.params["bands"])
    weights = tuple(float(value) for value in cfg.params["band_weights"])
    if len(weights) != len(self.bands) + 1:
      raise ValueError("band weights need standing plus every impulse band")
    if abs(sum(weights) - 1.0) > 1e-9 or any(value < 0.0 for value in weights):
      raise ValueError("band weights must be nonnegative and sum to one")
    union = (min(low for low, _ in self.bands), max(high for _, high in self.bands))
    stages = cfg.params["stages"]
    # `stages` is inert here but reaches the manifest, so keep it honest.
    if len(stages) != 1 or tuple(stages[0][1]) != union:
      raise ValueError(f"stages must record the single band union {union}")
    self.band_weights = weights
    self._band_weights = torch.tensor(weights, device=env.device)
    self.sampled_band = torch.full(
      (env.num_envs,), -1, dtype=torch.long, device=env.device
    )

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Schedule an impulse and draw the reset cohort's magnitude band."""
    super().reset(env_ids)
    ids = (
      torch.arange(self._env.num_envs, device=self._env.device)
      if env_ids is None
      else env_ids
    )
    if ids.numel() == 0:
      return
    self.sampled_band[ids] = (
      torch.multinomial(self._band_weights, len(ids), replacement=True) - 1
    )

  def _due(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Exclude the standing cohort from scheduled disturbances."""
    return super()._due(env) & (self.sampled_band >= 0)

  def _trigger(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    duration_range_s: tuple[float, float],
    height_range_m: tuple[float, float],
    stages: tuple[tuple[int, tuple[float, float]], ...],
  ) -> None:
    """Draw each due environment inside the band chosen at its last reset."""
    del stages
    for band, values in enumerate(self.bands):
      ids = env_ids[self.sampled_band[env_ids] == band]
      if ids.numel():
        super()._trigger(env, ids, duration_range_s, height_range_m, ((0, values),))


class achievement_finite_impulse_curriculum(finite_impulse_curriculum):
  """Apply checkpointed difficulty with standing and prior-stage rehearsal."""

  is_achievement_curriculum = True

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    self.rehearsal_weights = cfg.params["rehearsal_weights"]
    self.current_stage = int(cfg.params.get("initial_stage", 0))
    self.sampled_stage = torch.full(
      (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
    self.set_stage(self.current_stage)

  def set_stage(self, stage: int) -> None:
    """Select the mixture used by environments at their next reset."""
    if not 0 <= stage < len(self.rehearsal_weights):
      raise ValueError(f"invalid achievement stage {stage}")
    weights = self.rehearsal_weights[stage]
    if len(weights) != len(self.rehearsal_weights) + 1:
      raise ValueError("rehearsal weights need standing plus every physical stage")
    if abs(sum(weights) - 1.0) > 1e-9 or any(value < 0.0 for value in weights):
      raise ValueError("rehearsal weights must be nonnegative and sum to one")
    if any(weights[stage + 2 :]):
      raise ValueError("rehearsal mixture cannot sample a future stage")
    self.current_stage = stage

  def reset(self, env_ids: torch.Tensor | None = None) -> None:
    """Schedule an impulse and sample the reset cohort's rehearsal level."""
    super().reset(env_ids)
    ids = (
      torch.arange(self._env.num_envs, device=self._env.device)
      if env_ids is None
      else env_ids
    )
    if ids.numel() == 0:
      return
    weights = torch.tensor(
      self.rehearsal_weights[self.current_stage], device=self._env.device
    )
    self.sampled_stage[ids] = torch.multinomial(weights, len(ids), replacement=True) - 1

  def _due(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Exclude the standing cohort from scheduled disturbances."""
    return super()._due(env) & (self.sampled_stage >= 0)

  def _trigger(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    duration_range_s: tuple[float, float],
    height_range_m: tuple[float, float],
    stages: tuple[tuple[int, tuple[float, float]], ...],
  ) -> None:
    """Sample each due environment from its reset-time rehearsal stage."""
    for stage in range(self.current_stage + 1):
      ids = env_ids[self.sampled_stage[env_ids] == stage]
      if ids.numel():
        super()._trigger(
          env,
          ids,
          duration_range_s,
          height_range_m,
          ((0, stages[stage][1]),),
        )

  def curriculum_state(self) -> dict[str, float | int]:
    """Expose current target, standing share, and earlier-stage rehearsal."""
    weights = self.rehearsal_weights[self.current_stage]
    return {
      "stage": self.current_stage,
      "standing_share": weights[0],
      "earlier_stage_share": sum(weights[1 : self.current_stage + 1]),
      "target_stage_share": weights[self.current_stage + 1],
    }


#: Age reported for an env that has not been pushed inside its current episode.
NEVER_AGE = 1 << 30


def _push_term(env: ManagerBasedRlEnv, term_name: str) -> recorded_disturbance:
  """The recorded disturbance behind ``term_name``, or a ``TypeError``."""
  term = env.event_manager.get_term_cfg(term_name).func
  if not isinstance(term, recorded_disturbance):
    raise TypeError(
      f"event term {term_name!r} must record disturbances for a "
      f"disturbance-gated reward to know when it fired, got {type(term).__name__}"
    )
  return term


def achievement_curriculum_state(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  term_name: str = "push_robot",
) -> dict[str, float | int]:
  """Report the achievement event's reset-cohort mixture."""
  del env_ids
  term = _push_term(env, term_name)
  if not isinstance(term, achievement_finite_impulse_curriculum):
    raise TypeError(f"event term {term_name!r} is not achievement gated")
  return term.curriculum_state()


def impulse_speed(
  env: ManagerBasedRlEnv, term_name: str = "push_robot"
) -> torch.Tensor:
  """Equivalent delta-velocity magnitude of each environment's last impulse."""
  return torch.linalg.vector_norm(_push_term(env, term_name).last_push_vel, dim=1)


def _age_since_push(env: ManagerBasedRlEnv, term: recorded_disturbance) -> torch.Tensor:
  """See :func:`steps_since_push`; this is that, with the term already resolved."""
  # Python int on the left: `torch.as_tensor` here would be an H2D copy per step.
  age = env.common_step_counter - term.last_push_step
  return age.masked_fill(age >= env.episode_length_buf, NEVER_AGE)


def steps_since_push(
  env: ManagerBasedRlEnv, term_name: str = "push_robot"
) -> torch.Tensor:
  """Policy steps since each env was last pushed *within its current episode*."""
  return _age_since_push(env, _push_term(env, term_name))


def push_recency(
  env: ManagerBasedRlEnv, term_name: str = "push_robot", tau_s: float = 2.0
) -> torch.Tensor:
  """1 at the instant of a push, decaying to 0; 0 for an env not pushed this episode."""
  # Bounded on purpose: `steps_since_push` reports NEVER_AGE (2^30), which would
  # wreck the observation normalizer if fed in raw.
  age = _age_since_push(env, _push_term(env, term_name)).clamp(min=0)
  return torch.exp(-age.float() * env.step_dt / tau_s).unsqueeze(-1)


def last_push_velocity(
  env: ManagerBasedRlEnv, term_name: str = "push_robot"
) -> torch.Tensor:
  """The velocity delta of the last push, zeroed once it predates this episode."""
  term = _push_term(env, term_name)
  within = (_age_since_push(env, term) < NEVER_AGE).unsqueeze(-1)
  return term.last_push_vel * within


def record_disturbance(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  equivalent_velocity_b: torch.Tensor,
  term_name: str = "push_robot",
) -> None:
  """Record a deterministic external disturbance for recovery terms."""
  term = _push_term(env, term_name)
  term.last_push_vel[env_ids] = equivalent_velocity_b
  term.last_push_step[env_ids] = env.common_step_counter


class recovery_dcm:
  """``dcm_stability``, paid only in the window after a push."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = _zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )
    # Resolved here: `get_term_cfg` walks every mode's name list.
    self._push = _push_term(env, cfg.params.get("push_term_name", "push_robot"))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    window_s: float,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    push_term_name: str = "push_robot",
    action_name: str = "mc_rtc_residual",
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg, push_term_name  # Resolved at init.
    error, normal_force = self._sensors.dcm_offset(
      env, action_name, min_normal_force, plane_height
    )
    age = _age_since_push(env, self._push)
    gate = (age >= 1) & (age <= round(window_s / env.step_dt))
    return (
      torch.exp(-torch.square(error / std)) * gate * (normal_force >= min_normal_force)
    )


class recovery_dcm_error:
  """Command-relative DCM error during a recorded recovery window."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = _zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )
    self._push = _push_term(env, cfg.params.get("push_term_name", "push_robot"))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    window_s: float,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    push_term_name: str = "push_robot",
    action_name: str = "mc_rtc_residual",
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg, push_term_name
    error, normal_force = self._sensors.dcm_offset(
      env, action_name, min_normal_force, plane_height
    )
    age = _age_since_push(env, self._push)
    active = (age >= 1) & (age <= round(window_s / env.step_dt))
    return error * active * (normal_force >= min_normal_force)


class recovery_authority_coverage:
  """Recovery-window steps that carried authority; read over ``recovery_active``."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = _zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )
    self._push = _push_term(env, cfg.params.get("push_term_name", "push_robot"))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    window_s: float,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    push_term_name: str = "push_robot",
    action_name: str = "mc_rtc_residual",
    min_normal_force: float = 20.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg, push_term_name
    normal_force = self._sensors.normal_forces(env).sum(dim=1)
    age = _age_since_push(env, self._push)
    active = (age >= 1) & (age <= round(window_s / env.step_dt))
    active = active & (normal_force >= min_normal_force)
    return active & (_residual_term(env, action_name).last_gate > 0.0)


class recovery_active:
  """Grounded indicator for the recorded post-disturbance recovery window."""

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = _zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )
    self._push = _push_term(env, cfg.params.get("push_term_name", "push_robot"))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    window_s: float,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    push_term_name: str = "push_robot",
    min_normal_force: float = 20.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg, push_term_name
    normal_force = self._sensors.normal_forces(env).sum(dim=1)
    age = _age_since_push(env, self._push)
    active = (age >= 1) & (age <= round(window_s / env.step_dt))
    return active * (normal_force >= min_normal_force)
