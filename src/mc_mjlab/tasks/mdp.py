"""MDP terms that talk about the mc_rtc controller itself, not just robot state."""

from __future__ import annotations

from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import mujoco
import torch
from mjlab.envs.mdp import events
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

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


def action_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared magnitude of the residual action, saturating where the clip does."""
  # Past the clip a larger raw action has no physical effect, so paying more for it
  # is a runaway: it diverged a run. docs/reward-shaping.md#action_l2
  action = env.action_manager.action.clamp(-RAW_CLIP, RAW_CLIP)
  return torch.sum(torch.square(action), dim=1)


def action_rate_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Squared change in the residual action, saturating where the clip does."""
  # mjlab's version charges the raw action unboundedly, the same runaway
  # `action_l2` was clamped for. docs/reward-shaping.md#action_l2
  manager = env.action_manager
  delta = manager.action.clamp(-RAW_CLIP, RAW_CLIP) - manager.prev_action.clamp(
    -RAW_CLIP, RAW_CLIP
  )
  return torch.sum(torch.square(delta), dim=1)


def _restrict(term: McRtcResidualActionBase, values: torch.Tensor) -> torch.Tensor:
  """Keep only the columns carrying the residual (see ``residual_ids``)."""
  ids = term.residual_ids
  return values if ids is None else values[:, ids]


def executed_action(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The residual as actually applied, gate included -- not the raw request."""
  # `last_action` is the network's intent; the gate scales it on ~59% of steps, so
  # the policy's own history would misreport its dynamics. docs/observations.md
  term = _residual_term(env, action_name)
  return term.processed_action * term.last_gate.unsqueeze(-1)


def gate_mean(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """Coherence-gate factor, 1 where nothing is withheld."""
  return _residual_term(env, action_name).last_gate


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

#: Raw action at which the residual clip binds, given `clip` is set to `scale`.
RAW_CLIP = 1.0


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
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(distance from the divergent component to the CoP, vertical force)``."""
    # LIPM: d(xi)/dt = omega * (xi - CoP), so this norm *is* the divergence rate.
    measured, normal_force = self.measured_offset(env, min_normal_force, plane_height)
    data = env.sim.data
    com = data.subtree_com[:, self.root_body_id]
    com_vel = data.subtree_linvel[:, self.root_body_id]
    omega = torch.sqrt(GRAVITY / com[:, 2].clamp(min=MIN_COM_HEIGHT))
    capture = com_vel[:, :2] / omega.unsqueeze(-1)
    return torch.linalg.vector_norm(capture - measured, dim=1), normal_force


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
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg  # Resolved at init.
    error, normal_force = self._sensors.dcm_offset(env, min_normal_force, plane_height)
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
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg  # Resolved at init.
    error, normal_force = self._sensors.dcm_offset(env, min_normal_force, plane_height)
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


class push_and_record:
  """``push_by_setting_velocity``, plus a record of when it last fired."""

  #: Monotone counter, so this reads as "no push yet" for any run length.
  NEVER = -(1 << 30)

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    del cfg
    self.last_push_step = torch.full(
      (env.num_envs,), self.NEVER, dtype=torch.long, device=env.device
    )
    self.last_push_vel = torch.zeros((env.num_envs, 3), device=env.device)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg | None = None,
    warmup_s: float = 0.0,
  ) -> None:
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    if warmup_s > 0.0:
      # Suppress, do not reschedule: `EventManager` re-samples the countdown
      # whenever this fires. docs/difficulty.md#push_warmup_s
      ids = ids[env.episode_length_buf[ids] * env.step_dt >= warmup_s]
      if ids.numel() == 0:
        return
    asset = env.scene[(asset_cfg or SceneEntityCfg("robot")).name]
    # Mirrors `events.push_by_setting_velocity`, sampling here so the delta can be
    # recorded: `root_link_vel_w` comes from `cvel`, which MuJoCo does not
    # recompute until the next forward, so a before/after difference reads zero.
    vel_w = asset.data.root_link_vel_w[ids]
    delta = events._sample_se3_range(velocity_range, vel_w.shape, str(env.device))
    asset.write_root_link_velocity_to_sim(vel_w + delta, env_ids=ids)
    self.last_push_vel[ids] = quat_apply_inverse(
      asset.data.root_link_quat_w[ids], delta[:, :3]
    )
    self.last_push_step[ids] = env.common_step_counter


#: Age reported for an env that has not been pushed inside its current episode.
NEVER_AGE = 1 << 30


def _push_term(env: ManagerBasedRlEnv, term_name: str) -> push_and_record:
  """The ``push_and_record`` behind ``term_name``, or a ``TypeError``."""
  term = env.event_manager.get_term_cfg(term_name).func
  if not isinstance(term, push_and_record):
    raise TypeError(
      f"event term {term_name!r} must be `mdp.push_and_record` for a "
      f"disturbance-gated reward to know when it fired, got {type(term).__name__}"
    )
  return term


def _age_since_push(env: ManagerBasedRlEnv, term: push_and_record) -> torch.Tensor:
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
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg, push_term_name  # Resolved at init.
    error, normal_force = self._sensors.dcm_offset(env, min_normal_force, plane_height)
    age = _age_since_push(env, self._push)
    gate = (age >= 1) & (age <= round(window_s / env.step_dt))
    return (
      torch.exp(-torch.square(error / std)) * gate * (normal_force >= min_normal_force)
    )
