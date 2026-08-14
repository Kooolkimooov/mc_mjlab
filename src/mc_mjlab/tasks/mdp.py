"""MDP terms specific to residual control on top of an mc_rtc controller.

mjlab's own terms cover generic robot state; these exist because a residual
task needs to talk about the controller itself: how far the policy departs
from it (``action_l2``), what it is asking for versus what it got
(``controller_position_error``, ``controller_reference_velocity``,
``zmp_tracking``, ``com_velocity_tracking``), what it intends
(``controller_planned_zmp_offset``, ``controller_planned_com_velocity``),
whether it is still generating a gait (``controller_reference_motion``) and
whether it has given up (``controller_failed``). What they compare against is
the raw controller output with the residual excluded, read off the action term.

The two ``controller_planned_*`` terms are observations, not rewards, and they
exist to close a gap the residual balance task ran into: it *pays* for tracking
``planned_zmp`` and ``control_com_vel`` while showing the actor neither, so the
policy was scored against a plan it could not see. What it could see of the
controller was ``controller_reference_velocity``, joint-level and one
integration removed from the centroidal quantities the reward is written on.
A policy with no way to know when intervening helps has one safe strategy left
-- intervene less -- and that is what the measurements showed it converging to:
mean action falling 31% -> 14% of its clip while performance approached the
zero-residual baseline from below rather than passing it.

Both are controller-internal and exact, so they carry no observation noise, the
same as ``controller_reference_velocity``. They are *plans*, not errors: the
measured side of ``com_velocity_tracking`` is largely inferable from
``base_lin_vel``, but the measured side of ``zmp_tracking`` -- the centre of
pressure under the feet -- is in no observation group either, so adding it is
the obvious next thing to try if these two are not enough on their own.

``controller_reference_motion`` is here but is *not* wired into the residual
balance task, and should not be without re-measuring: the controller's
joint-velocity reference is larger in the run-up to a fall than in normal
walking (1.78 vs 0.64 rad/s over 96 s x 16 envs), because a falling robot's
controller thrashes, so paying for it pays for the fall.

A note on what is *not* here, because it looks like it should be. A term
comparing commanded joint positions against measured ones does not measure
"is the controller's plan being executed" under this coupling: the joints are
position-controlled with stiff PD, so the measured angle follows the commanded
one to within 0.04 rad even while the robot topples (measured), and since the
command is reference-plus-residual such a term reduces to a second penalty on
the residual. Whole-body failure shows up in the base -- attitude, height,
travel -- which is what the task's terminations and ``base_progress_tanh``
read instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import mujoco
import torch
from mjlab.envs.mdp import events
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase

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
  """Squared magnitude of the residual action."""
  return torch.sum(torch.square(env.action_manager.action), dim=1)


def _restrict(term: McRtcResidualActionBase, values: torch.Tensor) -> torch.Tensor:
  """Keep only the columns carrying the residual (see ``residual_ids``)."""
  ids = term.residual_ids
  return values if ids is None else values[:, ids]


def controller_position_error(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """What the controller asked of the residual joints, minus what it got."""
  term = _residual_term(env, action_name)
  asset = env.scene[term.cfg.entity_name]
  error = term.controller_reference("q") - asset.data.joint_pos[:, term.target_ids]
  return _restrict(term, error)


def controller_reference_velocity(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The controller's joint-velocity reference: where its gait is headed."""
  term = _residual_term(env, action_name)
  return _restrict(term, term.controller_reference("alpha"))


def controller_planned_zmp_offset(
  env: ManagerBasedRlEnv, action_name: str = "mc_rtc_residual"
) -> torch.Tensor:
  """The controller's planned CoM-to-ZMP offset: where it means to push.

  The observation face of ``planned_zmp_offset``, which the ZMP reward and metric
  score against -- same quantity, so they cannot drift apart.
  """
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


##
# ZMP tracking.
##

# mc_rtc's names for the force sensors carrying the ground reaction. The MuJoCo
# models call the matching pair "<prefix><name>_fsensor"/"_tsensor" (mc_mujoco's
# convention -- the very sensors ControllerIoBinding feeds the controller, so
# the reward measures the ZMP off the same signals the stabilizer sees).
GROUND_CONTACT_SENSORS = ("LeftFootForceSensor", "RightFootForceSensor")


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


class _ZmpSensors:
  """Sensor plumbing for the measured centre of pressure, resolved once.

  Split out of ``zmp_tracking`` so the reward is not the only way to reach this
  number. The reward reports ``exp(-(error/std)^2)``, a bounded kernel output
  that says nothing about metres, and every episode-sum of it turned out to
  correlate with episode length at r = +0.98 -- so asking "did the residual
  actually move the centre of pressure" needed the raw quantity, which only
  existed inside the reward's ``__call__``.
  """

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
    # subtree_com of the robot's root body is the whole robot's CoM, and it is
    # MuJoCo's own, so it needs no sensor and carries no estimation error.
    self.root_body_id = env.scene[asset_name].indexing.root_body_id

    # One-slot memo for `offset_error`, keyed on the step it was computed for.
    self._cache_key: tuple[object, ...] | None = None
    self._cache: tuple[torch.Tensor, torch.Tensor] | None = None

  def measured_offset(
    self,
    env: ManagerBasedRlEnv,
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(CoM-to-ZMP offset xy, vertical contact force)``, both ``(num_envs, ...)``."""
    num_envs, k = env.num_envs, self.num_sensors
    data = env.sim.data

    rot = data.site_xmat[:, self.site_ids].reshape(num_envs, k, 3, 3)
    site_pos = data.site_xpos[:, self.site_ids].reshape(num_envs, k, 3)
    sensordata = data.sensordata
    force_s = sensordata[:, self.force_cols].reshape(num_envs, k, 3, 1)
    torque_s = sensordata[:, self.torque_cols].reshape(num_envs, k, 3, 1)

    # MuJoCo reports the wrench transmitted at the sensor site; the reaction the
    # ground applies to the robot is its negation -- the same `fs *= -1` the I/O
    # binding applies before handing these to mc_rtc.
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
    return measured, normal_force

  def offset_error(
    self,
    env: ManagerBasedRlEnv,
    action_name: str,
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """``(distance from the planned ZMP in metres, vertical contact force)``.

    Memoised for the current step, because three terms score this same number
    -- ``zmp_tracking``, ``recovery_tracking`` and the ``zmp_error`` metric --
    and computing it is not free: the ``site_xmat``/``site_xpos`` gathers, a
    batched ``(num_envs, k, 3, 3) @ (num_envs, k, 3, 1)``, the cross products
    and the action-term lookup, at 50 Hz x num_envs.

    ``common_step_counter`` is a sound key because terminations, rewards and
    metrics all run in one phase of ``ManagerBasedRlEnv.step`` off the same
    ``sim.data``; the tolerances are in the key too, so a term configured with
    different ones still gets its own value rather than someone else's.
    """
    key = (env.common_step_counter, action_name, min_normal_force, plane_height)
    if self._cache_key != key or self._cache is None:
      measured, normal_force = self.measured_offset(env, min_normal_force, plane_height)
      error = torch.linalg.vector_norm(
        measured - planned_zmp_offset(env, action_name), dim=1
      )
      self._cache_key, self._cache = key, (error, normal_force)
    return self._cache


#: Per-env ``_ZmpSensors``, keyed weakly so they die with the env they resolved
#: their model addresses against.
_ZMP_SENSOR_CACHE: WeakKeyDictionary[
  ManagerBasedRlEnv, dict[tuple[tuple[str, ...], str], _ZmpSensors]
] = WeakKeyDictionary()


def _zmp_sensors(
  env: ManagerBasedRlEnv, sensor_names: tuple[str, ...], asset_name: str
) -> _ZmpSensors:
  """The one :class:`_ZmpSensors` for this env and sensor set.

  Terms resolve their plumbing at init, so three of them each building their
  own meant three copies of the same index tensors and, worse, three separate
  ``offset_error`` memos -- which defeats the memo entirely.
  """
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


class zmp_error:
  """Distance between the measured centre of pressure and the planned one, in metres.

  A *metric*, not a reward: unlike ``zmp_tracking`` this is not passed through a
  kernel and not accumulated over the episode, so it is comparable between runs
  of different length. That matters because the episode-sum rewards proved to be
  episode length in disguise (r = +0.98), which made the tensorboard curves
  unreadable as a measure of control quality.

  **Read it together with :class:`zmp_grounded`.** Steps with the feet unloaded
  contribute 0 m here rather than a number divided by a vanishing normal force,
  and ``MetricsManager`` averages as ``sum / step_count`` over *every* step --
  so on its own this curve falls when the robot spends more time off the ground,
  which is the wrong direction for a tracking error. ``MetricsTermCfg.reduce``
  offers no masked mean, so the denominator is published separately instead:
  the grounded-conditional error is ``zmp_error / zmp_grounded``.
  """

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
  """Share of steps whose feet carry enough load for a centre of pressure.

  The denominator :class:`zmp_error` needs to be read in metres, and a health
  curve in its own right: a robot lifting off, stumbling or on its way down
  spends more steps ungrounded, and that shows up here before it shows up in a
  termination.
  """

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
    # Free: `zmp_error` computed this on the same step, and the sensors are
    # shared, so this reads the memo rather than the sensors.
    _, normal_force = self._sensors.offset_error(
      env, action_name, min_normal_force, plane_height
    )
    return (normal_force >= min_normal_force).float()


class zmp_tracking:
  """Reward the measured ZMP sitting where the mc_rtc controller wants it.

  The controller side is ``planned_zmp``: the centroidal ZMP of the QP's own
  solution, i.e. the ZMP the motion mc_rtc commands this period implies (see
  ``mc_rtc_controller_host._planned_zmp``). The sim side is the centre of
  pressure of the foot wrenches, summed and moved onto the ground plane exactly
  as ``mc_rbdyn::zmp`` does -- ``zmp = p + n x moment_p / (n.f)``, which for a
  flat floor is ``(-M_y, M_x) / F_z``.

  Both are taken **relative to their own CoM**, and that is not cosmetic: the
  controller places its plan against the state its observers estimate
  (KinematicInertial legged odometry), which drifts from MuJoCo's ground truth
  over an episode. Differencing absolute positions would charge the residual for
  that drift; the CoM-to-ZMP offset is drift-free and is anyway the quantity the
  LIPM relation the plan comes from is written on.

  Payment is ``exp(-(error/std)^2)`` on the horizontal offset error, and zero
  while the feet carry less than ``min_normal_force``: a robot in the air has no
  centre of pressure to place, and dividing by its vanishing normal force would
  put the ZMP anywhere.
  """

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
  """Reward the robot's CoM moving the way the controller is commanding.

  The velocity half of the LIPM state, and the complement to ``zmp_tracking``:
  that term compares the CoM-to-ZMP offset (the forcing), this one the CoM
  velocity it is supposed to produce. Between them they span the planar state
  the plan is written on, which is why there is no separate DCM term -- the
  divergent mode ``com + comVel/omega`` is a linear combination of the two, so
  any DCM weighting is already reachable by choosing these two weights.

  Unlike the ZMP this needs no drift correction: KinematicInertial integrates
  *position* from the anchor frame, so position is what drifts away from
  MuJoCo's truth, while velocity is differential and directly comparable.

  The vertical axis is scored *separately* from the horizontal pair, and that
  is what makes the term worth having beyond ``zmp_tracking``: a crouch-collapse
  -- the most common baseline failure -- is the CoM sinking against a plan that
  holds its height, visible here long before ``collapsed`` fires. It only works
  split, for two reasons that compound. One exponential over a 3-norm lets the
  largest axis saturate the kernel and kill the gradient on the others, exactly
  when the others still matter; and the vertical error is far smaller than the
  horizontal one (median 0.0010 against 0.0117 m/s over 64 s x 16 envs), so a
  shared ``std`` makes the vertical channel contribute nothing at all. Two
  kernels with their own scales, averaged, keeps both channels live.

  Horizontal stays a 2-norm rather than two more kernels: x and y are
  interchangeable for balance, so the term should not care which way the robot
  is drifting.
  """

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
    # subtree_linvel is the subtree CoM's velocity; MuJoCo fills it because the
    # robots carry a subtree sensor (the RL-only `root_angmom`), which is what
    # makes mj_subtreeVel run.
    error = env.sim.data.subtree_linvel[:, self._root_body_id] - term.controller_vector(
      "control_com_vel"
    )
    horizontal = torch.linalg.vector_norm(error[:, :2], dim=1)
    vertical = error[:, 2].abs()
    return 0.5 * (
      torch.exp(-torch.square(horizontal / std))
      + torch.exp(-torch.square(vertical / std_vertical))
    )


class push_and_record:
  """``push_by_setting_velocity``, plus a record of when it last fired.

  The kick itself is mjlab's, unchanged. What this adds is ``last_push_step``,
  which is what lets a reward pay attention only to the window where the residual
  can plausibly help. Most steps of an episode are nominal walking, where mc_rtc
  already tracks its own plan and there is nothing for a residual to add; the
  measured per-step tracking rate is the same with a trained policy as with none
  (0.01183 vs 0.01196, SE 0.00015). Those steps dilute the gradient from the few
  hundred milliseconds after a disturbance where the difference is made.

  Gating on this is safe in a way that gating on *measured error* would not be:
  the push schedule is drawn by the event manager and is entirely independent of
  the policy, so the agent cannot arrange to be paid more often. A gate keyed on
  its own tracking error would be exactly that -- an incentive to enter the
  high-paying state.
  """

  #: ``common_step_counter`` is monotone across episodes, so a value this far in
  #: the past reads as "no push yet" for any run length.
  NEVER = -(1 << 30)

  def __init__(self, cfg, env: ManagerBasedRlEnv) -> None:
    del cfg
    self.last_push_step = torch.full(
      (env.num_envs,), self.NEVER, dtype=torch.long, device=env.device
    )

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
      # Suppress, do not reschedule. `EventManager` owns the countdown and
      # re-samples it whenever the term fires regardless of what this returns,
      # so skipping here delays the *first* push without altering the 5-7 s
      # cadence that follows -- and the first real push then lands on the first
      # tick after the warm-up, which desynchronises it across envs instead of
      # hitting every robot at the same phase.
      #
      # Why it exists: measured over 528 zero-residual episodes, the hazard rate
      # is 0.122/s in the 4-8 s window and ~0.019/s everywhere after, so 48% of
      # all deaths landed on the first push -- which arrives while the robot is
      # still finishing its ~4 s posture settle. That made the task a startup
      # lottery rather than a test of push recovery while walking.
      ids = ids[env.episode_length_buf[ids] * env.step_dt >= warmup_s]
      if ids.numel() == 0:
        return
    events.push_by_setting_velocity(
      env, ids, velocity_range, asset_cfg or SceneEntityCfg("robot")
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
  # A Python int on the left: `int - tensor` broadcasts, and wrapping the step
  # counter in `torch.as_tensor` would put a host-to-device copy on a path that
  # runs once per step per reward.
  age = env.common_step_counter - term.last_push_step
  # Never-pushed and previous-episode pushes both read as "long ago".
  return age.masked_fill(age >= env.episode_length_buf, NEVER_AGE)


def steps_since_push(
  env: ManagerBasedRlEnv, term_name: str = "push_robot"
) -> torch.Tensor:
  """Policy steps since each env was last pushed *within its current episode*.

  Bounded by ``episode_length_buf`` on purpose. Class-based event terms get no
  ``reset`` callback from mjlab's ``EventManager`` -- it only re-samples the
  interval timers -- so ``last_push_step`` survives an episode boundary. Rather
  than reach for a reset hook that does not exist, a push is simply not counted
  unless it happened after this episode started, which is what
  ``age < episode_length_buf`` says. Envs with no push yet read a huge age.

  Reading ``last_push_step`` rather than watching the event manager's interval
  countdown is the only way to get this right: the countdown is re-sampled
  whenever the term *fires*, including the ticks ``push_and_record`` suppresses
  during ``warmup_s``, so a timer-based detector counts pushes that never landed.

  Resolves the event term on every call, which is what a diagnostic wants;
  ``recovery_tracking`` resolves it once at init and calls
  :func:`_age_since_push`.
  """
  return _age_since_push(env, _push_term(env, term_name))


class recovery_tracking:
  """``zmp_tracking``, paid only in the window after a push.

  Same quantity and same kernel as :class:`zmp_tracking` -- the point is not to
  measure something new but to stop averaging the informative steps into the
  ~90% of an episode where the controller is undisturbed and the residual has
  nothing to contribute. Sized from the measured recovery profile: the ZMP error
  is elevated for roughly the first two seconds after a kick and settles after
  that, so a window much longer than that would re-admit the steps this exists
  to exclude.

  Interval events fire *after* the reward is computed (see
  ``manager_based_rl_env.step``), so the first step this can pay on has an age of
  1, never 0.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self._sensors = _zmp_sensors(
      env, cfg.params["sensor_names"], cfg.params["asset_cfg"].name
    )
    # Resolved here rather than per call: `get_term_cfg` walks every mode's
    # name list, and this is a reward.
    self._push = _push_term(env, cfg.params.get("push_term_name", "push_robot"))

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std: float,
    window_s: float,
    sensor_names: tuple[str, ...],
    asset_cfg: SceneEntityCfg,
    action_name: str = "mc_rtc_residual",
    push_term_name: str = "push_robot",
    min_normal_force: float = 20.0,
    plane_height: float = 0.0,
  ) -> torch.Tensor:
    del sensor_names, asset_cfg, push_term_name  # Resolved at init.
    error, normal_force = self._sensors.offset_error(
      env, action_name, min_normal_force, plane_height
    )
    age = _age_since_push(env, self._push)
    gate = (age >= 1) & (age <= round(window_s / env.step_dt))
    return (
      torch.exp(-torch.square(error / std)) * gate * (normal_force >= min_normal_force)
    )
