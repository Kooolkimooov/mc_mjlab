"""MDP terms specific to residual control on top of an mc_rtc controller.

mjlab's own terms cover generic robot state; these exist because a residual
task needs to talk about the controller itself: how far the policy departs
from it (``action_l2``), what it is asking for versus what it got
(``controller_position_error``, ``controller_reference_velocity``,
``zmp_tracking``, ``com_velocity_tracking``), whether it is still generating a
gait (``controller_reference_motion``) and whether it has given up
(``controller_failed``). What they compare against is the raw controller output
with the residual excluded, read off the action term.

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

import mujoco
import torch
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
    mj_model = env.sim.mj_model
    force_cols: list[int] = []
    torque_cols: list[int] = []
    site_ids: list[int] = []
    for name in cfg.params["sensor_names"]:
      f_adr, site_id = _wrench_sensor(
        mj_model, f"{name}_fsensor", mujoco.mjtSensor.mjSENS_FORCE
      )
      t_adr, _ = _wrench_sensor(
        mj_model, f"{name}_tsensor", mujoco.mjtSensor.mjSENS_TORQUE
      )
      force_cols += [f_adr, f_adr + 1, f_adr + 2]
      torque_cols += [t_adr, t_adr + 1, t_adr + 2]
      site_ids.append(site_id)

    self._force_cols = torch.tensor(force_cols, device=env.device, dtype=torch.long)
    self._torque_cols = torch.tensor(torque_cols, device=env.device, dtype=torch.long)
    self._site_ids = torch.tensor(site_ids, device=env.device, dtype=torch.long)
    self._num_sensors = len(site_ids)
    # subtree_com of the robot's root body is the whole robot's CoM, and it is
    # MuJoCo's own, so it needs no sensor and carries no estimation error.
    self._root_body_id = env.scene[cfg.params["asset_cfg"].name].indexing.root_body_id

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
    num_envs, k = env.num_envs, self._num_sensors
    data = env.sim.data

    rot = data.site_xmat[:, self._site_ids].reshape(num_envs, k, 3, 3)
    site_pos = data.site_xpos[:, self._site_ids].reshape(num_envs, k, 3)
    sensordata = data.sensordata
    force_s = sensordata[:, self._force_cols].reshape(num_envs, k, 3, 1)
    torque_s = sensordata[:, self._torque_cols].reshape(num_envs, k, 3, 1)

    # MuJoCo reports the wrench transmitted at the sensor site; the reaction the
    # ground applies to the robot is its negation -- the same `fs *= -1` the I/O
    # binding applies before handing these to mc_rtc.
    force_w = -(rot @ force_s).squeeze(-1)
    torque_w = -(rot @ torque_s).squeeze(-1)

    com = data.subtree_com[:, self._root_body_id]
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

    term = _residual_term(env, action_name)
    planned = (
      term.controller_vector("planned_zmp") - term.controller_vector("control_com")
    )[:, :2]

    error = torch.linalg.vector_norm(measured - planned, dim=1)
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
