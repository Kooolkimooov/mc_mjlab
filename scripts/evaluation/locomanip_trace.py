"""Policy-step physical records and verified native force-offset delivery."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from evaluation.rollout import metric_snapshot
from mc_mjlab.tasks.locomanip.mdp import accessors, observations

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

  from mc_mjlab.actions.residual_feedback_action import ResidualFeedbackActionCfg

FAILURES = (*accessors.FALL_TERMINATIONS, "controller_failed")


class DispatchTrace:
  """Measure the force correction in the shared-memory input on every dispatch."""

  def __init__(self, env: ManagerBasedRlEnv) -> None:
    self.env = env
    self.term = accessors.residual_action(env)
    self.bridge = self.term._bridge
    self.original = self.bridge.fill_controller_input
    layout = self.bridge.layout.input
    cfg = cast("ResidualFeedbackActionCfg", self.term.cfg)
    names = cfg.wrench_sensor_names
    if (
      cfg.feedback_modalities != ("wrench",)
      or not cfg.wrench_force_only
      or len(names) != 2
      or self.term.executed_physical_action.shape[1] != 6
    ):
      raise ValueError("dispatch trace requires two force-only hand-wrench channels")
    self.columns = [
      layout.force_sensors_offset() + layout.force_sensors.index(name) * 6 + axis
      for name in names
      for axis in range(3)
    ]
    self.sensors = [
      int(env.sim.mj_model.sensor(f"robot/{name}_fsensor").adr[0]) + axis
      for name in names
      for axis in range(3)
    ]
    self.clear()
    self.bridge.fill_controller_input = self.capture  # ty: ignore[invalid-assignment]

  def clear(self) -> None:
    """Begin a new policy interval without losing individual dispatch extrema."""
    shape = (self.env.num_envs, 6)
    self.total = np.zeros(shape)
    self.square = np.zeros(shape)
    self.peak = np.zeros(shape)
    self.last = np.zeros(shape)
    self.active = np.zeros(self.env.num_envs)
    self.error = np.zeros(self.env.num_envs)
    self.count = 0

  def capture(self, rows: np.ndarray) -> None:
    """Subtract raw sensor readings from the actual native input force columns."""
    self.original(rows)
    raw = self.env.sim.data.sensordata[:, self.sensors].cpu().numpy()
    delivered = rows[:, self.columns] - raw
    expected = self.term.executed_physical_action.cpu().numpy()
    self.last = delivered.copy()
    self.total += delivered
    self.square += delivered**2
    self.peak = np.maximum(self.peak, np.abs(delivered))
    self.active += np.any(np.abs(delivered) > 1e-6, axis=1)
    self.error = np.maximum(self.error, np.max(np.abs(delivered - expected), axis=1))
    self.count += 1

  def close(self) -> None:
    """Restore the bridge before closing or reusing the environment."""
    self.bridge.fill_controller_input = self.original  # ty: ignore[invalid-assignment]


class SafetyTrace:
  """Keep first-episode extrema and terminal records before any reset."""

  def __init__(self, env: ManagerBasedRlEnv, rows: list[dict[str, Any]]) -> None:
    self.env, self.rows = env, rows
    self.finished = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.start_cart = (
      env.scene[accessors.OBJECT_ENTITY].data.root_link_pos_w[:, 0].clone()
    )
    self.start_root = env.scene["robot"].data.root_link_pos_w[:, 0].clone()
    self.lowest = np.full(env.num_envs, np.inf)
    self.tilt = np.zeros(env.num_envs)
    self.lag = np.zeros(env.num_envs)
    self.force_square = np.zeros((env.num_envs, 6))
    self.force_peak = np.zeros((env.num_envs, 6))
    self.force_count = np.zeros(env.num_envs)
    self.force_duration = np.zeros(env.num_envs)
    self.delivery_error = np.zeros(env.num_envs)
    self.valid = np.ones(env.num_envs, dtype=bool)
    self.results: list[dict[str, Any] | None] = [None] * env.num_envs

  def update(self, dispatch: DispatchTrace) -> dict[str, list]:
    """Accumulate live rows and return a per-policy-step diagnostic frame."""
    env = self.env
    robot = env.scene["robot"].data
    term = accessors.residual_action(env)
    signed = accessors.object_position_error(env)
    tilt = torch.acos((-robot.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
    tensors = {
      "time_s": env.episode_length_buf * env.step_dt,
      "root_position_m": robot.root_link_pos_w - env.scene.env_origins,
      "root_linear_velocity_m_s": robot.root_link_lin_vel_w,
      "root_angular_velocity_rad_s": robot.root_link_ang_vel_b,
      "tilt_rad": tilt,
      "cart_position_m": (
        env.scene[accessors.OBJECT_ENTITY].data.root_link_pos_w - env.scene.env_origins
      ),
      "cart_velocity_m_s": env.scene[accessors.OBJECT_ENTITY].data.root_link_lin_vel_w,
      "reference_position_m": term.datastore_vector_output(
        accessors.OBJECT_REFERENCE_POSITION
      ),
      "signed_cart_error_m": signed,
      "phases": observations.manipulation_phase(env),
      "hand_contact_force_n": accessors.hand_contact_forces(env),
      "requested_action": term.raw_action,
      "controller_q": term.controller_reference("q"),
    }
    frame = {name: value.cpu().tolist() for name, value in tensors.items()}
    frame["dispatched_force_n"] = dispatch.last.tolist()
    frame["mean_dispatched_force_n"] = (
      dispatch.total / max(dispatch.count, 1)
    ).tolist()
    frame["dispatch_error_n"] = dispatch.error.tolist()
    termination_names = env.termination_manager.active_terms
    termination_values = (
      torch.stack(
        [env.termination_manager.get_term(name) for name in termination_names], dim=1
      )
      .cpu()
      .tolist()
    )
    frame["terminations"] = [
      dict(zip(termination_names, values, strict=True)) for values in termination_values
    ]
    live = ~self.finished.cpu().numpy()
    self.lowest[live] = np.minimum(
      self.lowest[live], np.asarray(frame["root_position_m"])[live, 2]
    )
    self.tilt[live] = np.maximum(self.tilt[live], np.asarray(frame["tilt_rad"])[live])
    self.lag[live] = np.maximum(
      self.lag[live], signed[:, 0].clamp(min=0).cpu().numpy()[live]
    )
    self.force_square[live] += dispatch.square[live]
    self.force_count[live] += dispatch.count
    self.force_peak[live] = np.maximum(self.force_peak[live], dispatch.peak[live])
    self.force_duration[live] += (
      dispatch.active[live] * env.cfg.sim.mujoco.timestep * term.cfg.frameskip
    )
    self.delivery_error[live] = np.maximum(
      self.delivery_error[live], dispatch.error[live]
    )
    for value in tensors.values():
      finite = torch.isfinite(value).reshape(env.num_envs, -1).all(dim=1).cpu().numpy()
      self.valid[live] &= finite[live]
    return frame

  def finish(
    self, ids: torch.Tensor, frame: dict[str, list], observed: bool = True
  ) -> None:
    """Save terminal outcomes and metric reductions while the episode still exists."""
    env = self.env
    measured = metric_snapshot(env, ids)
    for offset, index in enumerate(ids.tolist()):
      flags = frame["terminations"][index]
      failed = any(flags.get(name, False) for name in FAILURES)
      worker = flags.get("controller_worker_failed", False)
      valid = bool(self.valid[index]) and observed
      complete = (
        bool(flags.get("task_complete", False)) and valid and not worker and not failed
      )
      self.results[index] = {
        **self.rows[index],
        "terminations": flags,
        "combined_failure": failed,
        "worker_failure": worker,
        "observed": valid,
        "complete": complete,
        "incomplete": not complete and not failed,
        "time_s": frame["time_s"][index],
        "failure_time_s": frame["time_s"][index] if failed else None,
        "phases": frame["phases"][index],
        "min_height_m": float(self.lowest[index]),
        "max_tilt_rad": float(self.tilt[index]),
        "max_forward_lag_m": float(self.lag[index]),
        "terminal_position_error_m": float(
          accessors.object_position_error(env)[index].norm()
        ),
        "cart_displacement_m": float(
          env.scene[accessors.OBJECT_ENTITY].data.root_link_pos_w[index, 0]
          - self.start_cart[index]
        ),
        "robot_displacement_m": float(
          env.scene["robot"].data.root_link_pos_w[index, 0] - self.start_root[index]
        ),
        "force_rms_n": np.sqrt(
          self.force_square[index] / max(self.force_count[index], 1)
        ).tolist(),
        "force_peak_n": self.force_peak[index].tolist(),
        "force_duration_s": float(self.force_duration[index]),
        "delivery_error_n": float(self.delivery_error[index]),
        "metrics": {name: values[offset] for name, values in measured.items()},
      }
    self.finished[ids] = True
