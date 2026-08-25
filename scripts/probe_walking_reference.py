"""Sweep DCM-feedback gains on the gated walking-reference velocity channel."""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from typing import cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase
from mc_mjlab.tasks import mdp
from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import (
  WALKING_REFERENCE_SCALE,
  _make_env_cfg,
)


@dataclass
class GainResult:
  """Fixed-horizon outcomes for one feedback gain."""

  gain: float
  envs: int
  hazards: int
  worker_failures: int
  mean_end_s: float
  recovery_dcm_error: float
  mean_command_delta: float
  max_restore_error: float


def dcm_error_vector(env, sensors, term) -> tuple[torch.Tensor, torch.Tensor]:
  """Return command-relative horizontal DCM error and grounded mask."""
  root = env.scene["robot"].indexing.root_body_id
  measured, normal_force = sensors.measured_offset(env)
  com = env.sim.data.subtree_com[:, root]
  com_vel = env.sim.data.subtree_linvel[:, root]
  commanded = term.controller_vector("control_com_vel")[:, :2]
  omega = torch.sqrt(mdp.GRAVITY / com[:, 2].clamp(min=mdp.MIN_COM_HEIGHT))
  error = (com_vel[:, :2] - commanded) / omega.unsqueeze(-1) - measured
  return error, normal_force >= 20.0


def apply_fixed_impulse(env, speed: float, height: float, duration_s: float) -> None:
  """Install one identical sagittal force-equivalent impulse in every world."""
  push = cast(mdp.finite_impulse_curriculum, mdp._push_term(env, "push_robot"))
  ids = torch.arange(env.num_envs, device=env.device)
  delta_b = torch.zeros(env.num_envs, 3, device=env.device)
  delta_b[:, 0] = speed
  quat = push.asset.data.root_link_quat_w
  delta_w = quat_apply(quat, delta_b)
  body_ids = push.asset.indexing.body_ids
  mass = env.sim.model.body_mass[:, body_ids].sum(dim=1)
  steps = round(duration_s / env.step_dt)
  force = mass.unsqueeze(-1) * delta_w / (steps * env.step_dt)
  offset_b = torch.zeros_like(force)
  offset_b[:, 2] = height
  torque = torch.cross(quat_apply(quat, offset_b), force, dim=1)
  push.force[ids, 0] = force
  push.torque[ids, 0] = torque
  push.remaining[ids] = steps
  push.asset.write_external_wrench_to_sim(
    push.force, push.torque, env_ids=ids, body_ids=[0]
  )
  push.last_push_vel[ids] = delta_b
  push.last_push_step[ids] = env.common_step_counter


def probe_walking_reference(args: argparse.Namespace) -> list[GainResult]:
  """Run every gain concurrently and return one aggregate row per cohort."""
  gains = tuple(args.gain or (-2.0, 0.0, 2.0, 4.0))
  num_envs = len(gains) * args.envs_per_gain
  cfg = _make_env_cfg(
    "position",
    num_envs=num_envs,
    num_workers=min(args.num_workers, num_envs),
    episode_length_s=args.seconds,
    walking_reference_velocity_scale=WALKING_REFERENCE_SCALE,
  )
  cfg.auto_reset = False
  cfg.seed = args.seed
  cfg.events["reset_base"].params["pose_range"] = {}
  cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
  cfg.events["push_robot"].params["enabled"] = False
  env = ManagerBasedRlEnv(cfg, device=args.device)
  step_dt = env.step_dt
  term = env.action_manager.get_term("mc_rtc_residual")
  if not isinstance(term, McRtcResidualActionBase):
    raise TypeError(f"unexpected action term {type(term).__name__}")
  sensors = mdp._ZmpSensors(env, mdp.GROUND_CONTACT_SENSORS, "robot")
  gain_by_env = torch.tensor(gains, device=env.device).repeat_interleave(
    args.envs_per_gain
  )
  scale = torch.tensor(WALKING_REFERENCE_SCALE, device=env.device)
  action = torch.zeros(num_envs, env.action_manager.total_action_dim, device=env.device)
  active = torch.ones(num_envs, dtype=torch.bool, device=env.device)
  elapsed = torch.zeros(num_envs, dtype=torch.long, device=env.device)
  hazards = torch.zeros_like(active)
  workers = torch.zeros_like(active)
  error_sum = torch.zeros(num_envs, device=env.device, dtype=torch.float64)
  error_count = torch.zeros_like(error_sum)
  command_sum = torch.zeros_like(error_sum)
  command_count = torch.zeros_like(error_sum)
  nominal_command = torch.zeros(num_envs, 3, device=env.device)
  try:
    env.reset()
    impulse_step = round(args.warmup_s / step_dt)
    impulse_fired = False
    while bool(active.any()):
      if not impulse_fired and int(elapsed.max()) >= impulse_step:
        nominal_command.copy_(term.controller_vector("walking_ref_vel"))
        apply_fixed_impulse(env, args.push_speed, args.push_height, args.push_duration)
        impulse_fired = True
      error, grounded = dcm_error_vector(env, sensors, term)
      delta_w = error * gain_by_env.unsqueeze(-1)
      delta_b = quat_apply_inverse(
        env.scene["robot"].data.root_link_quat_w,
        torch.nn.functional.pad(delta_w, (0, 1)),
      )
      action.zero_()
      action[:, -3:] = (delta_b / scale).clamp(-1.0, 1.0)
      elapsed[active] += 1
      _, _, terminated, time_outs, _ = env.step(action)

      age = mdp.steps_since_push(env)
      recovery = active & grounded & (age >= 1) & (age <= round(2.0 / env.step_dt))
      error_sum[recovery] += torch.linalg.vector_norm(error[recovery], dim=1).double()
      error_count[recovery] += 1.0
      command = torch.linalg.vector_norm(term.walking_reference_velocity[:, :2], dim=1)
      enabled = active & (term.last_gate > 0.0)
      command_sum[enabled] += command[enabled].double()
      command_count[enabled] += 1.0

      worker = env.termination_manager.get_term("controller_worker_failed")
      done = active & (terminated | time_outs)
      hazards |= done & terminated & ~worker
      workers |= done & worker
      active &= ~done
  finally:
    term.close()
    env.close()
    del env
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

  results = []
  restore_error = torch.linalg.vector_norm(
    term.controller_vector("walking_ref_vel") - nominal_command, dim=1
  )
  for gain in gains:
    cohort = gain_by_env == gain
    errors = error_sum[cohort].sum() / error_count[cohort].sum().clamp(min=1.0)
    commands = command_sum[cohort].sum() / command_count[cohort].sum().clamp(min=1.0)
    results.append(
      GainResult(
        gain=gain,
        envs=int(cohort.sum()),
        hazards=int(hazards[cohort].sum()),
        worker_failures=int(workers[cohort].sum()),
        mean_end_s=float(elapsed[cohort].float().mean()) * step_dt,
        recovery_dcm_error=float(errors),
        mean_command_delta=float(commands),
        max_restore_error=float(restore_error[cohort].max()),
      )
    )
  return results


def main() -> None:
  """Parse arguments, run the sweep, and print CSV."""
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--gain", action="append", type=float)
  parser.add_argument("--envs-per-gain", type=int, default=2)
  parser.add_argument("--num-workers", type=int, default=8)
  parser.add_argument("--seconds", type=float, default=22.0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--warmup-s", type=float, default=10.0)
  parser.add_argument("--push-speed", type=float, default=0.25)
  parser.add_argument("--push-height", type=float, default=0.20)
  parser.add_argument("--push-duration", type=float, default=0.10)
  args = parser.parse_args()
  print(
    "gain,envs,hazards,worker_failures,mean_end_s,recovery_dcm_error,"
    "mean_command_delta,max_restore_error",
    flush=True,
  )
  for result in probe_walking_reference(args):
    print(
      f"{result.gain:.3f},{result.envs},{result.hazards},"
      f"{result.worker_failures},{result.mean_end_s:.3f},"
      f"{result.recovery_dcm_error:.6f},{result.mean_command_delta:.6f},"
      f"{result.max_restore_error:.9f}",
      flush=True,
    )


if __name__ == "__main__":
  main()
