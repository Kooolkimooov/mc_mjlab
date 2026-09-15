"""Sweep recovery-gated walking step-duration deltas after a fixed impulse."""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from typing import cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import quat_apply

from mc_mjlab.actions.mc_rtc_residual_action import (
  McRtcResidualActionBase,
  McRtcResidualActionCfg,
)
from mc_mjlab.tasks import mdp
from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import _make_env_cfg

STEP_TIME_GETTER = "ismpc_walking::get_ts_target"
STEP_TIME_CALLBACKS = (STEP_TIME_GETTER, "ismpc_walking::set_ts")


@dataclass
class DurationResult:
  """Fixed-horizon outcomes for one walking step-duration delta."""

  delta_s: float
  envs: int
  hazards: int
  worker_failures: int
  mean_end_s: float
  recovery_dcm_error: float
  improvement_pct: float
  mean_abs_applied_delta_s: float
  max_restore_error_s: float


def dcm_error_vector(
  env: ManagerBasedRlEnv, sensors: mdp._ZmpSensors, term: McRtcResidualActionBase
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return command-relative horizontal DCM error and grounded mask."""
  root = env.scene["robot"].indexing.root_body_id
  measured, normal_force = sensors.measured_offset(env)
  com = env.sim.data.subtree_com[:, root]
  com_vel = env.sim.data.subtree_linvel[:, root]
  commanded = term.controller_vector(mdp.CONTROL_COM_VEL)[:, :2]
  omega = torch.sqrt(mdp.GRAVITY / com[:, 2].clamp(min=mdp.MIN_COM_HEIGHT))
  error = (com_vel[:, :2] - commanded) / omega.unsqueeze(-1) - measured
  return error, normal_force >= 20.0


def apply_fixed_impulse(
  env: ManagerBasedRlEnv, speed: float, height: float, duration_s: float
) -> None:
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


def probe_step_duration(args: argparse.Namespace) -> list[DurationResult]:
  """Run all duration cohorts concurrently and aggregate their outcomes."""
  deltas = tuple(args.delta or (-0.4, -0.2, 0.0, 0.2, 0.4))
  if 0.0 not in deltas:
    raise ValueError("the sweep requires a zero-delta baseline")
  num_envs = len(deltas) * args.envs_per_delta
  cfg = _make_env_cfg(
    "position",
    num_envs=num_envs,
    num_workers=min(args.num_workers, num_envs),
    episode_length_s=args.seconds,
  )
  cfg.auto_reset = False
  cfg.seed = args.seed
  cfg.events["reset_base"].params["pose_range"] = {}
  cfg.events["encoder_bias"].params["bias_range"] = (0.0, 0.0)
  cfg.events["push_robot"].params["enabled"] = False
  action_cfg = cast(McRtcResidualActionCfg, cfg.actions["mc_rtc_residual"])
  action_cfg.datastore_scalar_commands = (STEP_TIME_CALLBACKS,)
  env = ManagerBasedRlEnv(cfg, device=args.device)
  step_dt = env.step_dt
  term = env.action_manager.get_term("mc_rtc_residual")
  if not isinstance(term, McRtcResidualActionBase):
    raise TypeError(f"unexpected action term {type(term).__name__}")
  sensors = mdp._ZmpSensors(env, mdp.GROUND_CONTACT_SENSORS, "robot")
  delta_by_env = torch.tensor(deltas, device=env.device).repeat_interleave(
    args.envs_per_delta
  )
  action = torch.zeros(num_envs, env.action_manager.total_action_dim, device=env.device)
  alive = torch.ones(num_envs, dtype=torch.bool, device=env.device)
  elapsed = torch.zeros(num_envs, dtype=torch.long, device=env.device)
  hazards = torch.zeros_like(alive)
  workers = torch.zeros_like(alive)
  error_sum = torch.zeros(num_envs, device=env.device, dtype=torch.float64)
  error_count = torch.zeros_like(error_sum)
  applied_sum = torch.zeros_like(error_sum)
  applied_count = torch.zeros_like(error_sum)
  restore_error = torch.zeros(num_envs, device=env.device)
  command_baseline = torch.zeros(num_envs, device=env.device)
  was_commanded = torch.zeros_like(alive)
  try:
    env.reset()
    impulse_step = round(args.warmup_s / step_dt)
    impulse_fired = False
    while bool(alive.any()):
      if not impulse_fired and int(elapsed.max()) >= impulse_step:
        apply_fixed_impulse(env, args.push_speed, args.push_height, args.push_duration)
        impulse_fired = True

      gate = term.last_gate.clone()
      requested = delta_by_env * gate
      command_active = alive & (gate > 0.0) & (delta_by_env != 0.0)
      term.set_datastore_scalar_delta(STEP_TIME_GETTER, command_active, requested)
      elapsed[alive] += 1
      _, _, terminated, time_outs, _ = env.step(action)

      error, grounded = dcm_error_vector(env, sensors, term)
      age = mdp.steps_since_push(env)
      recovery = (
        alive & grounded & (age >= 1) & (age <= round(args.recovery_s / step_dt))
      )
      error_sum[recovery] += torch.linalg.vector_norm(error[recovery], dim=1).double()
      error_count[recovery] += 1.0

      enabled = alive & (term.last_gate > 0.0) & (delta_by_env != 0.0)
      was_commanded |= enabled
      command_baseline[enabled] = term.controller_scalar_baseline(STEP_TIME_GETTER)[
        enabled
      ]
      applied = term.controller_scalar(STEP_TIME_GETTER) - command_baseline
      applied_sum[enabled] += applied[enabled].double().abs()
      applied_count[enabled] += 1.0
      restored = (
        alive
        & was_commanded
        & (delta_by_env != 0.0)
        & (term.last_gate == 0.0)
        & (age > round((args.recovery_s + 0.25) / step_dt))
      )
      restore_error[restored] = torch.maximum(
        restore_error[restored], applied[restored].abs()
      )

      worker = env.termination_manager.get_term("controller_worker_failed")
      done = alive & (terminated | time_outs)
      hazards |= done & terminated & ~worker
      workers |= done & worker
      alive &= ~done

    cohort_errors = {
      delta: float(
        error_sum[delta_by_env == delta].sum()
        / error_count[delta_by_env == delta].sum().clamp(min=1.0)
      )
      for delta in deltas
    }
    baseline_error = cohort_errors[0.0]
    results = []
    for delta in deltas:
      cohort = delta_by_env == delta
      applied = applied_sum[cohort].sum() / applied_count[cohort].sum().clamp(min=1.0)
      results.append(
        DurationResult(
          delta_s=delta,
          envs=int(cohort.sum()),
          hazards=int(hazards[cohort].sum()),
          worker_failures=int(workers[cohort].sum()),
          mean_end_s=float(elapsed[cohort].float().mean()) * step_dt,
          recovery_dcm_error=cohort_errors[delta],
          improvement_pct=100.0
          * (baseline_error - cohort_errors[delta])
          / baseline_error,
          mean_abs_applied_delta_s=float(applied),
          max_restore_error_s=float(restore_error[cohort].max()),
        )
      )
  finally:
    term.close()
    env.close()
    del env
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
  return results


def main() -> None:
  """Parse arguments, run the sweep, and print CSV."""
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--delta", action="append", type=float)
  parser.add_argument("--envs-per-delta", type=int, default=2)
  parser.add_argument("--num-workers", type=int, default=8)
  parser.add_argument("--seconds", type=float, default=22.0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--warmup-s", type=float, default=10.0)
  parser.add_argument("--recovery-s", type=float, default=2.0)
  parser.add_argument("--push-speed", type=float, default=0.35)
  parser.add_argument("--push-height", type=float, default=0.20)
  parser.add_argument("--push-duration", type=float, default=0.10)
  args = parser.parse_args()
  print(
    "delta_s,envs,hazards,worker_failures,mean_end_s,recovery_dcm_error,"
    "improvement_pct,mean_abs_applied_delta_s,max_restore_error_s",
    flush=True,
  )
  for result in probe_step_duration(args):
    print(
      f"{result.delta_s:+.3f},{result.envs},{result.hazards},"
      f"{result.worker_failures},{result.mean_end_s:.3f},"
      f"{result.recovery_dcm_error:.6f},{result.improvement_pct:+.3f},"
      f"{result.mean_abs_applied_delta_s:.6f},{result.max_restore_error_s:.9f}",
      flush=True,
    )


if __name__ == "__main__":
  main()
