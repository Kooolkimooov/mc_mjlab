"""Measure untouched mc_rtc feasibility for each robustness component."""

from __future__ import annotations

import argparse
import gc
import statistics

import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase
from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import _make_env_cfg

EVENT_COMPONENTS = {
  "friction": "randomize_friction",
  "pd-gains": "randomize_pd_gains",
  "strength": "randomize_strength",
}
PROFILES = (
  "nominal",
  *EVENT_COMPONENTS,
  "actor-delay",
  "actuator-delay",
  "all",
)


def select_components(cfg: ManagerBasedRlEnvCfg, profile: str) -> None:
  """Keep only the requested stage-one randomization component."""
  selected = set(EVENT_COMPONENTS) if profile == "all" else {profile}
  for component, event_name in EVENT_COMPONENTS.items():
    if component not in selected:
      cfg.events.pop(event_name)
  if profile not in ("actor-delay", "all"):
    for term in cfg.observations["actor"].terms.values():
      term.delay_max_lag = 0
  if profile not in ("actuator-delay", "all"):
    articulation = cfg.scene.entities["robot"].articulation
    assert articulation is not None
    for actuator in articulation.actuators:
      actuator.delay_max_lag = 0


def run_profile(profile: str, args: argparse.Namespace) -> dict[str, float]:
  """Run one no-residual cohort to its first termination or time limit."""
  stage = 0 if profile == "nominal" else 1
  cfg = _make_env_cfg(
    "position",
    num_envs=args.num_envs,
    num_workers=args.num_workers,
    disturbance="none",
    randomization_stage=stage,
    console_output="none",
  )
  if stage:
    select_components(cfg, profile)

  cfg.seed = args.seed
  cfg.episode_length_s = args.seconds
  cfg.events["reset_base"].params["pose_range"] = {}

  env = ManagerBasedRlEnv(cfg, device=args.device)
  action = env.action_manager.get_term("mc_rtc_residual")
  if not isinstance(action, McRtcResidualActionBase):
    raise TypeError(f"unexpected residual action type: {type(action).__name__}")

  zero = torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )
  step_dt = env.step_dt
  active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  elapsed_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  failures = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  worker_failures = torch.zeros_like(failures)
  alpha_medians: list[float] = []

  try:
    env.reset()
    while bool(active.any()):
      elapsed_steps[active] += 1
      _, _, terminated, time_outs, _ = env.step(zero)
      if bool(active.any()):
        alpha = action.controller_reference("alpha")[active]
        if action.residual_ids is not None:
          alpha = alpha[:, action.residual_ids]
        alpha = alpha.abs().median()
        alpha_medians.append(float(alpha))

      worker = env.termination_manager.get_term("controller_worker_failed")
      done = terminated | time_outs
      failures |= active & terminated & ~worker
      worker_failures |= active & worker
      active &= ~done
  finally:
    action.close()
    env.close()
    del env
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

  return {
    "survival": 1.0 - float(failures.float().mean()),
    "worker_failure": float(worker_failures.float().mean()),
    "mean_end_s": float(elapsed_steps.float().mean()) * step_dt,
    "alpha_median": statistics.median(alpha_medians),
  }


def main() -> None:
  """Run selected profiles and print one compact comparison table."""
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--profile", action="append", choices=PROFILES)
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--num-workers", type=int, default=4)
  parser.add_argument("--seconds", type=float, default=12.0)
  parser.add_argument("--min-survival", type=float, default=0.95)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  profiles = args.profile or list(PROFILES)
  print("profile,survival,worker_failure,mean_end_s,alpha_median", flush=True)
  failed: list[str] = []
  for profile in profiles:
    result = run_profile(profile, args)
    print(
      f"{profile},{result['survival']:.3f},{result['worker_failure']:.3f},"
      f"{result['mean_end_s']:.3f},{result['alpha_median']:.6f}",
      flush=True,
    )
    if result["survival"] < args.min_survival or result["worker_failure"]:
      failed.append(profile)
  if failed:
    raise SystemExit(f"baseline feasibility gate failed: {', '.join(failed)}")


if __name__ == "__main__":
  main()
