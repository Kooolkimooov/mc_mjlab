"""Map zero-residual walking stability against compiled whole-robot COM error."""

from __future__ import annotations

import argparse
import gc
import statistics

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg

from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase
from mc_mjlab.robots import mc_rtc_robot_configuration as mc_rtc
from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import _make_env_cfg

AXES = {"x": 0, "y": 1, "z": 2}
DEFAULT_OFFSETS_MM = (-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0)


def shift_compiled_com(cfg: ManagerBasedRlEnvCfg, axis: str, offset_mm: float) -> float:
  """Shift torso inertia so the initial aggregate COM moves by ``offset_mm``."""
  robot_cfg = cfg.scene.entities["robot"]
  base_spec_fn = robot_cfg.spec_fn
  root_name = mc_rtc.get_root_body(cfg.actions["mc_rtc_residual"].mc_rtc_robot_name)
  torso_shift_mm = 0.0

  def shifted_spec() -> mujoco.MjSpec:
    nonlocal torso_shift_mm
    spec = base_spec_fn()
    root = spec.body(root_name)
    total_mass = sum(float(body.mass) for body in spec.bodies)
    torso_shift_mm = offset_mm * total_mass / float(root.mass)
    ipos = list(root.ipos)
    ipos[AXES[axis]] += torso_shift_mm / 1000.0
    root.ipos = ipos
    return spec

  robot_cfg.spec_fn = shifted_spec
  shifted_spec()
  return torso_shift_mm


def run_offset(
  axis: str, offset_mm: float, args: argparse.Namespace
) -> dict[str, float]:
  """Run one compiled COM mismatch cohort without policy action or pushes."""
  cfg = _make_env_cfg(
    "position",
    num_envs=args.num_envs,
    num_workers=args.num_workers,
    disturbance="none",
    console_output="none",
  )
  torso_shift_mm = shift_compiled_com(cfg, axis, offset_mm)
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
  active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  elapsed = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
  hazards = torch.zeros_like(active)
  workers = torch.zeros_like(active)
  alpha_medians: list[float] = []
  step_dt = env.step_dt
  try:
    env.reset()
    while bool(active.any()):
      elapsed[active] += 1
      _, _, terminated, time_outs, _ = env.step(zero)
      alpha = action.controller_reference("alpha")[active]
      if action.residual_ids is not None:
        alpha = alpha[:, action.residual_ids]
      alpha_medians.append(float(alpha.abs().median()))
      worker = env.termination_manager.get_term("controller_worker_failed")
      done = terminated | time_outs
      hazards |= active & terminated & ~worker
      workers |= active & worker
      active &= ~done
  finally:
    action.close()
    env.close()
    del env
    gc.collect()
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
  return {
    "torso_shift_mm": torso_shift_mm,
    "survival": 1.0 - float(hazards.float().mean()),
    "worker_failure": float(workers.float().mean()),
    "mean_end_s": float(elapsed.float().mean()) * step_dt,
    "alpha_median": statistics.median(alpha_medians),
  }


def map_com_stability() -> None:
  """Sweep requested axes and aggregate COM offsets."""
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--axis", action="append", choices=AXES)
  parser.add_argument("--offset-mm", action="append", type=float)
  parser.add_argument("--num-envs", type=int, default=4)
  parser.add_argument("--num-workers", type=int, default=4)
  parser.add_argument("--seconds", type=float, default=12.0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  axes = args.axis or list(AXES)
  offsets = args.offset_mm or list(DEFAULT_OFFSETS_MM)
  print(
    "axis,com_offset_mm,torso_offset_mm,survival,worker_failure,"
    "mean_end_s,alpha_median",
    flush=True,
  )
  for axis in axes:
    for offset_mm in offsets:
      result = run_offset(axis, offset_mm, args)
      print(
        f"{axis},{offset_mm:.3f},{result['torso_shift_mm']:.3f},"
        f"{result['survival']:.3f},{result['worker_failure']:.3f},"
        f"{result['mean_end_s']:.3f},{result['alpha_median']:.6f}",
        flush=True,
      )


if __name__ == "__main__":
  map_com_stability()
