"""Measure what a constant residual at full clip does to the cart -- docs/locomanip.md."""

from __future__ import annotations

import argparse
import json
import math

import torch
from evaluation.rollout import managed_env

from mc_mjlab.tasks.locomanip.locomanip_residual_env_cfg import (
  make_locomanip_residual_env_cfg,
)
from mc_mjlab.tasks.locomanip.mdp import accessors, observations


def run(args: argparse.Namespace) -> dict:
  """Step one zero-residual arm beside constant-residual arms, on one cart mass."""
  levels = [0.0] + [float(level) for level in args.levels.split(",") if level.strip()]
  cfg = make_locomanip_residual_env_cfg(
    num_envs=len(levels),
    num_workers=min(len(levels), args.num_workers),
    cart_mass_range_kg=(args.mass, args.mass),
  )
  cfg.episode_length_s = args.seconds + 1.0
  cfg.auto_reset = False

  with managed_env(cfg, args.device) as env:
    term = accessors.residual_action(env)
    action = (
      torch.tensor(levels, device=env.device)
      .unsqueeze(-1)
      .expand(len(levels), env.action_manager.total_action_dim)
    )
    env.reset()
    start = env.scene[accessors.OBJECT_ENTITY].data.root_link_pos_w[:, 0].clone()
    lowest = env.scene["robot"].data.root_link_pos_w[:, 2].clone()
    held = torch.zeros(len(levels), device=env.device)

    for _ in range(math.ceil(args.seconds / env.step_dt)):
      env.step(action.contiguous())
      height = env.scene["robot"].data.root_link_pos_w[:, 2]
      lowest = torch.minimum(lowest, height)
      held += accessors.holding(env) * env.step_dt

    cart = env.scene[accessors.OBJECT_ENTITY].data.root_link_pos_w[:, 0] - start
    executed = term.executed_physical_action.abs().mean(dim=1)
    phases = observations.manipulation_phase(env)
    rows = [
      {
        "level": level,
        "cart_travel_m": float(cart[i]),
        "mean_residual_rad": float(executed[i]),
        "held_s": float(held[i]),
        "min_base_z_m": float(lowest[i]),
        "phases": [int(p) for p in phases[i]],
        "controller_failed": bool(term.controller_failed[i]),
      }
      for i, level in enumerate(levels)
    ]

  return {"mass_kg": args.mass, "seconds": args.seconds, "arms": rows}


def main() -> None:
  """Print one row per residual level, and the travel each one bought or cost."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--levels", default="1.0,-1.0")
  parser.add_argument("--mass", type=float, default=10.0)
  parser.add_argument("--seconds", type=float, default=16.0)
  parser.add_argument("--num-workers", type=int, default=4)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()

  result = run(args)
  baseline = result["arms"][0]["cart_travel_m"]
  print(f"\ncart {result['mass_kg']:g} kg, {result['seconds']:g} s")
  print(
    f"  {'level':>7} {'cart (m)':>10} {'vs zero':>10} {'residual':>10} {'held s':>8} {'min z':>7}"
  )
  for row in result["arms"]:
    print(
      f"  {row['level']:7.2f} {row['cart_travel_m']:10.4f} "
      f"{row['cart_travel_m'] - baseline:+10.4f} {row['mean_residual_rad']:10.5f} "
      f"{row['held_s']:8.2f} {row['min_base_z_m']:7.3f}"
    )
  print(json.dumps(result))


if __name__ == "__main__":
  main()
