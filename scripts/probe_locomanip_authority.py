"""Measure position or hand-force residual authority against matched zero controls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from evaluation.locomanip_authority import (
  CHANNELS,
  AuthorityTrace,
  action_matrix,
  conditions,
)
from evaluation.rollout import managed_env

from mc_mjlab.bridge.config import get_main_robot_name
from mc_mjlab.tasks.locomanip.locomanip_feedback_env_cfg import (
  make_locomanip_feedback_env_cfg,
)
from mc_mjlab.tasks.locomanip.locomanip_residual_env_cfg import (
  make_locomanip_residual_env_cfg,
)
from mc_mjlab.tasks.locomanip.profiles import PROFILES

ROBOTS = {get_main_robot_name(p.mc_rtc_yaml): p for p in PROFILES}


def run(args: argparse.Namespace) -> dict[str, Any]:
  """Measure fixed-payload conditions with equal encoders and first-episode records."""
  default_levels = "0.1,0.5,1.0" if args.task == "feedback" else "1.0,-1.0"
  levels = [float(level) for level in (args.levels or default_levels).split(",")]
  if any(not math.isfinite(level) or abs(level) > 1 for level in levels):
    raise ValueError("levels must be finite normalized actions in [-1, 1]")
  rows = conditions(args.task, levels, args.channel)
  builder = (
    make_locomanip_feedback_env_cfg
    if args.task == "feedback"
    else make_locomanip_residual_env_cfg
  )
  extra = {"force_scale": args.force_scale} if args.task == "feedback" else {}
  cfg = builder(
    profile=ROBOTS[args.robot],
    num_envs=len(rows),
    num_workers=min(len(rows), args.num_workers),
    cart_mass_range_kg=(args.mass, args.mass),
    **extra,
  )
  cfg.seed = 42
  cfg.events.pop("encoder_bias")
  cfg.episode_length_s = args.seconds + 1.0
  cfg.auto_reset = False

  with managed_env(cfg, args.device) as env:
    env.reset()
    actions = action_matrix(env, rows)
    trace = AuthorityTrace(env)
    for _ in range(math.ceil(args.seconds / env.step_dt)):
      _, _, terminated, truncated, _ = env.step(actions)
      trace.update(terminated, truncated)
      actions[trace.finished] = 0
      if trace.finished.all():
        break
    results = trace.results(rows)

  return {
    "robot": args.robot,
    "task": args.task,
    "mass_kg": args.mass,
    "seconds": args.seconds,
    "force_scale_n": args.force_scale if args.task == "feedback" else None,
    "residual_unit": "N" if args.task == "feedback" else "rad",
    "seed": 42,
    "encoder_bias": False,
    "arms": results,
  }


def main() -> None:
  """Report authority by channel, amplitude, robot and payload."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=("position", "feedback"), default="position")
  parser.add_argument("--robot", choices=sorted(ROBOTS), default="HRP5P")
  parser.add_argument("--channel", choices=("all", *CHANNELS), default="all")
  parser.add_argument("--levels")
  parser.add_argument("--force-scale", type=float, default=50.0)
  parser.add_argument("--mass", type=float, default=10.0)
  parser.add_argument("--seconds", type=float, default=16.0)
  parser.add_argument("--num-workers", type=int, default=4)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  if args.seconds <= 0 or args.mass <= 0 or args.num_workers < 1:
    parser.error("seconds, mass and num-workers must be positive")
  result = run(args)
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
