"""Run one cart push per hand-wrench blend setting -- docs/locomanip.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from evaluation.rollout import managed_env

from mc_mjlab.tasks.locomanip.locomanip_env_cfg import MC_RTC_YAML
from mc_mjlab.tasks.locomanip.locomanip_residual_env_cfg import (
  make_locomanip_residual_env_cfg,
)
from mc_mjlab.tasks.locomanip.mdp import accessors, metrics, observations

#: The push state's blend settings, as `HandWrenchAdaptation` understands them.
#: The shipped projection drops HRP5P at 100 kg. docs/locomanip.md
ARMS: dict[str, dict[str, Any]] = {
  "shipped": {"enable": True, "alpha": 0.02},
  "off": {"enable": False},
  "push-axis": {
    "enable": True,
    "alpha": 0.02,
    "forceProjection": [1.0, 0.0, 0.0],
    "momentProjection": [0.0, 0.0, 0.0],
  },
}

PUSH_STATE = "LMC::DemoPush"


def configure(arm: str, log_dir: Path | None, scratch: Path) -> Path:
  """Write an mc_rtc config for this arm, with controller logging when asked."""
  config = yaml.safe_load(MC_RTC_YAML.read_text())
  state = config["OverwriteConfigList"]["MjlabCartDemo"]["DemoFSM"]["states"][
    PUSH_STATE
  ]
  state.setdefault("configs", {})["HandWrenchAdaptation"] = ARMS[arm]
  if log_dir is not None:
    log_dir.mkdir(parents=True, exist_ok=True)
    config["Log"] = True
    config["LogDirectory"] = str(log_dir)
    config["LogTemplate"] = f"mjlab-{arm}"
  path = scratch / f"mc_rtc_locomanip_{arm}.yaml"
  path.write_text(yaml.safe_dump(config, sort_keys=False))
  return path


def run(arm: str, args: argparse.Namespace) -> dict:
  """Step one zero-residual episode and report where the cart and robot ended."""
  cfg = make_locomanip_residual_env_cfg(
    num_envs=1,
    num_workers=1,
    cart_mass_range_kg=(args.mass, args.mass),
    mc_rtc_yaml=configure(arm, args.log_dir, args.scratch),
  )
  cfg.auto_reset = False
  cfg.episode_length_s = args.seconds + 4.0

  with managed_env(cfg, args.device) as env:
    env.reset()
    action = torch.zeros(
      (env.num_envs, env.action_manager.total_action_dim), device=env.device
    )
    lowest = env.scene["robot"].data.root_link_pos_w[0, 2].clone()
    released = False
    steps = 0
    for _ in range(int(args.seconds / env.step_dt)):
      _, _, terminated, truncated, _ = env.step(action)
      steps += 1
      lowest = torch.minimum(lowest, env.scene["robot"].data.root_link_pos_w[0, 2])
      released = released or bool(metrics.hands_released(env)[0])
      if bool((terminated | truncated).any()):
        break
    return {
      "arm": arm,
      "mass_kg": args.mass,
      "seconds": round(steps * env.step_dt, 2),
      "complete": bool(metrics.task_complete(env)[0]),
      "released": released,
      "object_position_error_m": float(metrics.object_position_error(env)[0]),
      "min_base_z_m": float(lowest),
      "phases": [int(p) for p in observations.manipulation_phase(env)[0]],
      "controller_failed": bool(accessors.residual_action(env).controller_failed[0]),
    }


def main() -> None:
  """Run every requested arm at one cart mass and print a row each."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--mass", type=float, default=100.0)
  parser.add_argument("--arms", default=",".join(ARMS))
  parser.add_argument("--seconds", type=float, default=56.0)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--log-dir",
    type=Path,
    default=None,
    help="write mc_rtc binary logs here; without it the controller logs nothing",
  )
  parser.add_argument("--scratch", type=Path, default=Path("logs/locomanip"))
  args = parser.parse_args()
  args.scratch.mkdir(parents=True, exist_ok=True)

  print(
    f"\n{'arm':>10} {'s':>6} {'complete':>9} {'released':>9} {'err m':>7} {'min z':>7}"
  )
  for arm in (name.strip() for name in args.arms.split(",") if name.strip()):
    row = run(arm, args)
    print(
      f"{row['arm']:>10} {row['seconds']:6.1f} {str(row['complete']):>9} "
      f"{str(row['released']):>9} {row['object_position_error_m']:7.3f} "
      f"{row['min_base_z_m']:7.3f}",
      flush=True,
    )
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
  main()
