"""Measure zero-residual walking through both native action modes."""

import argparse
import json

import torch
from mjlab.envs import ManagerBasedRlEnv

from mc_mjlab.tasks.zero_residual.zero_residual_env_cfg import (
  zero_residual_position_env_cfg,
  zero_residual_torque_env_cfg,
)


def measure(mode: str, seconds: float, device: str) -> None:
  """Run a bounded demo and report root height, displacement and reference speed."""
  builder = (
    zero_residual_position_env_cfg
    if mode == "position"
    else zero_residual_torque_env_cfg
  )
  cfg = builder(play=True)
  cfg.scene.num_envs = 1
  cfg.actions["robot_joints"].num_workers = 1
  cfg.actions["robot_joints"].print_residual_every = 0
  env = ManagerBasedRlEnv(cfg, device=device)
  action = env.action_manager.get_term("robot_joints")
  try:
    env.reset()
    asset = env.scene["robot"]
    start = asset.data.root_link_pos_w[0, :2].clone()
    height, speed = [], []
    failures = 0
    with torch.inference_mode():
      for step in range(round(seconds / env.step_dt)):
        env.step(torch.zeros(1, action.action_dim, device=device))
        failures += int(
          action.controller_failed.any() or action.controller_worker_failed.any()
        )
        if step * env.step_dt > 2:
          height.append(float(asset.data.root_link_pos_w[0, 2]))
          speed.append(float(action.controller_reference("alpha")[0].abs().median()))
    displacement = float(
      torch.linalg.vector_norm(asset.data.root_link_pos_w[0, :2] - start)
    )
    result = dict(
      mode=mode,
      samples=len(height),
      seconds=seconds,
      min_z=min(height),
      max_z=max(height),
      displacement=displacement,
      median_abs_alpha=float(torch.tensor(speed).median()),
      failed_steps=failures,
    )
    print(json.dumps(result), flush=True)
    assert min(height) > 0.65 and displacement > 0.1 and failures == 0, result
  finally:
    action.close()
    env.close()


def main() -> None:
  """Run the selected live actuator mode."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--mode", choices=("position", "torque"), required=True)
  parser.add_argument("--seconds", type=float, default=12)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  measure(args.mode, args.seconds, args.device)


if __name__ == "__main__":
  main()
