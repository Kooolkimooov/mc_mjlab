"""Check two PPO updates during manipulation and a fresh-runner checkpoint reload."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch
from evaluation.rollout import managed_env
from mjlab.rl import RslRlVecEnvWrapper

from mc_mjlab.bridge.config import get_main_robot_name
from mc_mjlab.rl.runner import McRtcResidualOnPolicyRunner
from mc_mjlab.tasks.locomanip import FEEDBACK_TASK_IDS
from mc_mjlab.tasks.locomanip.locomanip_feedback_env_cfg import (
  make_locomanip_feedback_env_cfg,
)
from mc_mjlab.tasks.locomanip.locomanip_ppo_cfg import locomanip_ppo_cfg
from mc_mjlab.tasks.locomanip.mdp.accessors import holding
from mc_mjlab.tasks.locomanip.profiles import PROFILES

ROBOTS = {get_main_robot_name(p.mc_rtc_yaml): p for p in PROFILES}


def finite_gradient(gradient: torch.Tensor) -> torch.Tensor:
  """Fail on nonfinite gradients before the optimizer can consume them."""
  if not torch.isfinite(gradient).all():
    raise AssertionError("nonfinite PPO gradient")
  return gradient


def run(args: argparse.Namespace) -> None:
  """Warm up at zero feedback, train briefly and verify a serialized policy."""
  torch.manual_seed(42)
  cfg = make_locomanip_feedback_env_cfg(
    profile=ROBOTS[args.robot],
    num_envs=8,
    num_workers=4,
    cart_mass_range_kg=(10.0, 10.0),
  )
  cfg.seed = 42
  cfg.events.pop("encoder_bias")
  agent = asdict(locomanip_ppo_cfg(experiment_name=FEEDBACK_TASK_IDS[args.robot]))
  agent.update(logger="tensorboard", upload_model=False, check_for_nan=True)
  output = args.output / args.robot
  output.mkdir(parents=True, exist_ok=True)

  with managed_env(cfg, args.device) as env:
    wrapped = RslRlVecEnvWrapper(env)
    zero = torch.zeros(env.num_envs, 6, device=env.device)
    for _ in range(math.ceil(20 / env.step_dt)):
      if holding(env).bool().all():
        break
      env.step(zero)
    else:
      raise AssertionError("not every smoke-test row reached Hold within 20 s")

    runner = McRtcResidualOnPolicyRunner(
      wrapped, deepcopy(agent), str(output), args.device
    )
    losses = []
    update = runner.alg.update

    def checked_update() -> dict[str, float]:
      result = update()
      if not all(math.isfinite(value) for value in result.values()):
        raise AssertionError(f"nonfinite PPO losses: {result}")
      losses.append(result)
      return result

    parameters = [*runner.alg.actor.parameters(), *runner.alg.critic.parameters()]
    handles = [p.register_hook(finite_gradient) for p in parameters if p.requires_grad]
    try:
      with patch.object(runner.alg, "update", checked_update):
        runner.learn(num_learning_iterations=2)
    finally:
      for handle in handles:
        handle.remove()
    if not all(torch.isfinite(p).all() for p in parameters):
      raise AssertionError("nonfinite trained parameters")

    checkpoint = output / "smoke.pt"
    runner.save(str(checkpoint))
    restored = McRtcResidualOnPolicyRunner(wrapped, deepcopy(agent), device=args.device)
    restored.load(str(checkpoint))
    obs = wrapped.get_observations()
    with torch.inference_mode():
      expected = runner.get_inference_policy(device=args.device)(obs)
      actual = restored.get_inference_policy(device=args.device)(obs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    result = {
      "robot": args.robot,
      "updates": len(losses),
      "losses": losses,
      "action_dimension": env.action_manager.total_action_dim,
      "actor_observations": env.observation_manager.group_obs_dim["actor"],
      "critic_observations": env.observation_manager.group_obs_dim["critic"],
      "checkpoint_reload_exact": True,
      "seed": 42,
      "cart_mass_kg": 10,
      "encoder_bias": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main() -> None:
  """Select the robot and an isolated directory for smoke-test artifacts."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--robot", choices=sorted(ROBOTS), default="HRP5P")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument(
    "--output", type=Path, default=Path("logs/locomanip/feedback-smoke")
  )
  run(parser.parse_args())


if __name__ == "__main__":
  main()
