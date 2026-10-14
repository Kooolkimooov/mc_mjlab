"""Verify multi-rate PPO, native braking delivery and an exact checkpoint reload."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import torch
from evaluation.rollout import managed_env
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from mc_mjlab.actions.locomanip_motion_action import (
  MOTION_COUNT_GETTER,
  LocomanipMotionAction,
)
from mc_mjlab.rl.runner import McRtcResidualOnPolicyRunner
from mc_mjlab.tasks.locomanip.cart import cart_nominal_mass_kg
from mc_mjlab.tasks.locomanip.mdp import accessors
from mc_mjlab.tasks.locomanip.mdp.events import mass_alpha_range
from mc_mjlab.tasks.locomanip_motion import TASK_IDS

if TYPE_CHECKING:
  from mc_mjlab.actions.locomanip_motion_action import LocomanipMotionActionCfg


def finite_gradient(gradient: torch.Tensor) -> torch.Tensor:
  """Reject nonfinite gradients before the optimizer consumes them."""
  if not torch.isfinite(gradient).all():
    raise AssertionError("nonfinite PPO gradient")
  return gradient


def run(output: Path, device: str) -> None:
  """Exercise decisions during Hold with real controllers and actual PPO storage."""
  torch.manual_seed(42)
  task = TASK_IDS["HRP5P"]
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = 16
  cfg.seed = 42
  action_cfg = cast("LocomanipMotionActionCfg", cfg.actions[accessors.ACTION_NAME])
  action_cfg.num_workers = 8
  cfg.events["cart_payload"].params["alpha_range"] = mass_alpha_range(
    (10.0, 10.0), cart_nominal_mass_kg()
  )
  agent = asdict(load_rl_cfg(task))
  agent.update(logger="tensorboard", upload_model=False, check_for_nan=True)
  output.mkdir(parents=True, exist_ok=True)
  losses: list[dict] = []
  selections = torch.zeros(16, dtype=torch.long, device=device)
  starts = torch.zeros(16, device=device)
  chosen = torch.zeros(16, device=device)
  pending: dict[str, torch.Tensor] = {}
  with managed_env(cfg, device) as env:
    wrapped = RslRlVecEnvWrapper(env)
    term = cast(LocomanipMotionAction, accessors.residual_action(env))
    zeros = torch.zeros(16, env.action_manager.total_action_dim, device=device)
    for _ in range(math.ceil(20 / env.step_dt)):
      if accessors.holding(env).bool().all():
        break
      env.step(zeros)
    else:
      raise AssertionError("not all smoke rows reached Hold within 20 seconds")
    torch.testing.assert_close(
      accessors.object_mass_kg(env), torch.full((16,), 10.0, device=device)
    )

    runner = McRtcResidualOnPolicyRunner(wrapped, deepcopy(agent), str(output), device)
    act = runner.alg.act
    process = runner.alg.process_env_step
    update = runner.alg.update

    def checked_act(obs: Any) -> torch.Tensor:
      """Match the recorded pre-action mask to the action term's decision clock."""
      due = obs["motion_decision"][:, 0].bool()
      torch.testing.assert_close(due, term.motion_decision_state[:, 0].bool())
      actions = act(obs)
      chosen.copy_(torch.where(due, actions[:, -1], chosen))
      pending["expected"] = chosen.clone()
      pending["due"] = due.clone()
      selections.add_(due)
      params = runner.alg.transition.distribution_params
      assert params is not None
      torch.testing.assert_close(params[2][:, 0].bool(), due)
      assert (actions[~due, -1] == 0).all()
      return actions

    def checked_process(
      obs: Any, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> None:
      """Check actual latched requests while collecting native braking starts."""
      torch.testing.assert_close(
        term.motion_decision_state[~dones.bool(), 2], pending["expected"][~dones.bool()]
      )
      chosen.masked_fill_(dones.bool(), 0.0)
      starts.copy_(
        torch.maximum(starts, term.datastore_scalar_output(MOTION_COUNT_GETTER))
      )
      process(obs, rewards, dones, extras)

    def checked_update() -> dict[str, float]:
      """Require finite losses from the real shuffled PPO minibatches."""
      result = update()
      if not all(math.isfinite(value) for value in result.values()):
        raise AssertionError(f"nonfinite PPO losses: {result}")
      losses.append(result)
      return result

    parameters = [*runner.alg.actor.parameters(), *runner.alg.critic.parameters()]
    handles = [p.register_hook(finite_gradient) for p in parameters if p.requires_grad]
    try:
      with (
        patch.object(runner.alg, "act", checked_act),
        patch.object(runner.alg, "process_env_step", checked_process),
        patch.object(runner.alg, "update", checked_update),
      ):
        runner.learn(num_learning_iterations=4)
    finally:
      for handle in handles:
        handle.remove()
    if starts.sum() == 0:
      raise AssertionError("exploration did not reach a native braking start")
    checkpoint = output / "smoke.pt"
    runner.save(str(checkpoint))
    restored = McRtcResidualOnPolicyRunner(wrapped, deepcopy(agent), device=device)
    restored.load(str(checkpoint))
    obs = wrapped.get_observations()
    with torch.inference_mode():
      expected = runner.get_inference_policy(device=device)(obs)
      actual = restored.get_inference_policy(device=device)(obs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    result = {
      "seed": 42,
      "num_envs": 16,
      "mass_kg": 10.0,
      "encoder_bias": True,
      "updates": len(losses),
      "losses": losses,
      "motion_period_s": term.cfg.motion_decision_period_s,
      "policy_step_s": env.step_dt,
      "selections_per_row": selections.cpu().tolist(),
      "max_native_starts_per_row": starts.cpu().tolist(),
      "checkpoint_reload_exact": True,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
  """Select an isolated output directory and simulation device."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  run(args.output, args.device)


if __name__ == "__main__":
  main()
