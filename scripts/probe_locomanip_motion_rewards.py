"""Validate motion reward weights on matched multi-mass controller rollouts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, cast

import mjlab  # noqa: F401
import torch
from evaluation.locomanip_inputs import (
  assert_payload_masses,
  condition_id,
  fixed_payloads,
  matched_encoder_bias,
)
from evaluation.rollout import managed_env, reset_done
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from probe_locomanip_motion import _frames

from mc_mjlab.actions.locomanip_motion_action import (
  MOTION_STATE_GETTER,
  LocomanipMotionAction,
)
from mc_mjlab.rl.controller_provenance import collect_controller_provenance
from mc_mjlab.tasks.locomanip.cart import cart_nominal_mass_kg
from mc_mjlab.tasks.locomanip.mdp import accessors
from mc_mjlab.tasks.locomanip_motion import TASK_IDS

if TYPE_CHECKING:
  from mjlab.rl import RslRlOnPolicyRunnerCfg

  from mc_mjlab.actions.locomanip_motion_action import LocomanipMotionActionCfg

ARMS = ("baseline", "slow", "hold_full", "hold_half", "hold_forever")


def main() -> None:
  """Share one controller batch across explicit payload and encoder conditions."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--masses", type=float, nargs="+", required=True)
  parser.add_argument("--seeds", type=int, nargs="+", required=True)
  parser.add_argument("--simulation-seed", type=int, default=42)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  if len(set(args.masses)) != len(args.masses) or any(
    not math.isfinite(m) or m <= 0 for m in args.masses
  ):
    parser.error("masses must be unique, finite and positive")
  if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
    parser.error("condition seeds must be unique and nonnegative")
  rows = [
    {
      "mass_kg": mass,
      "seed": seed,
      "arm": arm,
      "condition_id": condition_id(seed, mass, 0),
    }
    for seed in args.seeds
    for mass in args.masses
    for arm in ARMS
  ]
  task = TASK_IDS["HRP5P"]
  cfg = load_env_cfg(task)
  cfg.seed = args.simulation_seed
  cfg.scene.num_envs = len(rows)
  cfg.auto_reset = False
  action_cfg = cast("LocomanipMotionActionCfg", cfg.actions[accessors.ACTION_NAME])
  action_cfg.num_workers = min(16, len(rows))
  payload = cfg.events["cart_payload"]
  cfg.events["cart_payload"] = EventTermCfg(
    func=fixed_payloads,
    mode="reset",
    params={
      "rows": rows,
      "nominal_mass": cart_nominal_mass_kg(),
      "asset_cfg": payload.params["asset_cfg"],
    },
  )
  cfg.events["encoder_bias"] = EventTermCfg(
    func=matched_encoder_bias, mode="startup", params={"rows": rows, "enabled": True}
  )
  args.output.mkdir(parents=True, exist_ok=True)
  paths = {
    row["condition_id"]: args.output / f"seed-{row['seed']}-mass-{row['mass_kg']:g}"
    for row in rows
  }
  streams = {}
  results: list[dict | None] = [None] * len(rows)
  triggered: list[float | None] = [None] * len(rows)
  held_since: list[float | None] = [None] * len(rows)
  with managed_env(cfg, args.device) as env:
    env.reset()
    assert_payload_masses(accessors.object_mass_kg(env), rows)
    masses = accessors.object_mass_kg(env).cpu().tolist()
    biases = env.scene[accessors.ROBOT_ENTITY].data.encoder_bias.cpu().tolist()
    term = cast(LocomanipMotionAction, accessors.residual_action(env))
    provenance = collect_controller_provenance(RslRlVecEnvWrapper(env))
    weights = {
      name: env.reward_manager.get_term_cfg(name).weight
      for name in env.reward_manager.active_terms
    }
    gamma = cast("RslRlOnPolicyRunnerCfg", load_rl_cfg(task)).algorithm.gamma
    start = (
      env.scene[accessors.OBJECT_ENTITY].data.root_link_pos_w[:, 0]
      - env.scene.env_origins[:, 0]
    ).clone()
    actions = torch.zeros(
      len(rows), env.action_manager.total_action_dim, device=env.device
    )
    try:
      for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        streams[key] = (path / "trace.jsonl").open("w")
      for step in range(env.max_episode_length):
        time = step * env.step_dt
        hold = accessors.holding(env).bool().cpu().tolist()
        progress = (
          (
            term.datastore_vector_output(accessors.OBJECT_REFERENCE_POSITION)[:, 0]
            - start
          )
          .cpu()
          .tolist()
        )
        states = term.datastore_scalar_output(MOTION_STATE_GETTER).cpu().tolist()
        for index, row in enumerate(rows):
          if results[index] is not None or row["arm"] == "baseline":
            actions[index, -1] = 0
            continue
          if triggered[index] is None and hold[index] and progress[index] >= 0.15:
            triggered[index] = time
          if triggered[index] is None:
            continue
          if row["arm"] == "slow":
            actions[index, -1] = 0.5
            continue
          if states[index] == 2 and held_since[index] is None:
            held_since[index] = time
          began = held_since[index]
          release = began is not None and time - began >= 4.0
          actions[index, -1] = (
            0.5
            if release and row["arm"] == "hold_half"
            else 0.0
            if release and row["arm"] == "hold_full"
            else 1.0
          )
        _, rewards, terminated, truncated, _ = env.step(actions)
        frames = _frames(env, term, time + env.step_dt)
        rates = env.reward_manager._step_reward.cpu().tolist()
        reward_values = rewards.cpu().tolist()
        ends = terminated.cpu().tolist()
        truncations = truncated.cpu().tolist()
        done = (terminated | truncated).cpu().tolist()
        for index, frame in enumerate(frames):
          if results[index] is not None:
            continue
          frame.update(
            **rows[index],
            trigger_time_s=triggered[index],
            reward=reward_values[index],
            reward_rates=dict(zip(weights, rates[index], strict=True)),
            terminated=ends[index],
            truncated=truncations[index],
          )
          streams[rows[index]["condition_id"]].write(json.dumps(frame) + "\n")
          if done[index]:
            results[index] = frame
        if all(result is not None for result in results):
          break
        if any(done):
          reset_done(env, (terminated | truncated).nonzero().flatten())
      if any(result is None for result in results):
        raise RuntimeError("motion reward probe lacks a terminal observation")
      for key, path in paths.items():
        indices = [i for i, row in enumerate(rows) if row["condition_id"] == key]
        first = indices[0]
        for index in indices:
          if biases[index] != biases[first]:
            raise AssertionError("encoder inputs differ within a paired condition")
        cohort_results = [cast(dict, results[i]) for i in indices]
        report = {
          **rows[first],
          "simulation_seed": args.simulation_seed,
          "encoder_bias": True,
          "encoder_bias_rad": biases[first],
          "actual_mass_kg": [masses[i] for i in indices],
          "episode_cap_s": cfg.episode_length_s,
          "step_dt": env.step_dt,
          "gamma": gamma,
          "motion_decision_period_s": action_cfg.motion_decision_period_s,
          "requested_hold_s": 4.0,
          "reward_weights": weights,
          "provenance": provenance,
          "results": cohort_results,
        }
        report.pop("arm")
        (path / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
          json.dumps(
            {
              "condition_id": key,
              "outcomes": [
                (
                  r["arm"],
                  r["time_s"],
                  bool(r["complete"]),
                  bool(r["fell_over"] or r["collapsed"] or r["controller_failed"]),
                )
                for r in cohort_results
              ],
            }
          ),
          flush=True,
        )
    finally:
      for stream in streams.values():
        stream.close()


if __name__ == "__main__":
  main()
