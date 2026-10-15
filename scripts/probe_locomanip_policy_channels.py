"""Separate joint and motion contributions to a locomanip checkpoint's behavior."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
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
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from probe_locomanip_motion import _frames

from mc_mjlab.actions.locomanip_motion_action import LocomanipMotionAction
from mc_mjlab.rl.controller_provenance import collect_controller_provenance
from mc_mjlab.tasks.locomanip.cart import cart_nominal_mass_kg
from mc_mjlab.tasks.locomanip.mdp import accessors
from mc_mjlab.tasks.locomanip_motion import TASK_IDS

if TYPE_CHECKING:
  from mc_mjlab.actions.locomanip_motion_action import LocomanipMotionActionCfg

ARMS = ("baseline", "policy", "joints_only", "motion_only")


def select_channels(
  actions: torch.Tensor, arms: tuple[str, ...] = ARMS
) -> torch.Tensor:
  """Keep each cohort's baseline, full policy and optional channel ablations."""
  selected = actions.clone()
  for index, arm in enumerate(arms):
    if arm == "baseline":
      selected[index :: len(arms)] = 0.0
    elif arm == "joints_only":
      selected[index :: len(arms), -1] = 0.0
    elif arm == "motion_only":
      selected[index :: len(arms), :-1] = 0.0
  return selected


def main() -> None:
  """Compare first episodes using matched payloads and explicit encoder draws."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--task", default=TASK_IDS["HRP5P"])
  parser.add_argument("--paired-only", action="store_true")
  parser.add_argument("--masses", type=float, nargs="+", default=[100.0, 550.0])
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--no-encoder-bias", action="store_true")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  if not args.checkpoint.is_file() or any(
    not math.isfinite(mass) or mass <= 0 for mass in args.masses
  ):
    parser.error("an existing checkpoint and positive masses are required")
  args.output.mkdir(parents=True, exist_ok=True)
  task = args.task
  arms = ("baseline", "policy") if args.paired_only else ARMS
  rows = [
    {"mass_kg": mass, "arm": arm, "condition_id": condition_id(args.seed, mass, 0)}
    for mass in args.masses
    for arm in arms
  ]
  cfg = load_env_cfg(task)
  cfg.seed = args.seed
  cfg.scene.num_envs = len(rows)
  cfg.auto_reset = False
  action_cfg = cast("LocomanipMotionActionCfg", cfg.actions[accessors.ACTION_NAME])
  action_cfg.num_workers = len(rows)
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
    func=matched_encoder_bias,
    mode="startup",
    params={"rows": rows, "enabled": not args.no_encoder_bias},
  )
  samples: list[list[dict]] = [[] for _ in rows]
  results: list[dict | None] = [None] * len(rows)
  with managed_env(cfg, args.device) as env:
    wrapped = RslRlVecEnvWrapper(env)
    runner_cls = load_runner_cls(task)
    assert runner_cls is not None
    agent_cfg = asdict(load_rl_cfg(task))
    agent_cfg["seed"] = args.seed
    runner = runner_cls(wrapped, agent_cfg, device=args.device)
    runner.load(
      str(args.checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)
    env.reset()
    actual_masses = accessors.object_mass_kg(env)
    assert_payload_masses(actual_masses, rows)
    actual = actual_masses.cpu().tolist()
    biases = env.scene[accessors.ROBOT_ENTITY].data.encoder_bias.cpu().tolist()
    term = cast(LocomanipMotionAction, accessors.residual_action(env))
    provenance = collect_controller_provenance(wrapped)
    with (args.output / "trace.jsonl").open("w") as stream:
      for step in range(env.max_episode_length):
        with torch.inference_mode():
          proposed = policy(wrapped.get_observations())
        actions = select_channels(proposed, arms)
        for index, result in enumerate(results):
          if result is not None:
            actions[index] = 0.0
        requests = actions[:, -1].cpu().tolist()
        joint_norms = actions[:, :-1].norm(dim=1).cpu().tolist()
        _, _, terminated, truncated, _ = env.step(actions)
        frames = _frames(env, term, (step + 1) * env.step_dt)
        done = (terminated | truncated).cpu().tolist()
        for index, frame in enumerate(frames):
          if results[index] is not None:
            continue
          frame.update(
            row=index,
            **rows[index],
            motion_request=requests[index],
            joint_request_norm=joint_norms[index],
            done=done[index],
          )
          samples[index].append(frame)
          stream.write(json.dumps(frame) + "\n")
          if done[index]:
            results[index] = frame
        if all(result is not None for result in results):
          break
        if any(done):
          mask = terminated | truncated
          policy.reset(mask)
          reset_done(env, mask.nonzero().flatten())
    if any(result is None for result in results):
      raise RuntimeError("channel probe lacks a terminal observation")
    summaries = [summarize(arm, env.step_dt) for arm in samples]
  report = {
    "checkpoint": str(args.checkpoint.resolve()),
    "task": task,
    "seed": args.seed,
    "encoder_bias": not args.no_encoder_bias,
    "observation_corruption": cfg.observations["actor"].enable_corruption,
    "episode_cap_s": cfg.episode_length_s,
    "controller_provenance": provenance,
    "actual_mass_kg": actual,
    "encoder_bias_rad": biases,
    "results": summaries,
  }
  (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps(summaries, indent=2))


def summarize(samples: list[dict], step_dt: float) -> dict:
  """Measure route progress, explicit holds and repeated braking transitions."""
  first, final = samples[0], samples[-1]
  count = len(samples)
  return {
    "mass_kg": final["mass_kg"],
    "arm": final["arm"],
    "time_s": final["time_s"],
    "complete": bool(final["complete"]),
    "failed": any(
      f[key]
      for f in samples
      for key in (*accessors.FALL_TERMINATIONS, "controller_failed")
    ),
    "worker_failed": any(f["worker_failed"] for f in samples),
    "robot_displacement_m": final["robot_x"] - first["robot_x"],
    "cart_displacement_m": final["cart_x"] - first["cart_x"],
    "reference_displacement_m": final["reference_x"] - first["reference_x"],
    "min_height_m": min(f["root_z"] for f in samples),
    "max_tilt_rad": max(f["tilt_rad"] for f in samples),
    "max_forward_lag_m": max(max(f["reference_x"] - f["cart_x"], 0.0) for f in samples),
    "braking_seconds": sum(f["motion_state"] == 1 for f in samples) * step_dt,
    "holding_seconds": sum(f["motion_state"] == 2 for f in samples) * step_dt,
    "braking_starts": sum(
      f["motion_state"] == 1 and before["motion_state"] != 1
      for before, f in zip(samples[:-1], samples[1:], strict=True)
    ),
    "explicit_hold_requests": sum(f["motion_request"] >= 0.95 for f in samples),
    "mean_motion_request": sum(f["motion_request"] for f in samples) / count,
    "mean_applied_scale": sum(f["applied_scale"] for f in samples) / count,
    "mean_joint_request_norm": sum(f["joint_request_norm"] for f in samples) / count,
    "input_delivery_verified": all(
      abs(f["native_input_scale"] - f["dispatched_scale"]) < 1e-6 for f in samples
    ),
  }


if __name__ == "__main__":
  main()
