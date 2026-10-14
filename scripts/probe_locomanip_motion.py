"""Compare normal motion, half-speed replanning and a four-second hold/resume."""

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
  fixed_payloads,
  matched_encoder_bias,
)
from evaluation.rollout import managed_env, reset_done
from mjlab.managers.event_manager import EventTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from mc_mjlab.actions.locomanip_motion_action import (
  MOTION_LIMIT_GETTER,
  MOTION_SCALE_GETTER,
  MOTION_SCALE_SETTER,
  MOTION_STATE_GETTER,
  LocomanipMotionAction,
)
from mc_mjlab.rl.controller_provenance import collect_controller_provenance
from mc_mjlab.tasks.locomanip.cart import cart_nominal_mass_kg
from mc_mjlab.tasks.locomanip.mdp import accessors
from mc_mjlab.tasks.locomanip_motion import TASK_IDS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlOnPolicyRunnerCfg

  from mc_mjlab.actions.locomanip_motion_action import LocomanipMotionActionCfg

ARMS = ("baseline", "slow", "hold_resume")


def main() -> None:
  """Write terminal-safe traces and fail when commanded motion is not demonstrated."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--robot", choices=list(TASK_IDS), default="HRP5P")
  parser.add_argument("--mass", type=float, default=10.0)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--hold-seconds", type=float, default=4.0)
  parser.add_argument("--resume-scale", type=float, default=1.0)
  parser.add_argument("--check-partial-reset", action="store_true")
  parser.add_argument("--registered-cap", action="store_true")
  parser.add_argument("--reward-audit", action="store_true")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  if args.mass <= 0 or args.hold_seconds <= 0:
    parser.error("mass and hold duration must be positive")
  if not 0.1 <= args.resume_scale <= 1:
    parser.error("resume scale must be in [0.1, 1]")
  args.output.mkdir(parents=True, exist_ok=True)
  arms = (*ARMS, "hold_forever") if args.reward_audit else ARMS
  cfg = load_env_cfg(TASK_IDS[args.robot])
  cfg.scene.num_envs = len(arms)
  cfg.seed = args.seed
  cfg.auto_reset = False
  if not args.registered_cap:
    cfg.episode_length_s = (
      cfg.episode_length_s / min(0.5, args.resume_scale) + args.hold_seconds + 10.0
    )
  action_cfg = cast("LocomanipMotionActionCfg", cfg.actions[accessors.ACTION_NAME])
  action_cfg.num_workers = len(arms)
  conditions = [
    {"mass_kg": args.mass, "condition_id": f"motion/{args.seed}/{args.mass:g}"}
    for _ in arms
  ]
  payload = cfg.events["cart_payload"]
  cfg.events["cart_payload"] = EventTermCfg(
    func=fixed_payloads,
    mode="reset",
    params={
      "rows": conditions,
      "nominal_mass": cart_nominal_mass_kg(),
      "asset_cfg": payload.params["asset_cfg"],
    },
  )
  cfg.events["encoder_bias"] = EventTermCfg(
    func=matched_encoder_bias,
    mode="startup",
    params={"rows": conditions, "enabled": False},
  )
  samples: list[list[dict]] = [[] for _ in arms]
  results: list[dict | None] = [None] * len(arms)
  triggered: list[float | None] = [None] * len(arms)
  held_since: float | None = None
  with managed_env(cfg, args.device) as env:
    env.reset()
    assert_payload_masses(accessors.object_mass_kg(env), conditions)
    term = cast(LocomanipMotionAction, accessors.residual_action(env))
    provenance = collect_controller_provenance(RslRlVecEnvWrapper(env))
    reward_weights = {
      name: env.reward_manager.get_term_cfg(name).weight
      for name in env.reward_manager.active_terms
    }
    start = (
      env.scene[accessors.OBJECT_ENTITY].data.root_link_pos_w[:, 0]
      - env.scene.env_origins[:, 0]
    ).clone()
    actions = torch.zeros(
      env.num_envs, env.action_manager.total_action_dim, device=env.device
    )
    with (args.output / "trace.jsonl").open("w") as stream:
      for step in range(env.max_episode_length):
        time = step * env.step_dt
        holding = accessors.holding(env).bool().cpu().tolist()
        reference = term.datastore_vector_output(accessors.OBJECT_REFERENCE_POSITION)
        progress = (reference[:, 0] - start).cpu().tolist()
        states = term.datastore_scalar_output(MOTION_STATE_GETTER).cpu().tolist()
        for index in range(1, len(arms)):
          if triggered[index] is None and holding[index] and progress[index] >= 0.15:
            triggered[index] = time
        if triggered[1] is not None:
          actions[1, -1] = 0.5
        if triggered[2] is not None:
          if states[2] == 2.0 and held_since is None:
            held_since = time
          actions[2, -1] = (
            1.0
            if held_since is None or time - held_since < args.hold_seconds
            else 1.0 - args.resume_scale
          )
        if args.reward_audit and triggered[3] is not None:
          actions[3, -1] = 1.0
        _, rewards, terminated, truncated, _ = env.step(actions)
        frames = _frames(env, term, time + env.step_dt)
        rates = env.reward_manager._step_reward.cpu().tolist()
        reward_values = rewards.cpu().tolist()
        terminated_values = terminated.cpu().tolist()
        truncated_values = truncated.cpu().tolist()
        done = (terminated | truncated).cpu().tolist()
        for index, frame in enumerate(frames):
          if results[index] is not None:
            continue
          frame.update(
            arm=arms[index],
            trigger_time_s=triggered[index],
            reward=reward_values[index],
            reward_rates=dict(zip(reward_weights, rates[index], strict=True)),
            terminated=terminated_values[index],
            truncated=truncated_values[index],
          )
          samples[index].append(frame)
          stream.write(json.dumps(frame) + "\n")
          if done[index]:
            results[index] = frame
        if all(result is not None for result in results):
          break
        if any(done):
          reset_done(env, (terminated | truncated).nonzero().flatten())
    for index, result in enumerate(results):
      if result is None:
        results[index] = samples[index][-1]
  report = {
    "mass_kg": args.mass,
    "seed": args.seed,
    "encoder_bias": False,
    "resume_scale": args.resume_scale,
    "episode_cap_s": cfg.episode_length_s,
    "step_dt": env.step_dt,
    "gamma": cast(
      "RslRlOnPolicyRunnerCfg", load_rl_cfg(TASK_IDS[args.robot])
    ).algorithm.gamma,
    "reward_weights": reward_weights,
    "results": results,
    "provenance": provenance,
    "checks": _checks(samples[: len(ARMS)], args.hold_seconds),
  }
  if args.check_partial_reset:
    held = [f for f in samples[2] if f["motion_state"] == 2.0]
    report["checks"]["other_row_reset_preserves_hold"] = bool(held) and (
      held[0]["time_s"] < samples[0][-1]["time_s"] < held[-1]["time_s"]
    )
  (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
  print(json.dumps({"results": results, "checks": report["checks"]}, indent=2))
  if not all(report["checks"].values()):
    raise SystemExit(1)


def _frames(
  env: ManagerBasedRlEnv, term: LocomanipMotionAction, time: float
) -> list[dict]:
  """Capture controller and physical measurements before any row resets."""
  reference = term.datastore_vector_output(accessors.OBJECT_REFERENCE_POSITION)
  root = env.scene[accessors.ROBOT_ENTITY].data
  tilt = torch.acos((-root.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
  values = (
    torch.stack(
      (
        reference[:, 0],
        env.scene[accessors.OBJECT_ENTITY].data.root_link_pos_w[:, 0]
        - env.scene.env_origins[:, 0],
        root.root_link_pos_w[:, 0] - env.scene.env_origins[:, 0],
        root.root_link_pos_w[:, 2],
        tilt,
        term.motion_scale_dispatched,
        term.datastore_scalar_output(MOTION_SCALE_GETTER),
        term.datastore_scalar_output(MOTION_STATE_GETTER),
        term.datastore_scalar_output(MOTION_LIMIT_GETTER),
        term.datastore_scalar_output(accessors.COMPLETE),
        term.controller_failed.to(torch.float32),
        term.controller_worker_failed.to(torch.float32),
        env.termination_manager.get_term("fell_over").to(torch.float32),
        env.termination_manager.get_term("collapsed").to(torch.float32),
        accessors.holding(env),
      ),
      dim=1,
    )
    .cpu()
    .tolist()
  )
  keys = (
    "reference_x",
    "cart_x",
    "robot_x",
    "root_z",
    "tilt_rad",
    "dispatched_scale",
    "applied_scale",
    "motion_state",
    "resume_limit",
    "complete",
    "controller_failed",
    "worker_failed",
    "fell_over",
    "collapsed",
    "holding",
  )
  column = term._datastore_scalar_input_columns[MOTION_SCALE_SETTER]
  assert term._in_np is not None
  native_inputs = term._in_np[:, column].tolist()
  return [
    {
      "time_s": time,
      "native_input_scale": native_inputs[index],
      **dict(zip(keys, row, strict=True)),
    }
    for index, row in enumerate(values)
  ]


def _checks(samples: list[list[dict]], hold_seconds: float) -> dict[str, bool]:
  """Require observed reference hold, slow motion, preserved goal and completion."""
  finals = [arm[-1] for arm in samples]
  held = [frame for frame in samples[2] if frame["motion_state"] == 2.0]
  slow = [frame for frame in samples[1] if abs(frame["applied_scale"] - 0.5) < 1e-6]
  resumed = [
    frame
    for frame in samples[2]
    if held and frame["time_s"] > held[-1]["time_s"] and frame["motion_state"] == 0.0
  ]
  return {
    "all_completed": all(frame["complete"] for frame in finals),
    "actual_native_delivery": all(
      abs(frame["native_input_scale"] - frame["dispatched_scale"]) < 1e-6
      for arm in samples
      for frame in arm
    ),
    "all_safe": all(
      math.isfinite(frame["root_z"])
      and math.isfinite(frame["tilt_rad"])
      and not frame["controller_failed"]
      and not frame["worker_failed"]
      and not frame["fell_over"]
      and not frame["collapsed"]
      for arm in samples
      for frame in arm
    ),
    "half_speed_observed": bool(slow),
    "post_hold_resume_limited": bool(resumed)
    and all(
      frame["resume_limit"] == 0.5 and frame["applied_scale"] <= 0.5
      for frame in resumed
    ),
    "slow_completion_later": finals[1]["time_s"] > finals[0]["time_s"] + 5.0,
    "hold_duration_observed": bool(held)
    and held[-1]["time_s"] - held[0]["time_s"] >= hold_seconds - 0.05,
    "hold_applied_zero": bool(held)
    and all(abs(frame["applied_scale"]) < 1e-6 for frame in held),
    "reference_stationary_during_hold": bool(held)
    and max(f["reference_x"] for f in held) - min(f["reference_x"] for f in held)
    < 1e-6,
    "physical_robot_holds": bool(held)
    and max(f["robot_x"] for f in held) - min(f["robot_x"] for f in held) < 0.05,
    "goal_preserved": max(f["reference_x"] for f in finals)
    - min(f["reference_x"] for f in finals)
    < 1e-4,
  }


if __name__ == "__main__":
  main()
