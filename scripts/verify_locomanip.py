"""Measure physical cart pushing and repeated reset with zero actions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv

from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase
from mc_mjlab.tasks.locomanip.locomanip_env_cfg import locomanip_env_cfg


def sample(
  env: ManagerBasedRlEnv, term: McRtcResidualActionBase, row: int
) -> dict[str, Any]:
  """Read terminal-safe local poses, phases, hand contacts and gripper targets."""
  robot, cart = env.scene["robot"], env.scene["cart"]
  origin = env.scene.env_origins[row]
  phases = [
    int(term.datastore_scalar_output(f"Locomanip::{side}Phase")[row])
    for side in ("left", "right")
  ]
  contacts, forces, wrenches = [], [], []
  for side in ("left", "right"):
    sensor = env.scene[f"{side}_hand_cart"].data
    recent = (
      sensor.force_history[row].abs().sum() if sensor.force_history is not None else 0
    )
    contacts.append(bool(sensor.found[row].any() or recent > 0))
    forces.append(sensor.force[row].sum(dim=0).tolist())
    name = f"robot/{side.title()}HandForceSensor_fsensor"
    adr = int(env.sim.mj_model.sensor(name).adr[0])
    wrenches.append(env.sim.data.sensordata[row, adr : adr + 3].tolist())
  joint_ids = [
    i
    for i, n in enumerate(robot.joint_names)
    if n
    in (
      "LTMP",
      "LTPIP",
      "LTDIP",
      "LIMP",
      "LIPIP",
      "LIDIP",
      "RTMP",
      "RTPIP",
      "RTDIP",
      "RIMP",
      "RIPIP",
      "RIDIP",
    )
  ]
  targets = term.controller_reference("q")[row]
  grippers = {
    name: float(targets[i])
    for i, name in enumerate(term._target_names)
    if name in [robot.joint_names[j] for j in joint_ids]
  }
  return {
    "time": float(env.episode_length_buf[row]) * env.step_dt,
    "robot": (robot.data.root_link_pos_w[row] - origin).tolist(),
    "cart": (cart.data.root_link_pos_w[row] - origin).tolist(),
    "cart_quat": cart.data.root_link_quat_w[row].tolist(),
    "reference": term.datastore_vector_output("Locomanip::objectReferencePosition")[
      row
    ].tolist(),
    "reference_rpy": term.datastore_vector_output("Locomanip::objectReferenceRpy")[
      row
    ].tolist(),
    "phases": phases,
    "complete": bool(term.datastore_scalar_output("Locomanip::complete")[row]),
    "contacts": contacts,
    "contact_forces": forces,
    "hand_sensor_forces": wrenches,
    "gripper_targets": grippers,
    "gripper_measured": {
      robot.joint_names[j]: float(robot.data.joint_pos[row, j]) for j in joint_ids
    },
    "controller_failed": bool(term.controller_failed[row]),
    "worker_failed": bool(term.controller_worker_failed[row]),
  }


def result(
  initial: dict[str, Any],
  final: dict[str, Any],
  minimum_height: float,
  bilateral_steps: int,
  hold_since: float | None,
  dt: float,
) -> dict[str, Any]:
  """Apply the milestone's physical acceptance tolerances to one cycle."""
  error = math.dist(final["cart"], final["reference"])
  w, x, y, z = final["cart_quat"]
  yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  yaw_error = abs(math.remainder(yaw - final["reference_rpy"][2], 2 * math.pi))
  displacement = final["robot"][0] - initial["robot"][0]
  held = final["time"] - hold_since if hold_since is not None else 0.0
  checks = {
    "completed": final["complete"] and final["time"] <= 60,
    "cart_position": error <= 0.15,
    "cart_orientation": yaw_error <= math.radians(10),
    "robot_displacement": displacement >= 0.5,
    "bilateral_contact": bilateral_steps > 0,
    "released": final["phases"] == [0, 0],
    "upright": minimum_height >= 0.55,
    "controller_healthy": not final["controller_failed"] and not final["worker_failed"],
    "held_two_seconds": held >= 2.0 - dt / 2,
  }
  return {
    "passed": all(checks.values()),
    "checks": checks,
    "cart_position_error_m": error,
    "cart_yaw_error_deg": math.degrees(yaw_error),
    "robot_forward_m": displacement,
    "minimum_height_m": minimum_height,
    "bilateral_contact_s": bilateral_steps * dt,
    "released_hold_s": held,
    "final": final,
  }


def run(args: argparse.Namespace) -> bool:
  """Run explicit reset cycles and optionally reset one of two shared-worker rows."""
  cfg = locomanip_env_cfg(play=args.console)
  cfg.auto_reset = False
  cfg.scene.num_envs = 2 if args.isolation else 1
  if args.isolation:
    cfg.episode_length_s = args.max_seconds + 12.0
  env = ManagerBasedRlEnv(cfg, device=args.device)
  results = []
  args.output.parent.mkdir(parents=True, exist_ok=True)
  try:
    term = env.action_manager.get_term("robot_joints")
    if not isinstance(term, McRtcResidualActionBase):
      raise TypeError(f"unexpected residual action type: {type(term).__name__}")
    actions = torch.zeros(
      (env.num_envs, env.action_manager.total_action_dim), device=env.device
    )
    with args.output.with_suffix(".jsonl").open("w") as trace:
      for cycle in range(args.cycles):
        env.reset()
        initial = [sample(env, term, row) for row in range(env.num_envs)]
        minimum = [s["robot"][2] for s in initial]
        bilateral = [0] * env.num_envs
        hold_since = [None] * env.num_envs
        previous = [None] * env.num_envs
        finished = [False] * env.num_envs
        reset_done = False
        run_seconds = args.max_seconds + (10.0 if args.isolation else 0.0)
        for step in range(math.ceil(run_seconds / env.step_dt)):
          _, _, terminated, truncated, _ = env.step(actions)
          for row in range(env.num_envs):
            if finished[row]:
              continue
            state = sample(env, term, row)
            minimum[row] = min(minimum[row], state["robot"][2])
            bilateral[row] += int(state["phases"] == [4, 4] and all(state["contacts"]))
            holding = state["complete"] and state["phases"] == [0, 0]
            if holding and hold_since[row] is None:
              hold_since[row] = state["time"]
            elif not holding:
              hold_since[row] = None
            transition = previous[row] != (state["phases"], state["complete"])
            if transition or step % 50 == 0:
              trace.write(json.dumps({"cycle": cycle + 1, "env": row, **state}) + "\n")
              trace.flush()
            if transition or step % 1000 == 0:
              print(
                f"cycle={cycle + 1} env={row} t={state['time']:.2f} "
                f"phase={state['phases']} cart={state['cart']} "
                f"z={state['robot'][2]:.3f} contacts={state['contacts']}",
                flush=True,
              )
            previous[row] = (state["phases"], state["complete"])
            held = (
              hold_since[row] is not None
              and state["time"] - hold_since[row] >= 2.0 - env.step_dt / 2
            )
            done = bool(terminated[row] or truncated[row])
            if held or done or step + 1 == math.ceil(run_seconds / env.step_dt):
              measured = result(
                initial[row],
                state,
                minimum[row],
                bilateral[row],
                hold_since[row],
                env.step_dt,
              )
              measured.update(
                cycle=cycle + 1, env=row, terminated=bool(terminated[row])
              )
              measured["passed"] &= not bool(terminated[row])
              results.append(measured)
              finished[row] = True
              print(json.dumps(measured), flush=True)
          if all(finished):
            break
          if (terminated | truncated).any():
            break
          if args.isolation and not reset_done and (step + 1) * env.step_dt >= 10:
            untouched = env.sim.data.qpos[1].clone()
            env.reset(env_ids=torch.tensor([0], device=env.device))
            if not torch.equal(untouched, env.sim.data.qpos[1]):
              raise AssertionError("reset of env 0 modified env 1 qpos")
            initial[0] = sample(env, term, 0)
            minimum[0] = initial[0]["robot"][2]
            bilateral[0], hold_since[0], previous[0] = 0, None, None
            reset_done = True
            trace.write(
              json.dumps({"cycle": cycle + 1, "reset_env": 0, "time": 10.0}) + "\n"
            )
        args.output.write_text(
          json.dumps({"isolation": args.isolation, "results": results}, indent=2) + "\n"
        )
  finally:
    env.close()
  expected = args.cycles * (2 if args.isolation else 1)
  return len(results) == expected and all(r["passed"] for r in results)


def main() -> None:
  """Parse headless verification options and return a failing status on any miss."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--cycles", type=int, default=5)
  parser.add_argument("--isolation", action="store_true")
  parser.add_argument("--max-seconds", type=float, default=60.0)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--console", action="store_true")
  parser.add_argument(
    "--output", type=Path, default=Path("logs/locomanip/verification.json")
  )
  args = parser.parse_args()
  if args.cycles < 1 or args.max_seconds <= 0:
    parser.error("cycles and max-seconds must be positive")
  raise SystemExit(0 if run(args) else 1)


if __name__ == "__main__":
  main()
