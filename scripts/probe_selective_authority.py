"""Measure two-second DCM, CoP, and effort responses by residual joint set."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from mc_mjlab import MC_RTC_YAML_PATH, mdp
from mc_mjlab.robots import robot_module as mc_rtc
from mc_mjlab.robots.registry import get_main_robot_spec
from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import (
  _make_env_cfg,
  _select_residual_joints,
)


@dataclass
class Response:
  """One paired pulse response row."""

  target: str
  joints: str
  dcm_signed_m: float
  dcm_change_m: float
  cop_shift_m: float
  effort_change_ratio: float


def target_sets(requested: list[str] | None) -> dict[str, tuple[str, ...]]:
  """Resolve individual joints and anatomy groups before constructing the env."""
  robot_name, robot = get_main_robot_spec(MC_RTC_YAML_PATH)
  upper = set(mc_rtc.get_upper_body_joints(robot_name))
  candidates = tuple(j for j in robot.get_residual_joints() if j not in upper)
  groups = {
    "ankle": _select_residual_joints(robot_name, candidates, "ankle"),
    "sagittal": _select_residual_joints(robot_name, candidates, "sagittal"),
    "all": candidates,
  }
  if requested is None:
    return {**{joint: (joint,) for joint in candidates}, **groups}
  output: dict[str, tuple[str, ...]] = {}
  for name in requested:
    if name in groups:
      output[name] = groups[name]
    elif name in candidates:
      output[name] = (name,)
    else:
      raise ValueError(f"unknown target {name!r}; joints are {candidates}")
  return output


def main() -> None:
  """Run every requested pulse alongside its own zero-residual environment."""
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--target", action="append")
  parser.add_argument("--control", choices=("position", "torque"), default="position")
  parser.add_argument("--level", type=float, default=1.0)
  parser.add_argument("--sign", type=float, choices=(-1.0, 1.0), default=1.0)
  parser.add_argument("--settle-s", type=float, default=10.0)
  parser.add_argument("--pulse-s", type=float, default=2.0)
  parser.add_argument("--num-workers", type=int, default=12)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--dump", type=Path)
  args = parser.parse_args()
  targets = target_sets(args.target)
  cfg = _make_env_cfg(
    args.control,
    num_envs=2 * len(targets),
    num_workers=args.num_workers,
    recovery_detector_path=None,
    disturbance="none",
  )
  cfg.events["reset_base"].params["pose_range"] = {}
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg, device=args.device)
  term = mdp.sensors._residual_term(env, "mc_rtc_residual")
  ids = term.residual_ids
  residual_names = (
    term.target_names if ids is None else tuple(term.target_names[i] for i in ids)
  )
  action_col = {name: index for index, name in enumerate(residual_names)}
  action = torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )
  for pair, joints in enumerate(targets.values()):
    for joint in joints:
      action[2 * pair + 1, action_col[joint]] = args.sign * args.level
  sensors = mdp.sensors._ZmpSensors(env, mdp.sensors.GROUND_CONTACT_SENSORS, "robot")
  robot_name = term.cfg.mc_rtc_robot_name
  limits = mc_rtc.get_effort_limits(robot_name)
  residual_target_ids = (
    torch.arange(len(term.target_names), device=env.device) if ids is None else ids
  )
  limit = torch.tensor([limits[name] for name in residual_names], device=env.device)
  settle_steps = round(args.settle_s / env.step_dt)
  pulse_steps = round(args.pulse_s / env.step_dt)
  env.reset()
  for _ in range(settle_steps):
    env.step(torch.zeros_like(action))
  accum = torch.zeros(len(targets), 4, device=env.device)
  for _ in range(pulse_steps):
    _, _, terminated, time_outs, _ = env.step(action)
    if bool((terminated | time_outs).any()):
      raise RuntimeError("an authority-probe environment ended during its pulse")
    dcm, _ = sensors.dcm_offset(env)
    cop, _ = sensors.measured_offset(env)
    effort = env.scene[term.cfg.entity_name].data.qfrc_actuator[:, term.target_ids]
    base = torch.arange(0, env.num_envs, 2, device=env.device)
    pulse = base + 1
    accum[:, 0] += dcm[pulse] - dcm[base]
    accum[:, 1] += (dcm[pulse] - dcm[base]).abs()
    accum[:, 2] += torch.linalg.vector_norm(cop[pulse] - cop[base], dim=1)
    effort_delta = (effort[pulse] - effort[base])[:, residual_target_ids]
    accum[:, 3] += (effort_delta.abs() / limit).amax(dim=1)
  env.close()
  mean = (accum / pulse_steps).cpu().tolist()
  rows = [
    Response(name, " ".join(joints), *values)
    for (name, joints), values in zip(targets.items(), mean, strict=True)
  ]
  print(
    f"{'target':<12} {'DCM signed':>11} {'|DCM change|':>13} "
    f"{'CoP shift':>10} {'effort ratio':>13}"
  )
  for row in rows:
    print(
      f"{row.target:<12} {row.dcm_signed_m:11.5f} {row.dcm_change_m:13.5f} "
      f"{row.cop_shift_m:10.5f} {row.effort_change_ratio:13.3f}"
    )
  if args.dump is not None:
    args.dump.parent.mkdir(parents=True, exist_ok=True)
    with args.dump.open("w", newline="") as handle:
      writer = csv.DictWriter(handle, fieldnames=tuple(Response.__annotations__))
      writer.writeheader()
      writer.writerows(vars(row) for row in rows)
    print(f"selective-authority responses -> {args.dump}")


if __name__ == "__main__":
  main()
