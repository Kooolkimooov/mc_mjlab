"""Map which velocity commands a ResidualMPC arm can actually track."""

from __future__ import annotations

import argparse
import csv
import itertools
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls

from mc_mjlab.tasks.residual_mpc.residual_mpc_env_cfg import residual_mpc_env_cfg

#: Paper Fig. 10 calls a command "achieved" below this planar tracking error.
ACHIEVED_ERROR = 0.25
#: Discarded before scoring. The ISMPC realises a new reference over many gait
#: cycles -- it was still accelerating 8 s after a 0.6 m/s command -- so a short
#: settle measures the transient and reads as "cannot track".
SETTLE_S = 15.0
SWEEP_DIR = Path("logs/envelopes")


@dataclass
class Sample:
  """One held command evaluated on one environment."""

  arm: str
  vx: float
  vy: float
  wz: float
  env_id: int
  linear_error: float
  angular_error: float
  achieved: int
  terminated: int


def _grid(spec: str) -> list[float]:
  """Parse ``lo:hi:count`` into evenly spaced command values."""
  lo, hi, count = spec.split(":")
  n = int(count)
  if n < 1:
    raise ValueError(f"{spec!r} needs at least one point")
  if n == 1:
    return [float(lo)]
  step = (float(hi) - float(lo)) / (n - 1)
  return [float(lo) + step * i for i in range(n)]


def evaluate(twist, arm: str, policy, args) -> list[Sample]:
  """Hold one twist across every environment and score the settled tracking."""
  cfg = residual_mpc_env_cfg(
    num_envs=args.num_envs,
    num_workers=args.num_workers,
    fixed_twist=twist,
    pushes=not args.no_pushes,
  )
  cfg.seed = args.seed
  cfg.episode_length_s = args.episode_length_s
  # Terminated envs must stay terminated: an in-place reset would silently score
  # a fresh episode against the same command.
  cfg.auto_reset = False
  if args.in_process:
    cfg.actions["mc_rtc_residual"].use_worker_processes = False
  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    wrapped = RslRlVecEnvWrapper(env)
    settle = round(args.settle_s / env.step_dt)
    steps = round(args.episode_length_s / env.step_dt)
    asset = env.scene["robot"]
    zeros = torch.zeros(
      env.num_envs, env.action_manager.total_action_dim, device=env.device
    )
    linear, angular = [], []
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    terminated = torch.zeros_like(alive)
    for step in range(steps):
      with torch.inference_mode():
        action = zeros if policy is None else policy(wrapped.get_observations())
      _, _, step_terminated, _, _ = env.step(action)
      terminated |= step_terminated.bool()
      alive &= ~step_terminated.bool()
      if step < settle:
        continue
      command = env.command_manager.get_command("twist")
      assert command is not None
      planar = asset.data.root_link_lin_vel_b[:, :2]
      # Only living envs contribute; a fallen robot's velocity is not tracking.
      linear.append(torch.linalg.vector_norm(command[:, :2] - planar, dim=1) * alive)
      angular.append(
        (command[:, 2] - asset.data.root_link_ang_vel_b[:, 2]).abs() * alive
      )
      if step == settle:
        counted = alive.clone().float()
      else:
        counted += alive.float()
    denominator = counted.clamp_min(1.0)
    lin = torch.stack(linear).sum(dim=0) / denominator
    ang = torch.stack(angular).sum(dim=0) / denominator
    return [
      Sample(
        arm=arm,
        vx=twist[0],
        vy=twist[1],
        wz=twist[2],
        env_id=i,
        linear_error=float(lin[i]),
        angular_error=float(ang[i]),
        achieved=int(float(lin[i]) < ACHIEVED_ERROR and not bool(terminated[i])),
        terminated=int(bool(terminated[i])),
      )
      for i in range(args.num_envs)
    ]
  finally:
    env.close()


def main() -> None:
  """Sweep a command grid for the MPC prior and, optionally, a checkpoint."""
  p = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  p.add_argument("--checkpoint", help="score this policy as well as the MPC prior")
  p.add_argument(
    "--task",
    default="Mc-Mjlab-Residual-Mpc-Logisticcontroller-Ismpc-Hrp5P-Joint-Torque",
  )
  p.add_argument("--vx", default="0.0:1.0:6", help="lo:hi:count")
  p.add_argument("--vy", default="0.0:0.0:1", help="lo:hi:count")
  p.add_argument("--wz", default="0.0:0.0:1", help="lo:hi:count")
  p.add_argument("--num-envs", type=int, default=8)
  p.add_argument("--num-workers", type=int, default=6)
  p.add_argument("--episode-length-s", type=float, default=30.0)
  p.add_argument("--settle-s", type=float, default=SETTLE_S)
  p.add_argument("--seed", type=int, default=42)
  p.add_argument("--no-pushes", action="store_true")
  p.add_argument("--in-process", action="store_true")
  p.add_argument("--device", default="cuda:0")
  p.add_argument("--out-dir", type=Path, default=SWEEP_DIR)
  args = p.parse_args()

  grid = list(itertools.product(_grid(args.vx), _grid(args.vy), _grid(args.wz)))
  print(
    f"[sweep] {len(grid)} commands x {args.num_envs} envs, {args.episode_length_s:g} s each"
  )

  arms: list[tuple[str, object]] = [("mpc", None)]
  if args.checkpoint:
    # Built once against the first command; the policy is command-independent.
    cfg = residual_mpc_env_cfg(num_envs=args.num_envs, num_workers=args.num_workers)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
    runner = runner_cls(
      RslRlVecEnvWrapper(env), asdict(load_rl_cfg(args.task)), device=args.device
    )
    runner.load(
      args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device
    )
    arms.append(("policy", runner.get_inference_policy(device=args.device)))
    env.close()

  samples: list[Sample] = []
  for arm, policy in arms:
    for twist in grid:
      batch = evaluate(twist, arm, policy, args)
      samples.extend(batch)
      achieved = sum(s.achieved for s in batch) / len(batch)
      error = statistics.fmean(s.linear_error for s in batch)
      print(
        f"[sweep] {arm:6s} vx={twist[0]:+.2f} vy={twist[1]:+.2f} wz={twist[2]:+.2f}"
        f"  err={error:.3f}  achieved={achieved:5.1%}"
      )

  args.out_dir.mkdir(parents=True, exist_ok=True)
  stem = Path(args.checkpoint).stem if args.checkpoint else "mpc_only"
  path = args.out_dir / f"{stem}_envelope.csv"
  with path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(asdict(samples[0])))
    writer.writeheader()
    writer.writerows(asdict(s) for s in samples)

  print(f"\n[sweep] achieved boundary (share of envs under {ACHIEVED_ERROR} m/s)")
  for arm, _ in arms:
    reached = [s for s in samples if s.arm == arm and s.achieved]
    if not reached:
      print(f"  {arm:6s} nothing achieved")
      continue
    print(
      f"  {arm:6s} vx<={max(s.vx for s in reached):+.2f}"
      f"  |vy|<={max(abs(s.vy) for s in reached):.2f}"
      f"  |wz|<={max(abs(s.wz) for s in reached):.2f}"
    )
  print(f"[sweep] rows -> {path}")


if __name__ == "__main__":
  main()
