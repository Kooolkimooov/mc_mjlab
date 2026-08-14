"""Does a constant residual move the centre of pressure? -- docs/evaluation.md."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time

import torch
from mjlab.envs import ManagerBasedRlEnv

from mc_mjlab.tasks import mdp
from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import _make_env_cfg

# Bin width for the recovery profile, in policy steps (50 Hz -> 5 bins/100 ms).
BIN_STEPS = 5
NUM_BINS = 80  # 400 steps = 8 s, longer than the 5-7 s push interval.


def main() -> None:
  p = argparse.ArgumentParser(
    description=(
      "Drive a constant residual and measure how far it moves the centre of "
      "pressure, to test whether the action has the authority to affect the "
      "objective at all."
    ),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  p.add_argument(
    "--num-envs", type=int, default=32, help="parallel environments to sample from"
  )
  p.add_argument(
    "--num-workers",
    type=int,
    default=12,
    help="mc_rtc worker processes hosting the controllers; keep well under "
    "cpu_count if a training run is using the machine",
  )
  p.add_argument(
    "--minutes",
    type=float,
    default=10.0,
    help="wall-clock budget; the per-step error saturates in 1-3 min, longer "
    "only tightens the push-recovery bins",
  )
  p.add_argument("--device", default="cuda:0", help="torch device for the simulation")
  p.add_argument(
    "--level",
    type=float,
    default=0.0,
    help="residual as a fraction of its clip; raw action 1.0 == residual_scale",
  )
  p.add_argument(
    "--pattern",
    default="alternating",
    choices=("all", "alternating", "random", "noise"),
    help=(
      "sign pattern over the residual joints: 'all' pushes every joint the same "
      "way (a gross posture bias), 'alternating' opposes neighbours, 'random' is "
      "fixed per env, 'noise' resamples every step (what an untrained policy "
      "looks like)"
    ),
  )
  p.add_argument("--dump", default=None, help="CSV path for the recovery profile")
  args = p.parse_args()

  cfg = _make_env_cfg(
    control="position",
    num_envs=args.num_envs,
    num_workers=args.num_workers,
    console_output="none",
  )
  # `step()` would otherwise reset in place and erase what this measures.
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg, device=args.device)

  dim = env.action_manager.total_action_dim
  action = _build_action(args.pattern, args.level, env.num_envs, dim, env.device)

  # The same plumbing the reward uses, in metres before the kernel flattens it.
  sensors = mdp._ZmpSensors(env, mdp.GROUND_CONTACT_SENSORS, "robot")

  print(
    f"[probe] level {args.level:g} pattern {args.pattern} · {args.num_envs} envs, "
    f"{args.num_workers} workers, {args.minutes:.0f} min"
  )

  env.reset()
  bin_sum = torch.zeros(NUM_BINS, dtype=torch.float64, device=env.device)
  bin_count = torch.zeros(NUM_BINS, dtype=torch.float64, device=env.device)
  errors: list[float] = []
  steps = 0

  deadline = time.monotonic() + args.minutes * 60.0
  while time.monotonic() < deadline:
    if args.pattern == "noise":
      action = _build_action("noise", args.level, env.num_envs, dim, env.device)
    _, _, terminated, time_outs, _ = env.step(action)
    steps += 1

    measured, normal_force = sensors.measured_offset(env)
    err = torch.linalg.vector_norm(measured - mdp.planned_zmp_offset(env), dim=1)
    grounded = normal_force >= 20.0

    # From the term's own record, not the interval countdown, which also ticks
    # for suppressed pushes. docs/evaluation.md#binning-by-time-since-a-push
    age = mdp.steps_since_push(env)

    slot = (age // BIN_STEPS).clamp(max=NUM_BINS - 1)
    # `age >= 1` matches `recovery_tracking`'s gate; age 0 predates the effect.
    keep = grounded & (age >= 1) & (age < NUM_BINS * BIN_STEPS)
    if keep.any():
      bin_sum.index_add_(0, slot[keep], err[keep].double())
      bin_count.index_add_(
        0, slot[keep], torch.ones_like(err[keep], dtype=torch.float64)
      )
      errors += err[keep].tolist()

    done = (terminated | time_outs).nonzero(as_tuple=False).flatten()
    if done.numel():
      env.reset(env_ids=done)

  env.close()

  if not errors:
    print("[probe] no grounded samples; raise --minutes")
    return

  mean = statistics.fmean(errors)
  sd = statistics.stdev(errors)
  print(f"\n[probe] {len(errors)} grounded samples over {steps} steps")
  print(f"  zmp_error mean   {mean:.5f} m  ± {sd / math.sqrt(len(errors)):.5f} sem")
  print(f"  zmp_error median {statistics.median(errors):.5f} m   sd {sd:.5f}")

  print("\n  recovery profile — mean zmp_error by time since push")
  counts = bin_count.cpu().tolist()
  sums = bin_sum.cpu().tolist()
  rows = []
  for i, (s, c) in enumerate(zip(sums, counts, strict=True)):
    if c < 50:
      continue
    t0 = i * BIN_STEPS * env.step_dt
    rows.append((t0, s / c, int(c)))
  # "No bin reached 3 s" is normal on a short run, and used to raise
  # `StatisticsError` here -- after `env.close()` and before `--dump`.
  settled = [m for t, m, _ in rows if t >= 3.0]
  tail = statistics.fmean(settled) if settled else float("nan")
  for t0, m, c in rows[:24]:
    bar = "" if not settled else "#" * max(0, round((m / max(tail, 1e-9) - 1.0) * 40))
    print(f"    {t0:5.2f}s  {m:.5f} m  n={c:7d}  {bar}")
  if settled:
    print(f"    settled level (>=3 s after a push): {tail:.5f} m")
  else:
    print(
      "    no bin reached 3 s with n>=50, so there is no settled level to "
      "compare against; raise --minutes or --num-envs"
    )

  if args.dump:
    with open(args.dump, "w", newline="") as fh:
      w = csv.writer(fh)
      w.writerow(["level", "pattern", "t_since_push_s", "mean_zmp_error", "n"])
      for t0, m, c in rows:
        w.writerow([args.level, args.pattern, f"{t0:.3f}", f"{m:.6f}", c])
    print(f"\n[probe] profile -> {args.dump}")


def _build_action(
  pattern: str, level: float, num_envs: int, dim: int, device: str
) -> torch.Tensor:
  """Raw actions in [-1, 1]; the action term scales them to ±``residual_scale``."""
  if pattern == "noise":
    return (torch.rand(num_envs, dim, device=device) * 2.0 - 1.0) * level
  if pattern == "random":
    signs = torch.randint(0, 2, (num_envs, dim), device=device) * 2 - 1
    return signs.float() * level
  if pattern == "alternating":
    signs = torch.ones(dim, device=device)
    signs[1::2] = -1.0
    return signs.expand(num_envs, dim) * level
  return torch.full((num_envs, dim), level, device=device)


if __name__ == "__main__":
  main()
