"""Does the residual actually move the centre of pressure?

The residual_balance reward proved to be blind to the policy: with episode length
divided out, the per-step ``zmp_tracking`` rate is the same for a zero residual
and for every trained checkpoint (0.01183 / 0.01187 / 0.01196, SE 0.00015). Two
things can cause that -- the reward is badly shaped, or the *action cannot reach
the objective at all*. This settles the second before anyone spends time on the
first, because at ``residual_scale = 0.01`` rad (~0.57 degrees of joint offset)
it is entirely possible the residual simply cannot shift the centre of pressure.

The probe drives a **constant** residual instead of a policy and measures
``mdp.zmp_error`` -- the distance in metres between the measured centre of
pressure and the one mc_rtc planned. A constant offset is the bluntest possible
input: if a full-scale one does not move that distance, nothing a policy does
will either, and the reward is not the binding constraint.

It also bins the error by *time since the last push*, which gives the recovery
profile -- how long after a disturbance the tracking error stays elevated. That
is the window a disturbance-gated reward should pay on, so the sweep sizes the
next change as well as justifying it.

  uv run python scripts/probe_residual_authority.py --level 0    --minutes 10
  uv run python scripts/probe_residual_authority.py --level 1.0  --pattern alternating
  uv run python scripts/probe_residual_authority.py --level 1.0  --pattern noise

Compare runs on ``zmp_error mean``. Authority is adequate if a full-scale
constant residual shifts it by >= 0.007 m (20% of the ~0.036 m operating error).
"""

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
  # Same reason as the baseline script: `step()` resets terminated envs in place
  # before returning, which would erase the episode this wants to measure.
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg, device=args.device)

  dim = env.action_manager.total_action_dim
  action = _build_action(args.pattern, args.level, env.num_envs, dim, env.device)

  # The ZMP error is read through the same sensor plumbing the reward uses, so
  # this measures exactly the quantity `zmp_tracking` is scored on -- in metres,
  # before the kernel flattens it.
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

    # Steps since the push term last actually fired, from the term's own
    # record. Watching the event manager's interval countdown instead -- the
    # obvious way, and the way this was written first -- counts pushes that
    # never landed: `EventManager` re-samples the countdown whenever the term
    # fires, including the ticks `push_and_record` suppresses during
    # `PUSH_WARMUP_S`. `steps_since_push` also reports a huge age for a push
    # older than the current episode, so a freshly reset env drops out of `keep`
    # below instead of binning its ~4 s posture settle as push recovery.
    age = mdp.steps_since_push(env)

    slot = (age // BIN_STEPS).clamp(max=NUM_BINS - 1)
    # `age >= 1` is `recovery_tracking`'s own gate: at age 0 the push has been
    # applied but no physics has run on it yet, so that sample predates its
    # effect. Matching the gate keeps these bins aligned with the steps the
    # reward is paid on, which is what the profile is used to size.
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
  # Bins are dropped below n=50, so "no bin reached 3 s" is a normal outcome of
  # a short run or of episodes that end before the profile fills -- and it used
  # to raise `StatisticsError` here, after `env.close()` and before `--dump`,
  # losing the whole run. The bars are relative to this level, so they go too.
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
