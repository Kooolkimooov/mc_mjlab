"""Does the DCM objective still prefer standing? -- docs/reward-shaping.md#dcm_stability."""

from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv

from mc_mjlab import MC_RTC_YAML_PATH
from mc_mjlab.tasks import mdp
from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import (
  DCM_STD,
  _make_env_cfg,
)

#: Candidate kernel widths to score the measured distribution through.
STD_CANDIDATES = (0.03, 0.04, 0.05, 0.06, 0.08, 0.10)

#: Bin width for the push profile, in policy steps (50 Hz -> 5 bins/100 ms).
BIN_STEPS = 5
NUM_BINS = 40


def posture_config(source: Path, out_dir: Path) -> Path:
  """A copy of the mc_rtc config running the posture controller, for the standing arm."""
  lines = []
  for raw in source.read_text().splitlines():
    head = raw.split("#", 1)[0].rstrip()
    lines.append("Enabled: Posture" if head.startswith("Enabled:") else raw)
  path = out_dir / "mc_rtc_posture.yaml"
  path.write_text("\n".join(lines) + "\n")
  return path


class Sample:
  """Grounded per-step readings of one regime, pooled over envs and steps."""

  def __init__(self) -> None:
    self.corrected: list[float] = []
    self.legacy: list[float] = []
    self.commanded_speed: list[float] = []
    self.speed_deficit: list[float] = []
    self.bin_sum = [0.0] * NUM_BINS
    self.bin_count = [0] * NUM_BINS

  def add(
    self,
    corrected: torch.Tensor,
    legacy: torch.Tensor,
    speed: torch.Tensor,
    deficit: torch.Tensor,
    age: torch.Tensor,
  ) -> None:
    """Record one step's grounded samples, binned by time since the last push."""
    self.corrected += corrected.tolist()
    self.legacy += legacy.tolist()
    self.commanded_speed += speed.tolist()
    self.speed_deficit += deficit.tolist()
    slots = (age // BIN_STEPS).clamp(max=NUM_BINS - 1)
    keep = (age >= 1) & (age < NUM_BINS * BIN_STEPS)
    for slot, value in zip(slots[keep].tolist(), corrected[keep].tolist(), strict=True):
      self.bin_sum[slot] += value
      self.bin_count[slot] += 1

  def score(self, std: float) -> float:
    """Mean ``exp(-(e/std)^2)`` over the sample -- the reward the term would pay."""
    return statistics.fmean(math.exp(-((e / std) ** 2)) for e in self.corrected)

  def legacy_score(self, std: float) -> float:
    """The same kernel on the pre-2026-08-19 zero-referenced error."""
    return statistics.fmean(math.exp(-((e / std) ** 2)) for e in self.legacy)


def quantiles(values: list[float]) -> dict[str, float]:
  """Mean, median, p75, p90 and p99 of a sample."""
  ordered = sorted(values)
  pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
  return {
    "mean": statistics.fmean(ordered),
    "median": statistics.median(ordered),
    "p75": pick(0.75),
    "p90": pick(0.90),
    "p99": pick(0.99),
  }


def run_regime(
  name: str, config: Path, level: float, args: argparse.Namespace
) -> Sample:
  """Step one regime with a fixed residual level and collect its grounded samples."""
  cfg = _make_env_cfg(
    control="position",
    num_envs=args.num_envs,
    num_workers=args.num_workers,
    console_output="none",
    mc_rtc_yaml=config,
  )
  # `step()` would otherwise reset in place and erase what this measures.
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg, device=args.device)
  sensors = mdp._ZmpSensors(env, mdp.GROUND_CONTACT_SENSORS, "robot")
  term = mdp._residual_term(env, "mc_rtc_residual")
  root = env.scene["robot"].indexing.root_body_id

  signs = torch.ones(env.action_manager.total_action_dim, device=env.device)
  signs[1::2] = -1.0
  action = signs.expand(env.num_envs, -1) * level

  sample = Sample()
  env.reset()
  print(f"[dcm] {name}: {args.steps} steps, residual level {level:g}", flush=True)
  started = time.monotonic()
  for step in range(args.steps):
    _, _, terminated, time_outs, _ = env.step(action)
    if step >= args.warmup_steps:
      measured, normal_force = sensors.measured_offset(env)
      com = env.sim.data.subtree_com[:, root]
      com_vel = env.sim.data.subtree_linvel[:, root]
      commanded = term.controller_vector("control_com_vel")[:, :2]
      omega = torch.sqrt(mdp.GRAVITY / com[:, 2].clamp(min=mdp.MIN_COM_HEIGHT))
      capture = com_vel[:, :2] / omega.unsqueeze(-1)
      grounded = normal_force >= 20.0
      sample.add(
        torch.linalg.vector_norm(
          capture - measured - commanded / omega.unsqueeze(-1), dim=1
        )[grounded],
        torch.linalg.vector_norm(capture - measured, dim=1)[grounded],
        torch.linalg.vector_norm(commanded, dim=1)[grounded],
        (
          torch.linalg.vector_norm(com_vel[:, :2], dim=1)
          - torch.linalg.vector_norm(commanded, dim=1)
        )[grounded],
        mdp.steps_since_push(env)[grounded],
      )
    # Outside the warm-up guard: `auto_reset=False` makes a missed reset fatal.
    done = (terminated | time_outs).nonzero(as_tuple=False).flatten()
    if done.numel():
      env.reset(env_ids=done)
  env.close()
  print(
    f"[dcm] {name}: {len(sample.corrected)} grounded samples in "
    f"{time.monotonic() - started:.0f}s",
    flush=True,
  )
  return sample


def report(samples: dict[str, Sample]) -> None:
  """Print the acceptance table the proposal's validation gate asks for."""
  print(f"\n{'=' * 78}\n  corrected DCM error, metres\n{'=' * 78}")
  print(
    f"  {'regime':<20} {'n':>8} {'mean':>8} {'median':>8} {'p75':>8} "
    f"{'p90':>8} {'p99':>8}"
  )
  for name, s in samples.items():
    q = quantiles(s.corrected)
    print(
      f"  {name:<20} {len(s.corrected):8d} {q['mean']:8.4f} {q['median']:8.4f} "
      f"{q['p75']:8.4f} {q['p90']:8.4f} {q['p99']:8.4f}"
    )

  print("\n  commanded CoM speed, m/s (the reference the error is taken against)")
  for name, s in samples.items():
    q = quantiles(s.commanded_speed)
    print(f"  {name:<20} mean {q['mean']:.4f}   median {q['median']:.4f}")

  print(
    f"\n{'=' * 78}\n  score at each candidate std: corrected vs the old "
    f"zero-referenced error\n{'=' * 78}"
  )
  header = "  ".join(f"{std:>8.2f}" for std in STD_CANDIDATES)
  print(f"  {'regime':<20} {'kernel':<10} {header}")
  for name, s in samples.items():
    for label, fn in (("corrected", s.score), ("legacy", s.legacy_score)):
      row = "  ".join(f"{fn(std):8.3f}" for std in STD_CANDIDATES)
      print(f"  {name:<20} {label:<10} {row}")

  if "walking" in samples and "standing" in samples:
    print(
      f"\n{'=' * 78}\n  standing edge over walking (the defect being fixed; "
      f"gate: |edge| < 10%)\n{'=' * 78}"
    )
    walk, stand = samples["walking"], samples["standing"]
    print(f"  {'std':>8} {'corrected':>22} {'legacy':>22}")
    for std in STD_CANDIDATES:
      edge = stand.score(std) / walk.score(std) - 1.0
      old = stand.legacy_score(std) / walk.legacy_score(std) - 1.0
      print(f"  {std:8.2f} {edge:+22.1%} {old:+22.1%}")

  if "walking+residual" in samples and "walking" in samples:
    base = quantiles(samples["walking"].commanded_speed)["mean"]
    moved = quantiles(samples["walking+residual"].commanded_speed)["mean"]
    print(
      f"\n  residual authority over the *target*: commanded speed "
      f"{base:.4f} -> {moved:.4f} m/s ({(moved - base) / max(base, 1e-9):+.1%})"
    )

  walk = samples.get("walking")
  if walk is not None:
    print(
      f"\n{'=' * 78}\n  does resisting the gait now cost? error by speed deficit "
      f"(walking)\n{'=' * 78}"
    )
    print(
      f"  {'measured - commanded speed, m/s':<34} {'n':>7} "
      f"{'corrected':>10} {'legacy':>10}"
    )
    bands = ((-9.0, -0.05), (-0.05, -0.02), (-0.02, 0.02), (0.02, 0.05), (0.05, 9.0))
    for low, high in bands:
      rows = [
        (c, legacy)
        for c, legacy, d in zip(
          walk.corrected, walk.legacy, walk.speed_deficit, strict=True
        )
        if low <= d < high
      ]
      if not rows:
        continue
      label = (
        f"  {low:+.2f} .. {high:+.2f}" if abs(low) < 9 else f"  slower than {high:+.2f}"
      )
      if low > 1.0 or high > 8.0:
        label = f"  faster than {low:+.2f}"
      print(
        f"{label:<36} {len(rows):7d} "
        f"{statistics.fmean(c for c, _ in rows):10.4f} "
        f"{statistics.fmean(le for _, le in rows):10.4f}"
      )

    print("\n  corrected error by time since a push (walking)")
    for i, (total, count) in enumerate(zip(walk.bin_sum, walk.bin_count, strict=True)):
      if count < 50:
        continue
      print(f"    {i * BIN_STEPS * 0.02:5.2f}s  {total / count:.4f} m  n={count:6d}")

  if walk is not None:
    print("\n  std for a target nominal score (std = q / sqrt(-log s)):")
    q = quantiles(walk.corrected)
    for target in (0.5, 0.571, 0.6):
      std = q["median"] / math.sqrt(-math.log(target))
      print(
        f"    score {target:.3f} at the walking median "
        f"({q['median']:.4f} m) -> std {std:.4f}"
      )
    print(f"    current DCM_STD = {DCM_STD}")


def main() -> None:
  p = argparse.ArgumentParser(
    description=(
      "Replay standing, walking and pushed walking with a zero residual and "
      "check that the command-relative DCM objective scores them alike."
    ),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  p.add_argument("--num-envs", type=int, default=16)
  p.add_argument("--num-workers", type=int, default=8)
  p.add_argument("--steps", type=int, default=1200, help="policy steps per regime")
  p.add_argument(
    "--warmup-steps",
    type=int,
    default=100,
    help="steps to discard while the controller settles into its gait",
  )
  p.add_argument("--device", default="cuda:0")
  p.add_argument(
    "--residual-level",
    type=float,
    default=1.0,
    help="residual fraction of the clip for the third arm; 0 skips that arm",
  )
  p.add_argument(
    "--out-dir", default="/tmp", help="where the posture config is written"
  )
  args = p.parse_args()

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  regimes = [
    ("walking", MC_RTC_YAML_PATH, 0.0),
    ("standing", posture_config(MC_RTC_YAML_PATH, out_dir), 0.0),
  ]
  if args.residual_level:
    regimes.append(("walking+residual", MC_RTC_YAML_PATH, args.residual_level))

  samples = {name: run_regime(name, cfg, level, args) for name, cfg, level in regimes}
  report(samples)


if __name__ == "__main__":
  main()
