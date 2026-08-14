"""Score a checkpoint against the zero-residual controller -- docs/evaluation.md."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import _make_env_cfg
from mc_mjlab.tasks.residual_balance.residual_balance_ppo_cfg import (
  residual_balance_ppo_cfg,
)


def _describe(values: Sequence[float]) -> dict[str, float]:
  """Mean/spread/quartiles for one per-episode quantity."""
  n = len(values)
  if n == 0:
    return dict.fromkeys(
      ("n", "mean", "std", "sem", "min", "q1", "median", "q3", "max"), float("nan")
    ) | {"n": 0}
  mean = statistics.fmean(values)
  std = statistics.stdev(values) if n > 1 else 0.0
  # numpy/pandas convention, so these paste into anything else.
  q1, med, q3 = (
    statistics.quantiles(values, n=4, method="inclusive")
    if n > 1
    else (values[0], values[0], values[0])
  )
  return {
    "n": n,
    "mean": mean,
    "std": std,
    "sem": std / math.sqrt(n) if n else 0.0,
    "min": min(values),
    "q1": q1,
    "median": med,
    "q3": q3,
    "max": max(values),
  }


def _wilson(k: int, n: int) -> tuple[float, float]:
  """95% interval for a proportion; behaves at 0/n and n/n, unlike the normal one."""
  if n == 0:
    return (float("nan"), float("nan"))
  z = 1.959963985
  p = k / n
  d = 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return (max(0.0, c - h), min(1.0, c + h))


def _two_proportion_p(k1: int, n1: int, k2: int, n2: int) -> float:
  """Two-sided p for equal proportions, pooled-variance normal approximation."""
  if n1 == 0 or n2 == 0:
    return float("nan")
  p = (k1 + k2) / (n1 + n2)
  se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
  if se == 0.0:
    return 1.0
  return math.erfc(abs(k1 / n1 - k2 / n2) / se / math.sqrt(2))


def _welch_p(a: dict[str, float], b: dict[str, float]) -> float:
  """Two-sided p that two means differ, from their standard errors."""
  se = math.hypot(a["sem"], b["sem"])
  if se == 0.0 or a["n"] < 2 or b["n"] < 2:
    return float("nan")
  return math.erfc(abs(a["mean"] - b["mean"]) / se / math.sqrt(2))


@dataclass
class Episode:
  env_id: int
  nth: int
  length: int
  terms: dict[str, int]
  rewards: dict[str, float]


@dataclass
class Arm:
  label: str
  #: Every env this arm owns, including ones that never finished an episode.
  env_ids: tuple[int, ...] = ()
  episodes: list[Episode] = field(default_factory=list)
  steps: int = 0

  def trimmed(self) -> tuple[list[Episode], int]:
    """The first K episodes of every env, K set by the env that finished fewest."""
    # K over `env_ids`, not over envs that finished: the latter drops survivors.
    # docs/evaluation.md#fixed-episodes-per-env-not-everything-that-finished
    per_env: dict[int, int] = dict.fromkeys(self.env_ids, -1)
    for e in self.episodes:
      per_env[e.env_id] = max(per_env.get(e.env_id, -1), e.nth)
    if not per_env:
      return [], 0
    k = min(per_env.values()) + 1  # nth is 0-based
    return [e for e in self.episodes if e.nth < k], k


def _run_both(env, wrapped, policy, minutes: float, policy_ids) -> tuple[Arm, Arm]:
  """Step both arms at once, split by env index -- docs/evaluation.md#both-arms-at-once."""
  policy_set = set(policy_ids)
  base = Arm(
    label="baseline",
    env_ids=tuple(i for i in range(env.num_envs) if i not in policy_set),
  )
  pol = Arm(label="policy", env_ids=tuple(policy_ids))
  reward_terms = env.reward_manager.active_terms
  term_names = env.termination_manager.active_terms
  action = torch.zeros(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )
  counter = [0] * env.num_envs
  is_policy = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  is_policy[policy_ids] = True

  env.reset()
  deadline = time.monotonic() + minutes * 60.0
  while time.monotonic() < deadline:
    action.zero_()
    with torch.inference_mode():
      # Masked, not sliced: the actor is batched, so wasted rows are cheaper.
      action[is_policy] = policy(wrapped.get_observations())[is_policy]
    _, _, terminated, time_outs, _ = env.step(action)
    base.steps += 1
    pol.steps += 1
    done = (terminated | time_outs).nonzero(as_tuple=False).flatten()
    if done.numel() == 0:
      continue
    sums = {n: env.reward_manager._episode_sums[n][done] for n in reward_terms}
    lengths = env.episode_length_buf[done].tolist()
    flags = {n: env.termination_manager.get_term(n)[done].tolist() for n in term_names}
    for i, env_id in enumerate(done.tolist()):
      (pol if bool(is_policy[env_id]) else base).episodes.append(
        Episode(
          env_id=env_id,
          nth=counter[env_id],
          length=int(lengths[i]),
          terms={n: int(flags[n][i]) for n in term_names},
          rewards={n: float(sums[n][i].item()) for n in reward_terms},
        )
      )
      counter[env_id] += 1
    _reset_done(env, done)
  return base, pol


def _reset_done(env, env_ids) -> None:
  """Recycle finished envs without pushing a second observation-history frame."""
  # `reset()` would append a frame for *every* env, halving the history's span.
  # docs/evaluation.md#resetting-without-corrupting-the-observation-history
  env._reset_idx(env_ids)
  env.scene.write_data_to_sim()
  env.sim.forward()


def _survival(eps: Sequence[Episode]) -> tuple[int, int]:
  return sum(e.terms.get("time_out", 0) for e in eps), len(eps)


def _report(base: Arm, pol: Arm, env, cfg, term_names, reward_terms) -> None:
  b_eps, b_k = base.trimmed()
  p_eps, p_k = pol.trimmed()
  dt = env.step_dt
  print(
    f"\n{'=' * 78}\n"
    f"paired comparison -- {len(b_eps)} vs {len(p_eps)} episodes "
    f"(first {b_k} and {p_k} per env; "
    f"{len(base.episodes)} and {len(pol.episodes)} finished in total)\n"
    f"{'=' * 78}"
  )
  if not b_eps or not p_eps:
    # K is the *minimum* over the arm's envs, so one env still inside its first
    # episode at the deadline takes it to zero. Reporting the episodes that did
    # finish instead would be exactly the 1/duration bias this guards against
    # -- the envs missing are the survivors.
    print(
      "\n  Not enough to compare: at least one env had not finished an episode\n"
      "  when the clock ran out, and dropping it would bias the sample toward\n"
      "  short episodes. Raise --minutes (or lower --num-envs)."
    )
    return

  bk, bn = _survival(b_eps)
  pk, pn = _survival(p_eps)
  bl, bh = _wilson(bk, bn)
  pl, ph = _wilson(pk, pn)
  p_surv = _two_proportion_p(pk, pn, bk, bn)
  print("\n  survival to the episode cap")
  print(
    f"    {'baseline':<10} {bk / bn if bn else float('nan'):7.1%}  [{bl:5.1%},{bh:5.1%}]  n={bn}"
  )
  print(
    f"    {'policy':<10} {pk / pn if pn else float('nan'):7.1%}  [{pl:5.1%},{ph:5.1%}]  n={pn}"
  )
  print(
    f"    {'delta':<10} {(pk / pn - bk / bn) if bn and pn else float('nan'):+7.1%}"
    f"   p = {p_surv:.2e}{'  ***' if p_surv < 1e-3 else ('  **' if p_surv < 0.01 else ('  *' if p_surv < 0.05 else ''))}"
  )

  bd = _describe([e.length for e in b_eps])
  pd_ = _describe([e.length for e in p_eps])
  print(
    f"\n  episode length in steps (cap {env.max_episode_length:.0f} = {cfg.episode_length_s:.0f} s)"
  )
  print(
    f"    {'':<10} {'mean':>9} {'sem':>7} {'median':>8} {'q1':>8} {'q3':>8} {'seconds':>9}"
  )
  for name, s in (("baseline", bd), ("policy", pd_)):
    print(
      f"    {name:<10} {s['mean']:9.1f} {s['sem']:7.1f} {s['median']:8.1f}"
      f" {s['q1']:8.1f} {s['q3']:8.1f} {s['mean'] * dt:9.1f}"
    )
  pl_ = _welch_p(bd, pd_)
  print(f"    {'delta':<10} {pd_['mean'] - bd['mean']:+9.1f}   p = {pl_:.2e}")

  print("\n  termination breakdown (share of episodes)")
  print(f"    {'term':<22} {'baseline':>10} {'policy':>10} {'delta':>10}")
  for n in term_names:
    b = sum(e.terms[n] for e in b_eps) / len(b_eps) if b_eps else float("nan")
    p = sum(e.terms[n] for e in p_eps) / len(p_eps) if p_eps else float("nan")
    print(f"    {n:<22} {b:10.1%} {p:10.1%} {p - b:+10.1%}")

  print("\n  reward per episode")
  print(f"    {'term':<22} {'baseline':>10} {'policy':>10} {'delta':>10} {'p':>10}")
  print("    " + "-" * 66)
  rows = list(reward_terms) + ["TOTAL"]
  for n in rows:
    if n == "TOTAL":
      bv = [sum(e.rewards.values()) for e in b_eps]
      pv = [sum(e.rewards.values()) for e in p_eps]
    else:
      bv = [e.rewards[n] for e in b_eps]
      pv = [e.rewards[n] for e in p_eps]
    b, p = _describe(bv), _describe(pv)
    print(
      f"    {n:<22} {b['mean']:10.3f} {p['mean']:10.3f} "
      f"{p['mean'] - b['mean']:+10.3f} {_welch_p(b, p):10.2e}"
    )
  print(
    "\n  Reward sums correlate strongly with episode length on this task, so a"
    "\n  reward delta that tracks the length delta is not independent evidence."
  )


def main() -> None:
  p = argparse.ArgumentParser(
    description=(
      "Score a checkpoint against the zero-residual mc_rtc controller in one "
      "paired run, and print the comparison table."
    ),
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  p.add_argument(
    "--checkpoint",
    required=True,
    help="policy checkpoint to score, e.g. logs/.../model_950.pt",
  )
  p.add_argument(
    "--num-envs",
    type=int,
    default=16,
    help="parallel environments, split evenly between the two arms",
  )
  p.add_argument(
    "--num-workers",
    type=int,
    default=6,
    help="mc_rtc worker processes hosting the controllers; the default is low "
    "so this can run beside a training job, which already holds cpu_count - 2",
  )
  p.add_argument(
    "--minutes",
    type=float,
    default=8.0,
    help="wall-clock budget for the whole run; both arms step together on "
    "half the envs each, so each arm gets this long at --num-envs/2",
  )
  p.add_argument(
    "--control",
    default="position",
    choices=("position", "torque"),
    help="action space the checkpoint was trained on; a mismatch fails to load",
  )
  p.add_argument("--device", default="cuda:0", help="torch device for the simulation")
  p.add_argument(
    "--drop-obs",
    default="",
    help=(
      "comma-separated observation terms to remove before loading, for "
      "checkpoints that predate them; only sound for terms appended last"
    ),
  )
  p.add_argument(
    "--dump",
    default=None,
    metavar="PATH",
    help="write every episode of both arms to this CSV (arm, env, nth_in_env, "
    "length, termination flags, reward terms) for analysis the table omits",
  )
  args = p.parse_args()

  cfg = _make_env_cfg(
    control=args.control,
    num_envs=args.num_envs,
    num_workers=args.num_workers,
    console_output="none",
  )
  # A checkpoint is only loadable against the observation space it was trained
  # on: the actor's first layer and its `obs_normalizer` are both sized by the
  # concatenated width, so adding an observation term retires every checkpoint
  # that predates it with a `size mismatch` at load. Removing one from the
  # *middle* of a group would silently shift later terms into the wrong column
  # rather than fail, so this is only sound for terms appended at the end.
  for name in (n.strip() for n in args.drop_obs.split(",") if n.strip()):
    for group in cfg.observations.values():
      group.terms.pop(name, None)
  # Without this every number reads zero: `step()` resets terminated envs *in
  # place* before returning, zeroing both `episode_length_buf` and the reward
  # manager's `_episode_sums`, so the episode being measured is erased before
  # the caller sees `terminated`.
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg, device=args.device)

  # The wrapper exists for two things: the runner's constructor reads its
  # shapes, and `get_observations()` assembles the actor's observation group.
  # Stepping still goes through the raw env, because the wrapper collapses
  # `terminated` and `time_outs` into one `dones` and this needs them apart to
  # tell a fall from a survival.
  wrapped = RslRlVecEnvWrapper(env)
  runner = MjlabOnPolicyRunner(
    wrapped, asdict(residual_balance_ppo_cfg()), device=args.device
  )
  runner.load(
    args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device
  )
  policy = runner.get_inference_policy(device=args.device)

  term_names = env.termination_manager.active_terms
  reward_terms = env.reward_manager.active_terms
  print(
    f"[compare] {args.num_envs} envs, {args.num_workers} workers, "
    f"episode {cfg.episode_length_s:.0f} s, {args.minutes:.0f} min per arm\n"
    f"[compare] checkpoint {args.checkpoint}"
  )

  # Both arms step together, split by env index, so they share the wall-clock
  # window and the worker pool instead of running in sequence.
  policy_ids = list(range(env.num_envs // 2, env.num_envs))
  base, pol = _run_both(env, wrapped, policy, args.minutes, policy_ids)
  print(
    f"[compare] done: {len(base.episodes)} baseline / {len(pol.episodes)} "
    f"policy episodes"
  )
  env.close()

  if args.dump:
    with open(args.dump, "w", newline="") as fh:
      w = csv.writer(fh)
      w.writerow(
        ["arm", "env", "nth_in_env", "length"] + list(term_names) + list(reward_terms)
      )
      for arm in (base, pol):
        for e in arm.episodes:
          w.writerow(
            [arm.label, e.env_id, e.nth, e.length]
            + [e.terms[n] for n in term_names]
            + [e.rewards[n] for n in reward_terms]
          )
    print(f"[compare] per-episode rows -> {args.dump}")

  if not base.episodes or not pol.episodes:
    print("[compare] an arm finished no episodes; raise --minutes")
    return
  _report(base, pol, env, cfg, term_names, reward_terms)


if __name__ == "__main__":
  main()
