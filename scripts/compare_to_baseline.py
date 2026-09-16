"""Score a checkpoint against the zero-residual controller -- docs/evaluation.md."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, TextIO

import torch
from evaluation.comparison import (
  Arm,
  ComparisonEpisode,
  by_stratum,
  describe,
  survival,
  two_proportion_p,
  welch_p,
  wilson,
)
from evaluation.rollout import (
  episode_snapshot,
  managed_env,
  metric_snapshot,
  reset_done,
)
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from mc_mjlab.tasks.evaluation import (
  EMPTY,
  TaskEvaluation,
  evaluation_for,
  stratum_labels,
)
from mc_mjlab.tasks.residual_balance.evaluation import RESIDUAL_BALANCE_EVALUATION
from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import (
  AUTHORITY_SETS,
  make_residual_balance_env_cfg,
)
from mc_mjlab.tasks.residual_balance.residual_balance_ppo_cfg import (
  residual_balance_ppo_cfg,
)
from mc_mjlab.tasks.residual_balance.residual_balance_runner import (
  ResidualBalanceOnPolicyRunner,
)


def _apply_scale(cfg: ManagerBasedRlEnvCfg, num_envs: int, num_workers: int) -> None:
  """Point a registered cfg at this comparison's environment and worker counts."""
  cfg.scene.num_envs = num_envs
  for action in cfg.actions.values():
    if hasattr(action, "num_workers"):
      action.num_workers = num_workers  # ty: ignore[invalid-assignment]
    if hasattr(action, "console_output"):
      action.console_output = "none"  # ty: ignore[invalid-assignment]


#: Where every comparison's CSV and printed report land, beside the training logs
#: they score. Kept out of a scratch dir: these are the record's raw data.
COMPARISON_DIR = Path("logs/comparisons")


class _Tee:
  """Write the report to the terminal and to the run's log file at once."""

  def __init__(self, stream: TextIO, path: Path) -> None:
    self._stream = stream
    # Line-buffered, so a killed run still leaves everything it printed.
    self._file = path.open("w", buffering=1)

  def write(self, text: str) -> int:
    self._stream.write(text)
    return self._file.write(text)

  def flush(self) -> None:
    self._stream.flush()
    self._file.flush()

  def __getattr__(self, name: str) -> Any:
    # `isatty`, `fileno` and friends: mjlab colourises on the first and mc_rtc's
    # fd redirection needs the second, so this must stay a real stdout otherwise.
    return getattr(self._stream, name)


def output_stem(checkpoint: str) -> str:
  """``<run directory>_<model>``: unique per checkpoint, sorted beside its siblings."""
  path = Path(checkpoint)
  return f"{path.parent.name}_{path.stem}"


def _run_both(
  env: ManagerBasedRlEnv,
  wrapped: RslRlVecEnvWrapper,
  policy: Any,
  minutes: float,
  policy_ids: Sequence[int],
  skip_steps: int = 0,
  metric_names: Sequence[str] = (),
) -> tuple[Arm, Arm]:
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
  # Reward accrued before the gait exists is a cost both arms pay identically, so
  # it only dilutes the delta. docs/evaluation.md#skip_s
  skip_sums = {n: torch.zeros(env.num_envs, device=env.device) for n in reward_terms}
  dropped = 0

  env.reset()

  deadline = time.monotonic() + minutes * 60.0
  while time.monotonic() < deadline:
    action.zero_()
    with torch.inference_mode():
      # Masked, not sliced: the actor is batched, so wasted rows are cheaper.
      action[is_policy] = policy(wrapped.get_observations())[is_policy]
    _, _, terminated, time_outs, _ = env.step(action)
    policy.reset(terminated | time_outs)
    base.steps += 1
    pol.steps += 1

    if skip_steps:
      # `episode_length_buf` increments by one per step, so each env crosses the
      # threshold exactly once per episode.
      at = (env.episode_length_buf == skip_steps).nonzero(as_tuple=False).flatten()
      if at.numel():
        for n in reward_terms:
          skip_sums[n][at] = env.reward_manager._episode_sums[n][at]

    done = (terminated | time_outs).nonzero(as_tuple=False).flatten()
    if done.numel() == 0:
      continue

    snapshot = episode_snapshot(env, done)
    sums, lengths, flags = snapshot.rewards, snapshot.lengths, snapshot.terminations
    # Read before `reset_done` below: reset clears the metric buffers too.
    metrics = metric_snapshot(env, done) if metric_names else {}

    for i, env_id in enumerate(done.tolist()):
      length = int(lengths[i]) - skip_steps
      counter[env_id] += 1
      if length <= 0:
        # Never reached the walking phase; it has no scoreable window at all.
        dropped += 1
        continue
      (pol if bool(is_policy[env_id]) else base).episodes.append(
        ComparisonEpisode(
          env_id=env_id,
          nth=counter[env_id] - 1,
          length=length,
          terms={n: int(flags[n][i]) for n in term_names},
          rewards={
            n: float((sums[n][i] - skip_sums[n][env_id]).item()) for n in reward_terms
          },
          metrics={n: float(metrics[n][i]) for n in metric_names},
        )
      )
    for n in reward_terms:
      skip_sums[n][done] = 0.0

    reset_done(env, done)

  if skip_steps and dropped:
    print(
      f"[compare] dropped {dropped} episodes shorter than the {skip_steps}-step skip"
    )
  return base, pol


def _resolve_evaluation(
  task_id: str | None, env: ManagerBasedRlEnv
) -> tuple[TaskEvaluation, list[str]]:
  """The task's declaration, narrowed to the metrics this environment provides."""
  evaluation = evaluation_for(task_id) if task_id else RESIDUAL_BALANCE_EVALUATION
  available = set(env.metrics_manager.active_terms)
  metrics = [name for name in evaluation.metrics if name in available]
  missing = [name for name in evaluation.metrics if name not in available]
  if missing:
    print(f"[compare] task metrics this env does not provide: {', '.join(missing)}")
  narrowed = replace(
    evaluation,
    metrics=tuple(metrics),
    success=evaluation.success if evaluation.success in metrics else None,
    stratify=evaluation.stratify if evaluation.stratify in metrics else None,
    strata=evaluation.strata if evaluation.stratify in metrics else (),
  )
  if metrics:
    print(f"[compare] task metrics: {', '.join(metrics)}")
  return narrowed, metrics


def _report_task(
  b_eps: Sequence[ComparisonEpisode],
  p_eps: Sequence[ComparisonEpisode],
  evaluation: TaskEvaluation,
  metric_names: Sequence[str],
) -> None:
  """The half of the report the task defines: its metrics, verdict and strata."""
  if not metric_names:
    return

  print("\n  task metrics per episode (the task's own declaration)")
  print(f"    {'metric':<24} {'baseline':>10} {'policy':>10} {'delta':>10} {'p':>10}")
  print("    " + "-" * 68)
  for name in metric_names:
    b = describe([e.metrics[name] for e in b_eps])
    p = describe([e.metrics[name] for e in p_eps])
    print(
      f"    {name:<24} {b['mean']:10.4f} {p['mean']:10.4f} "
      f"{p['mean'] - b['mean']:+10.4f} {welch_p(b, p):10.2e}"
    )

  success = evaluation.success
  if success is not None:
    bk = sum(round(e.metrics[success]) for e in b_eps)
    pk = sum(round(e.metrics[success]) for e in p_eps)
    bl, bh = wilson(bk, len(b_eps))
    pl, ph = wilson(pk, len(p_eps))
    print(f"\n  {success} rate")
    print(
      f"    {'baseline':<10} {bk / len(b_eps):7.1%}  [{bl:5.1%},{bh:5.1%}]  n={len(b_eps)}"
    )
    print(
      f"    {'policy':<10} {pk / len(p_eps):7.1%}  [{pl:5.1%},{ph:5.1%}]  n={len(p_eps)}"
    )
    print(
      f"    {'delta':<10} {pk / len(p_eps) - bk / len(b_eps):+7.1%}"
      f"   p = {two_proportion_p(pk, len(p_eps), bk, len(b_eps)):.2e}"
    )

  if evaluation.stratify is None:
    return

  key = evaluation.stratify
  labels = stratum_labels(evaluation.strata)
  b_buckets = by_stratum(b_eps, key, evaluation.strata)
  p_buckets = by_stratum(p_eps, key, evaluation.strata)
  headline = success or metric_names[0]
  print(f"\n  {headline} by {key}")
  print(
    f"    {'stratum':<14} {'n base':>7} {'n pol':>7} {'baseline':>10} {'policy':>10} {'delta':>10}"
  )
  for index, label in enumerate(labels):
    b_rows = b_buckets.get(index, [])
    p_rows = p_buckets.get(index, [])
    if not b_rows and not p_rows:
      continue
    b = (
      describe([e.metrics[headline] for e in b_rows])
      if b_rows
      else {"mean": float("nan")}
    )
    p = (
      describe([e.metrics[headline] for e in p_rows])
      if p_rows
      else {"mean": float("nan")}
    )
    print(
      f"    {label:<14} {len(b_rows):7d} {len(p_rows):7d} "
      f"{b['mean']:10.4f} {p['mean']:10.4f} {p['mean'] - b['mean']:+10.4f}"
    )
  print(
    "\n  Strata are the point of this table, not a breakdown: a mean over all of"
    "\n  them hides which regime the policy actually changed."
  )


def _report(
  base: Arm,
  pol: Arm,
  env: ManagerBasedRlEnv,
  cfg: ManagerBasedRlEnvCfg,
  term_names: Sequence[str],
  reward_terms: Sequence[str],
  evaluation: TaskEvaluation = EMPTY,
  metric_names: Sequence[str] = (),
) -> None:
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

  bk, bn = survival(b_eps)
  pk, pn = survival(p_eps)
  bl, bh = wilson(bk, bn)
  pl, ph = wilson(pk, pn)
  p_surv = two_proportion_p(pk, pn, bk, bn)
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

  bd = describe([e.length for e in b_eps])
  pd_ = describe([e.length for e in p_eps])
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
  pl_ = welch_p(bd, pd_)
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
    b, p = describe(bv), describe(pv)
    print(
      f"    {n:<22} {b['mean']:10.3f} {p['mean']:10.3f} "
      f"{p['mean'] - b['mean']:+10.3f} {welch_p(b, p):10.2e}"
    )
  print(
    "\n  Reward sums correlate strongly with episode length on this task, so a"
    "\n  reward delta that tracks the length delta is not independent evidence."
  )

  # The statistic that actually resolves the question, so it is printed last and
  # read first. docs/evaluation.md#per-step-rates-beat-survival
  print("\n  reward per step (episode length divided out)")
  print(f"    {'term':<22} {'baseline':>10} {'policy':>10} {'delta':>10} {'p':>10}")
  print("    " + "-" * 66)
  for n in rows:
    if n == "TOTAL":
      bv = [sum(e.rewards.values()) / e.length for e in b_eps]
      pv = [sum(e.rewards.values()) / e.length for e in p_eps]
    else:
      bv = [e.rewards[n] / e.length for e in b_eps]
      pv = [e.rewards[n] / e.length for e in p_eps]
    b, p = describe(bv), describe(pv)
    # A relative delta is undefined against a zero baseline -- the residual
    # penalties, which the baseline never pays.
    rel = f"{(p['mean'] - b['mean']) / b['mean']:+9.1%}" if b["mean"] else "        --"
    print(
      f"    {n:<22} {b['mean']:10.5f} {p['mean']:10.5f} {rel} {welch_p(b, p):10.2e}"
    )
  print(
    "\n  Survival is the headline but the weakest test here: it needs n~150 per"
    "\n  arm to resolve 5pp. Read the per-step rates first."
  )

  _report_task(b_eps, p_eps, evaluation, metric_names)


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
    "--feedback-modalities",
    default="",
    help="comma-separated feedback spaces for the residual-feedback task; the "
    "checkpoint's action width is set by these, so a mismatch fails to load",
  )
  p.add_argument(
    "--no-torque-channel",
    action="store_true",
    help="score a feedback-only residual-feedback checkpoint",
  )
  p.add_argument(
    "--nominal",
    action="store_true",
    help="evaluate on the true model: drop the startup randomization events and "
    "the actuator delay lag. Those draws are per-environment and fixed for the "
    "run, and the two arms hold different ones, so they are the dominant "
    "variance in a comparison (docs/evaluation.md#nominal)",
  )
  p.add_argument(
    "--skip-s",
    type=float,
    default=0.0,
    help="discard this many seconds at the start of every episode before "
    "scoring; the FSM stands for about 5.5 s after a controller reset, and "
    "that window dilutes the delta without biasing it "
    "(docs/evaluation.md#skip_s)",
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
  p.add_argument(
    "--authority-set",
    choices=AUTHORITY_SETS,
    default="uniform",
    help="residual joint/scaling screen the checkpoint was trained with",
  )
  p.add_argument(
    "--task",
    help="registered task id; resolves the env, PPO and runner from the "
    "registry instead of the residual-balance builders",
  )
  p.add_argument("--device", default="cuda:0", help="torch device for the simulation")
  p.add_argument(
    "--recovery-dcm-std",
    type=float,
    default=None,
    help="override the recovery DCM width when evaluating a reward-shape screen",
  )
  p.add_argument(
    "--drop-obs",
    default="",
    help=(
      "comma-separated observation terms to remove before loading, for "
      "checkpoints that predate them; only sound for terms appended last"
    ),
  )
  p.add_argument(
    "--out-dir",
    default=str(COMPARISON_DIR),
    metavar="PATH",
    help="directory for the per-episode CSV and the printed report; both are "
    "named after the checkpoint, so a rerun overwrites only its own pair",
  )
  p.add_argument(
    "--dump",
    default=None,
    metavar="PATH",
    help="override the CSV path (arm, env, nth_in_env, length, termination "
    "flags, reward terms); by default it lands in --out-dir",
  )
  args = p.parse_args()

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  stem = output_stem(args.checkpoint)
  dump_path = Path(args.dump) if args.dump else out_dir / f"{stem}.csv"
  # Replaced wholesale rather than wrapped in redirect_stdout: mc_rtc prints
  # through fd 1 regardless, and this keeps the report and its header together.
  sys.stdout = _Tee(sys.stdout, out_dir / f"{stem}.log")

  if args.task:
    # Any registered task, scored the same way: the report reads its reward and
    # termination terms from the live managers. docs/evaluation.md#--task
    cfg = load_env_cfg(args.task)
    _apply_scale(cfg, args.num_envs, args.num_workers)
    agent_cfg = asdict(load_rl_cfg(args.task))
    runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  else:
    cfg = make_residual_balance_env_cfg(
      control=args.control,
      num_envs=args.num_envs,
      num_workers=args.num_workers,
      console_output="none",
      authority_set=args.authority_set,
    )
    agent_cfg = asdict(residual_balance_ppo_cfg())
    runner_cls = ResidualBalanceOnPolicyRunner

  if args.recovery_dcm_std is not None:
    if "recovery_dcm" not in cfg.rewards:
      p.error("--recovery-dcm-std needs a task with a recovery_dcm reward")
    cfg.rewards["recovery_dcm"].params["std"] = args.recovery_dcm_std

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
  names = [n.strip() for n in args.feedback_modalities.split(",") if n.strip()]
  if names or args.no_torque_channel:
    action_cfg = cfg.actions["mc_rtc_residual"]
    if names:
      action_cfg.feedback_modalities = tuple(names)
    if args.no_torque_channel:
      action_cfg.torque_channel = False
    print(
      f"[compare] residual-feedback action: {action_cfg.feedback_modalities}, "
      f"torque_channel={action_cfg.torque_channel}"
    )
  if args.nominal:
    for name in (
      "randomize_friction",
      "randomize_pd_gains",
      "randomize_effort",
      "encoder_bias",
    ):
      cfg.events.pop(name, None)
    for entity in cfg.scene.entities.values():
      for actuator in getattr(entity, "articulation", entity).actuators:
        actuator.delay_max_lag = 0
    print("[compare] nominal model: startup randomization and delay lag removed")
  cfg.auto_reset = False
  with managed_env(cfg, args.device) as env:
    # The wrapper exists for two things: the runner's constructor reads its
    # shapes, and `get_observations()` assembles the actor's observation group.
    # Stepping still goes through the raw env, because the wrapper collapses
    # `terminated` and `time_outs` into one `dones` and this needs them apart to
    # tell a fall from a survival.
    wrapped = RslRlVecEnvWrapper(env)
    runner = runner_cls(wrapped, agent_cfg, device=args.device)
    runner.load(
      args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device
    )
    policy = runner.get_inference_policy(device=args.device)

    term_names = env.termination_manager.active_terms
    reward_terms = env.reward_manager.active_terms
    evaluation, metric_names = _resolve_evaluation(args.task, env)
    print(
      f"[compare] {args.num_envs} envs, {args.num_workers} workers, "
      f"episode {cfg.episode_length_s:.0f} s, {args.minutes:.0f} min per arm\n"
      f"[compare] checkpoint {args.checkpoint}"
    )

    # Both arms step together, split by env index, so they share the wall-clock
    # window and the worker pool instead of running in sequence.
    policy_ids = list(range(env.num_envs // 2, env.num_envs))
    base, pol = _run_both(
      env,
      wrapped,
      policy,
      args.minutes,
      policy_ids,
      round(args.skip_s / env.step_dt),
      metric_names,
    )
    print(
      f"[compare] done: {len(base.episodes)} baseline / {len(pol.episodes)} "
      f"policy episodes"
    )

  with open(dump_path, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(
      ["arm", "env", "nth_in_env", "length"]
      + list(term_names)
      + list(reward_terms)
      + list(metric_names)
    )
    for arm in (base, pol):
      for e in arm.episodes:
        w.writerow(
          [arm.label, e.env_id, e.nth, e.length]
          + [e.terms[n] for n in term_names]
          + [e.rewards[n] for n in reward_terms]
          + [e.metrics[n] for n in metric_names]
        )
  print(f"[compare] per-episode rows -> {dump_path}")

  if not base.episodes or not pol.episodes:
    print("[compare] an arm finished no episodes; raise --minutes")
    return
  _report(base, pol, env, cfg, term_names, reward_terms, evaluation, metric_names)


if __name__ == "__main__":
  main()
