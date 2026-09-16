"""Qualify every validation checkpoint on paired deterministic scenarios."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from dataclasses import asdict
from pathlib import Path

import torch
from evaluation.disturbances import PairedDisturbances
from evaluation.qualification import promotion, summarize, summarize_strata
from evaluation.qualification_strata import (
  StratifiedDiagnostics,
  StratumRecord,
)
from evaluation.records import QualificationEpisode
from evaluation.rollout import (
  episode_snapshot,
  managed_env,
  metric_snapshot,
  reset_done,
)
from evaluation.scenarios import DISTURBANCE_WARMUP_S
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper

from mc_mjlab.actions.mc_rtc_residual_action import McRtcResidualActionBase
from mc_mjlab.tasks.residual_balance.curriculum_stages import (
  ACHIEVEMENT_STAGES,
  MINIMUM_QUALIFICATION_SEEDS,
  REQUIRED_QUALIFICATION_SCENARIOS,
  achievement_contract,
  achievement_contract_sha256,
)
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

SCENARIOS = ("nominal", "current_kick", "finite_impulse", "robust")
#: Must stay in step with the env cfg's own authority-set choices.


def resolve_checkpoints(inputs: list[str]) -> list[Path]:
  """Expand checkpoint paths, directories, and globs in iteration order."""
  paths: set[Path] = set()
  for value in inputs:
    path = Path(value)
    if path.is_dir():
      paths.update(path.glob("model_*.pt"))
    elif any(char in value for char in "*?["):
      paths.update(Path(item) for item in glob.glob(value))
    elif path.is_file():
      paths.add(path)
    else:
      raise FileNotFoundError(f"checkpoint input matches nothing: {value}")

  def key(path: Path) -> tuple[str, int, str]:
    match = re.search(r"model_(\d+)$", path.stem)
    return (str(path.parent), int(match.group(1)) if match else -1, path.name)

  return sorted((path.resolve() for path in paths), key=key)


def scenario_cfg(
  name: str, seed: int, args: argparse.Namespace
) -> ManagerBasedRlEnvCfg:
  """Build a deterministic qualification cfg for one scenario."""
  cfg = make_residual_balance_env_cfg(
    control=args.control,
    num_envs=args.num_envs,
    num_workers=args.num_workers,
    push_velocity=0.0,
    console_output="none",
    disturbance="none",
    authority_set=args.authority_set,
    randomization_stage=1 if name == "robust" else 0,
  )
  cfg.seed = seed
  cfg.auto_reset = False
  cfg.episode_length_s = args.episode_length_s
  cfg.events["reset_base"].params["pose_range"] = {}
  return cfg


def run_checkpoint(
  checkpoint: Path, scenario: str, seed: int, args: argparse.Namespace
) -> tuple[list[QualificationEpisode], list[StratumRecord]]:
  """Run one checkpoint and scenario to a fixed paired episode count."""
  torch.manual_seed(seed)
  cfg = scenario_cfg(scenario, seed, args)
  with managed_env(cfg, args.device) as env:
    wrapped = RslRlVecEnvWrapper(env)

    runner = ResidualBalanceOnPolicyRunner(
      wrapped,
      asdict(residual_balance_ppo_cfg()),
      device=args.device,
    )
    runner.load(
      str(checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=args.device,
    )

    policy = runner.get_inference_policy(device=args.device)
    residual_term = env.action_manager.get_term("mc_rtc_residual")
    if not isinstance(residual_term, McRtcResidualActionBase):
      raise TypeError(
        f"unexpected residual action type: {type(residual_term).__name__}"
      )

    recovery_s = float(
      env.reward_manager.get_term_cfg("recovery_dcm").params["window_s"]
    )
    stratified = StratifiedDiagnostics(
      env, residual_term, DISTURBANCE_WARMUP_S, recovery_s
    )
    disturbances = PairedDisturbances(env, scenario, seed, args.achievement_stage)

    counts = [[0, 0] for _ in range(env.num_envs)]
    current_policy = torch.tensor(
      [bool(env_id % 2) for env_id in range(env.num_envs)],
      dtype=torch.bool,
      device=env.device,
    )
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    pairs = [0] * env.num_envs
    action = torch.zeros(
      env.num_envs, env.action_manager.total_action_dim, device=env.device
    )
    episodes: list[QualificationEpisode] = []
    strata: list[StratumRecord] = []
    term_names = env.termination_manager.active_terms
    reward_names = env.reward_manager.active_terms

    env.reset()
    while bool(active.any()):
      disturbances.before_step(active, pairs, env.episode_length_buf)
      action.zero_()

      if bool((current_policy & active).any()):
        with torch.inference_mode():
          proposed = policy(wrapped.get_observations())
        action[current_policy & active] = proposed[current_policy & active]
      _, _, terminated, time_outs, _ = env.step(action)
      policy.reset(terminated | time_outs)
      disturbances.after_step()
      stratified.capture(active)

      done = (terminated | time_outs).nonzero(as_tuple=False).flatten()
      if done.numel() == 0:
        continue

      metric_values = metric_snapshot(env, done)
      snapshot = episode_snapshot(env, done)
      reward_values = {
        name: values.tolist() for name, values in snapshot.rewards.items()
      }
      termination_terms = {
        name: env.termination_manager.get_term(name) for name in term_names
      }
      termination_values = snapshot.terminations
      lengths = snapshot.lengths

      strata.extend(
        stratified.finish(
          str(checkpoint),
          scenario,
          seed,
          done,
          pairs,
          current_policy,
          termination_terms,
        )
      )

      for row, env_id in enumerate(done.tolist()):
        if not bool(active[env_id]):
          continue
        arm_index = int(current_policy[env_id])
        episodes.append(
          QualificationEpisode(
            checkpoint=str(checkpoint),
            scenario=scenario,
            seed=seed,
            env_id=env_id,
            pair=pairs[env_id],
            arm="policy" if arm_index else "baseline",
            length=int(lengths[row]),
            terminations={
              name: int(termination_values[name][row]) for name in term_names
            },
            rewards={name: float(reward_values[name][row]) for name in reward_names},
            metrics={name: float(metric_values[name][row]) for name in metric_values},
          )
        )

        counts[env_id][arm_index] += 1
        if min(counts[env_id]) >= args.episodes_per_env:
          active[env_id] = False
        else:
          current_policy[env_id] = not current_policy[env_id]
          pairs[env_id] = counts[env_id][int(current_policy[env_id])]

      disturbances.clear(done)
      reset_done(env, done)

  return episodes, strata


def write_outputs(
  episodes: list[QualificationEpisode],
  strata: list[StratumRecord],
  report: dict,
  out_dir: Path,
) -> None:
  """Write raw paired episodes to CSV and complete summaries to JSON."""
  out_dir.mkdir(parents=True, exist_ok=True)
  term_names = sorted({name for episode in episodes for name in episode.terminations})
  reward_names = sorted({name for episode in episodes for name in episode.rewards})
  metric_names = sorted({name for episode in episodes for name in episode.metrics})
  with (out_dir / "qualification.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(
      ["checkpoint", "scenario", "seed", "env", "pair", "arm", "length"]
      + [f"termination/{name}" for name in term_names]
      + [f"reward/{name}" for name in reward_names]
      + [f"metric/{name}" for name in metric_names]
    )
    for episode in episodes:
      writer.writerow(
        [
          episode.checkpoint,
          episode.scenario,
          episode.seed,
          episode.env_id,
          episode.pair,
          episode.arm,
          episode.length,
        ]
        + [episode.terminations.get(name, 0) for name in term_names]
        + [episode.rewards.get(name, float("nan")) for name in reward_names]
        + [episode.metrics.get(name, float("nan")) for name in metric_names]
      )
  stratum_term_names = sorted(
    {name for record in strata for name in record.terminations}
  )
  stratum_metric_names = sorted({name for record in strata for name in record.metrics})
  identity = [
    "checkpoint",
    "scenario",
    "seed",
    "env",
    "pair",
    "arm",
    "regime",
    "axis",
    "direction",
    "steps",
    "duration_s",
  ]
  with (out_dir / "qualification_strata.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(
      identity
      + [f"termination/{name}" for name in stratum_term_names]
      + [f"metric/{name}" for name in stratum_metric_names]
    )
    for record in strata:
      writer.writerow(
        [
          record.checkpoint,
          record.scenario,
          record.seed,
          record.env_id,
          record.pair,
          record.arm,
          record.regime,
          record.axis,
          record.direction,
          record.steps,
          record.duration_s,
        ]
        + [record.terminations.get(name, 0) for name in stratum_term_names]
        + [record.metrics.get(name, float("nan")) for name in stratum_metric_names]
      )
  joint_fields = sorted(
    {name for record in strata for joint in record.joints.values() for name in joint}
  )
  with (out_dir / "qualification_joints.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(identity + ["joint"] + joint_fields)
    for record in strata:
      prefix = [
        record.checkpoint,
        record.scenario,
        record.seed,
        record.env_id,
        record.pair,
        record.arm,
        record.regime,
        record.axis,
        record.direction,
        record.steps,
        record.duration_s,
      ]
      for joint_name, values in record.joints.items():
        writer.writerow(
          prefix
          + [joint_name]
          + [values.get(name, float("nan")) for name in joint_fields]
        )
  (out_dir / "qualification.json").write_text(
    json.dumps(report, indent=2, allow_nan=True) + "\n"
  )


def main() -> None:
  """Qualify checkpoint inputs and select only among candidates passing gates."""
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("checkpoints", nargs="+", help="checkpoint paths, dirs, or globs")
  parser.add_argument("--scenario", action="append", choices=SCENARIOS)
  parser.add_argument("--episodes-per-env", type=int, default=2)
  # Clusters are seeds x environments; 8 could not resolve the effect the
  # 2026-08-25 screen measured. docs/evaluation.md#clusters_for_confidence
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--num-workers", type=int, default=6)
  parser.add_argument("--episode-length-s", type=float, default=90.0)
  parser.add_argument("--seed", type=int, action="append")
  parser.add_argument(
    "--achievement-stage", type=int, choices=range(len(ACHIEVEMENT_STAGES))
  )
  parser.add_argument("--control", choices=("position", "torque"), default="position")
  parser.add_argument(
    "--authority-set",
    choices=AUTHORITY_SETS,
    default=None,
  )
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--out-dir", type=Path, default=Path("logs/qualification"))
  args = parser.parse_args()

  args.authority_set = args.authority_set or (
    "ankle" if args.achievement_stage is not None else "uniform"
  )

  checkpoints = resolve_checkpoints(args.checkpoints)
  scenarios = args.scenario or list(SCENARIOS)
  seeds = args.seed or ([42, 43] if args.achievement_stage is not None else [42])
  if args.achievement_stage is not None:
    if len(checkpoints) != 1:
      parser.error("achievement qualification accepts exactly one checkpoint")
    if set(scenarios) != set(REQUIRED_QUALIFICATION_SCENARIOS):
      parser.error("achievement qualification requires every scenario")
    if len(set(seeds)) < MINIMUM_QUALIFICATION_SEEDS:
      parser.error(
        f"achievement qualification requires {MINIMUM_QUALIFICATION_SEEDS} seeds"
      )
  episodes: list[QualificationEpisode] = []
  strata: list[StratumRecord] = []
  summaries: dict[str, dict] = {}
  for checkpoint in checkpoints:
    print(f"[qualify] {checkpoint}", flush=True)
    by_scenario: dict[str, dict] = {}
    by_stratum: dict[str, dict] = {}
    for scenario in scenarios:
      print(f"[qualify]   {scenario}", flush=True)
      result: list[QualificationEpisode] = []
      scenario_strata: list[StratumRecord] = []
      for seed in seeds:
        print(f"[qualify]     seed {seed}", flush=True)
        seed_episodes, seed_strata = run_checkpoint(checkpoint, scenario, seed, args)
        result.extend(seed_episodes)
        scenario_strata.extend(seed_strata)
      episodes.extend(result)
      strata.extend(scenario_strata)
      by_scenario[scenario] = summarize(result)
      by_stratum[scenario] = summarize_strata(scenario_strata)
    gate = promotion(by_scenario)
    summaries[str(checkpoint)] = {
      "scenarios": by_scenario,
      "stratified": by_stratum,
      "promotion": gate,
    }
    print(
      f"[qualify]   {'PASS' if gate['eligible'] else 'FAIL'}: "
      + ("all gates" if gate["eligible"] else "; ".join(gate["reasons"])),
      flush=True,
    )
  eligible = [
    (values["promotion"]["rank"], checkpoint)
    for checkpoint, values in summaries.items()
    if values["promotion"]["eligible"]
  ]
  selected = max(eligible)[1] if eligible else None
  report = {
    "config": {
      "seeds": seeds,
      "scenarios": scenarios,
      "episodes_per_env": args.episodes_per_env,
      "num_envs": args.num_envs,
      "episode_length_s": args.episode_length_s,
      "authority_set": args.authority_set,
      "controller_history": 20,
      "proprio_history": 5,
      "recurrent": False,
      "achievement": (
        {
          "stage": args.achievement_stage,
          "contract": achievement_contract(args.achievement_stage),
          "contract_sha256": achievement_contract_sha256(args.achievement_stage),
        }
        if args.achievement_stage is not None
        else None
      ),
      "strata": {
        "startup": f"episode age <= {DISTURBANCE_WARMUP_S:g} s",
        "recovery": "reward-configured post-disturbance window",
        "sustained": "all remaining exposure",
        "direction_frame": "controller base frame",
      },
    },
    "checkpoints": summaries,
    "selected": selected,
    "selection_rule": "safety and nominal gates, then recovery/hazard/residual",
  }
  write_outputs(episodes, strata, report, args.out_dir)
  print(f"[qualify] selected: {selected or 'none'}")
  print(
    f"[qualify] outputs: {args.out_dir / 'qualification.csv'}, "
    "qualification_strata.csv, qualification_joints.csv, and qualification.json"
  )


if __name__ == "__main__":
  main()
