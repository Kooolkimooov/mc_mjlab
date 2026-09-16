"""Audit every live reward for policy-zero and optional checkpoint rollouts."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from evaluation.reward_audit import (
  RewardAuditRecorder,
  RewardAuditShapeError,
)
from evaluation.rollout import managed_env
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlVecEnvWrapper

from mc_mjlab.rl.effective_training_manifest import (
  build_effective_training_manifest,
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

OUTPUT_DIR = Path("logs/reward_audits")
CONDITIONAL_METRICS = ("zmp_grounded", "recovery_active")
TERM_DENOMINATORS = {
  "dcm_stability": "zmp_grounded",
  "foot_slip": "zmp_grounded",
  "recovery_dcm": "recovery_active",
}


def _parse_args() -> argparse.Namespace:
  """Parse rollout, policy-interface, and output settings."""
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--steps", type=int, default=1000)
  parser.add_argument("--warmup-steps", type=int, default=50)
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--num-workers", type=int, default=6)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--control", choices=("position", "torque"), default="position")
  parser.add_argument(
    "--disturbance", choices=("finite", "velocity", "none"), default="finite"
  )
  parser.add_argument("--push-velocity", type=float, default=0.4)
  parser.add_argument("--randomization-stage", type=int, choices=(0, 1), default=0)
  parser.add_argument(
    "--authority-set",
    choices=AUTHORITY_SETS,
    default="uniform",
  )
  parser.add_argument("--disable-observation-corruption", action="store_true")
  parser.add_argument(
    "--drop-obs",
    default="",
    help="comma-separated observation terms appended after an older checkpoint",
  )
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  if args.steps <= 0 or args.warmup_steps < 0:
    parser.error("--steps must be positive and --warmup-steps non-negative")
  if args.num_envs <= 0:
    parser.error("--num-envs must be positive")
  if args.checkpoint is not None and args.num_envs < 2:
    parser.error("a checkpoint comparison needs at least two environments")
  return args


def _output_path(args: argparse.Namespace) -> Path:
  """Choose a stable report path from the checkpoint identity."""
  if args.output is not None:
    return args.output
  if args.checkpoint is None:
    stem = f"policy_zero_{args.control}"
  else:
    checkpoint = args.checkpoint.expanduser()
    stem = f"{checkpoint.parent.name}_{checkpoint.stem}"
  return OUTPUT_DIR / f"{stem}.json"


def _make_cfg(args: argparse.Namespace) -> ManagerBasedRlEnvCfg:
  """Build the requested live environment without changing reward semantics."""
  cfg = make_residual_balance_env_cfg(
    control=args.control,
    num_envs=args.num_envs,
    num_workers=args.num_workers,
    push_velocity=args.push_velocity,
    console_output="none",
    disturbance=args.disturbance,
    authority_set=args.authority_set,
    randomization_stage=args.randomization_stage,
  )
  cfg.seed = args.seed
  if args.disable_observation_corruption:
    for group in cfg.observations.values():
      group.enable_corruption = False
  for name in (value.strip() for value in args.drop_obs.split(",") if value.strip()):
    found = False
    for group in cfg.observations.values():
      found = group.terms.pop(name, None) is not None or found
    if not found:
      raise KeyError(f"observation term {name!r} was not configured")
  return cfg


def _arms(args: argparse.Namespace, device: str) -> dict[str, torch.Tensor]:
  """Assign disjoint environments to policy-zero and checkpoint arms."""
  if args.checkpoint is None:
    return {"policy_zero": torch.arange(args.num_envs, device=device)}
  split = args.num_envs // 2
  return {
    "policy_zero": torch.arange(split, device=device),
    "checkpoint": torch.arange(split, args.num_envs, device=device),
  }


def _step(
  env: ManagerBasedRlEnv,
  wrapped: RslRlVecEnvWrapper,
  policy: Any,
  policy_ids: torch.Tensor | None,
  action: torch.Tensor,
) -> None:
  """Advance both rollout arms once and reset recurrent policy state."""
  action.zero_()
  if policy is not None and policy_ids is not None:
    with torch.inference_mode():
      proposed = policy(wrapped.get_observations())
    action[policy_ids] = proposed[policy_ids]
  _, _, terminated, time_outs, _ = env.step(action)
  if policy is not None:
    policy.reset(terminated | time_outs)


def _capture_conditionals(env: ManagerBasedRlEnv, audit: RewardAuditRecorder) -> None:
  """Capture grounded and recovery denominators computed on the audited state."""
  manager = env.metrics_manager
  step_values = getattr(manager, "_step_values", None)
  if step_values is None:
    return
  names = manager.active_terms
  for name in CONDITIONAL_METRICS:
    if name not in names:
      continue
    index = names.index(name)
    audit.capture_denominator(name, step_values[:, index])


def _format_weight(summary: dict[str, Any]) -> str:
  """Format a fixed or changing live weight for the console table."""
  if summary["min"] == summary["max"]:
    return f"{summary['last']:.3g}"
  return f"{summary['min']:.3g}..{summary['max']:.3g}"


def _print_report(report: dict[str, Any]) -> None:
  """Print the compact view while JSON retains complete distributions."""
  for arm, values in report["audit"]["arms"].items():
    print(f"\n[reward-audit] {arm} envs={values['env_ids']}")
    print(
      f"  {'term':<28} {'weight':>11} {'raw mean':>11} {'rate/s':>11} "
      f"{'nonzero':>9} {'q50':>10} {'q90':>10} {'q99':>10} {'integral':>11}"
    )
    for name, term in values["terms"].items():
      raw = term["raw"]
      rate = term["weighted_rate_per_second"]
      print(
        f"  {name:<28} {_format_weight(term['effective_weight']):>11} "
        f"{raw['mean']:11.4g} {rate['mean']:11.4g} "
        f"{raw['nonzero_fraction']:9.1%} {raw['q50']:10.4g} "
        f"{raw['q90']:10.4g} {raw['q99']:10.4g} "
        f"{term['integrated_contribution_per_env']:11.4g}"
      )
    for name, denominator in values["conditional_denominators"].items():
      print(
        f"  denominator/{name}: mean={denominator['mean']:.1%}, "
        f"n={denominator['sample_count']}"
      )


def _json_safe(value: Any) -> Any:
  """Replace non-finite report values with JSON nulls recursively."""
  if isinstance(value, float) and not math.isfinite(value):
    return None
  if isinstance(value, dict):
    return {key: _json_safe(item) for key, item in value.items()}
  if isinstance(value, list):
    return [_json_safe(item) for item in value]
  return value


def main() -> None:
  """Run the live reward audit and fail on shape or non-finite violations."""
  args = _parse_args()
  output = _output_path(args)
  output.parent.mkdir(parents=True, exist_ok=True)

  torch.manual_seed(args.seed)
  cfg = _make_cfg(args)
  train_cfg = asdict(residual_balance_ppo_cfg())
  audit: RewardAuditRecorder | None = None
  failure: str | None = None

  with managed_env(cfg, args.device) as env:
    try:
      wrapped = RslRlVecEnvWrapper(env)
      policy = None
      if args.checkpoint is not None:
        runner = ResidualBalanceOnPolicyRunner(wrapped, train_cfg, device=args.device)
        runner.load(
          str(args.checkpoint.expanduser()),
          load_cfg={"actor": True},
          strict=True,
          map_location=args.device,
        )
        policy = runner.get_inference_policy(device=args.device)

      manifest = build_effective_training_manifest(env, train_cfg)
      arms = _arms(args, str(env.device))
      policy_ids = arms.get("checkpoint")
      action = torch.zeros(
        env.num_envs, env.action_manager.total_action_dim, device=env.device
      )

      env.reset()
      print(
        f"[reward-audit] warmup={args.warmup_steps}, sample={args.steps}, "
        f"envs={args.num_envs}, disturbance={args.disturbance}",
        flush=True,
      )
      for _ in range(args.warmup_steps):
        _step(env, wrapped, policy, policy_ids, action)

      start_step = int(env.common_step_counter)
      audit = RewardAuditRecorder(env.reward_manager, arms, env.step_dt)
      try:
        with audit:
          for step in range(args.steps):
            _step(env, wrapped, policy, policy_ids, action)
            _capture_conditionals(env, audit)
            if (step + 1) % 100 == 0 or step + 1 == args.steps:
              print(f"[reward-audit] sampled {step + 1}/{args.steps}", flush=True)
      except RewardAuditShapeError as error:
        failure = str(error)

      audit_report = audit.report()
      issues = audit.issues()
      if failure is not None:
        issues.insert(0, failure)

      report = {
        "status": "failed" if issues else "passed",
        "issues": issues,
        "config": {
          "checkpoint": (
            str(args.checkpoint.expanduser().resolve())
            if args.checkpoint is not None
            else None
          ),
          "control": args.control,
          "disturbance": args.disturbance,
          "push_velocity": args.push_velocity,
          "randomization_stage": args.randomization_stage,
          "seed": args.seed,
          "warmup_steps": args.warmup_steps,
          "sample_steps": args.steps,
          "common_step_start": start_step,
          "common_step_end": int(env.common_step_counter),
          "authority_set": args.authority_set,
          "controller_history": 20,
          "proprio_history": 5,
          "recurrent": False,
          "walking_reference": False,
          "observation_corruption": not args.disable_observation_corruption,
        },
        "effective_manifest_sha256": manifest["record_sha256"],
        "effective_reward_contract": manifest["record"]["managers"]["reward"],
        "term_denominators": TERM_DENOMINATORS,
        "audit": audit_report,
      }
    finally:
      if audit is not None:
        audit.restore()

  output.write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n")
  _print_report(report)
  print(f"\n[reward-audit] {report['status']} -> {output}")
  if report["issues"]:
    raise SystemExit("; ".join(report["issues"]))


if __name__ == "__main__":
  main()
