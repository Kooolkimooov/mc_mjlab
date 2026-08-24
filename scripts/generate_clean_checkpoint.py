"""Generate an untrained checkpoint using the repository's current policy contract."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import _make_env_cfg
from mc_mjlab.tasks.residual_balance.residual_balance_ppo_cfg import (
  residual_balance_ppo_cfg,
)
from mc_mjlab.tasks.residual_balance.residual_balance_runner import (
  ResidualBalanceOnPolicyRunner,
)


def main() -> None:
  """Build the current actor and save it with controller provenance."""
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--control", choices=("position", "torque"), default="position")
  parser.add_argument(
    "--authority-set",
    choices=("uniform", "ankle", "sagittal", "hardware"),
    default="uniform",
  )
  parser.add_argument("--num-workers", type=int, default=1)
  parser.add_argument(
    "--controller-history", type=int, choices=(1, 5, 10, 20), default=20
  )
  parser.add_argument("--proprio-history", type=int, choices=(1, 5), default=5)
  parser.add_argument("--recurrent", action="store_true")
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  args.output.parent.mkdir(parents=True, exist_ok=True)
  cfg = _make_env_cfg(
    args.control,
    num_envs=1,
    num_workers=args.num_workers,
    push_velocity=0.0,
    authority_set=args.authority_set,
    controller_history=args.controller_history,
    proprio_history=args.proprio_history,
  )
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg, device=args.device)
  runner = ResidualBalanceOnPolicyRunner(
    RslRlVecEnvWrapper(env),
    asdict(residual_balance_ppo_cfg(recurrent=args.recurrent)),
    device=args.device,
  )
  runner.save(str(args.output))
  env.close()
  print(f"clean checkpoint -> {args.output}")


if __name__ == "__main__":
  main()
