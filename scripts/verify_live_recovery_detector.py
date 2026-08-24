"""Assert recovery authority through the live simulator and action term."""

from __future__ import annotations

import argparse

import torch
from mjlab.envs import ManagerBasedRlEnv

from mc_mjlab.tasks import mdp
from mc_mjlab.tasks.residual_balance.residual_balance_env_cfg import _make_env_cfg


def main() -> None:
  """Drive nonzero actions and verify inactive zeroing plus detector activation."""
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--num-envs", type=int, default=2)
  parser.add_argument("--num-workers", type=int, default=2)
  parser.add_argument("--steps", type=int, default=900)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  cfg = _make_env_cfg("position", num_envs=args.num_envs, num_workers=args.num_workers)
  cfg.events["push_robot"].params["planar_speed"] = 0.4
  env = ManagerBasedRlEnv(cfg, device=args.device)
  term = mdp._residual_term(env, "mc_rtc_residual")
  action = torch.ones(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )
  max_authority = 0.0
  inactive_peak = 0.0
  authority_sum = 0.0
  env.reset()
  for _ in range(args.steps):
    env.step(action)
    authority = term.last_gate
    inactive = authority == 0.0
    if bool(inactive.any()):
      peak = term.executed_physical_action[inactive].abs().amax()
      inactive_peak = max(inactive_peak, float(peak))
    max_authority = max(max_authority, float(authority.max()))
    authority_sum += float(authority.mean())
  env.close()
  assert inactive_peak == 0.0
  assert max_authority > 0.05
  print(
    f"live detector passed: max={max_authority:.3f}, "
    f"duty={authority_sum / args.steps:.3%}, inactive_peak={inactive_peak:g}"
  )


if __name__ == "__main__":
  main()
