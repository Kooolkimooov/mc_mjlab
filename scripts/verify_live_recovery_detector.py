"""Assert recovery authority through the live simulator and action term."""

from __future__ import annotations

import argparse

import torch
from mjlab.envs import ManagerBasedRlEnv

from mc_mjlab import mdp
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
  parser.add_argument("--disturbance", choices=("finite", "velocity"), default="finite")
  parser.add_argument("--randomization-stage", type=int, choices=(0, 1, 2), default=0)
  args = parser.parse_args()
  cfg = _make_env_cfg(
    "position",
    num_envs=args.num_envs,
    num_workers=args.num_workers,
    disturbance=args.disturbance,
    randomization_stage=args.randomization_stage,
  )
  if args.disturbance == "velocity":
    cfg.events["push_robot"].params["planar_speed"] = 0.4
  env = ManagerBasedRlEnv(cfg, device=args.device)
  term = mdp.sensors._residual_term(env, "mc_rtc_residual")
  action = torch.ones(
    env.num_envs, env.action_manager.total_action_dim, device=env.device
  )
  max_authority = 0.0
  inactive_peak = 0.0
  authority_sum = 0.0
  impulses = 0
  impulse_error = 0.0
  push_term = mdp.disturbances._push_term(env, "push_robot")
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
    if isinstance(push_term, mdp.disturbances.finite_impulse_curriculum):
      fired = push_term.last_push_step == env.common_step_counter
      ids = fired.nonzero(as_tuple=False).flatten()
      if ids.numel():
        body_ids = push_term.asset.indexing.body_ids
        mass = env.sim.model.body_mass[ids][:, body_ids].sum(dim=1)
        delivered = (
          torch.linalg.vector_norm(push_term.force[ids, 0], dim=1)
          * push_term.remaining[ids]
          * env.step_dt
          / mass
        )
        requested = torch.linalg.vector_norm(push_term.last_push_vel[ids], dim=1)
        impulse_error = max(impulse_error, float((delivered - requested).abs().max()))
        duration = push_term.remaining[ids] * env.step_dt
        assert bool(((duration >= 0.08) & (duration <= 0.20)).all())
        moment_arm = torch.linalg.vector_norm(push_term.torque[ids, 0], dim=1) / (
          torch.linalg.vector_norm(push_term.force[ids, 0], dim=1) + 1.0e-9
        )
        assert bool((moment_arm <= 0.25 + 1.0e-6).all())
        impulses += len(ids)
  env.close()
  assert inactive_peak == 0.0
  assert max_authority > 0.05
  if args.disturbance == "finite":
    assert impulses > 0 and impulse_error < 1.0e-5
  print(
    f"live detector passed: max={max_authority:.3f}, "
    f"duty={authority_sum / args.steps:.3%}, inactive_peak={inactive_peak:g}, "
    f"impulses={impulses}, impulse_error={impulse_error:.2g}"
  )


if __name__ == "__main__":
  main()
