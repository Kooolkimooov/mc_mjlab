#!/usr/bin/env python3
"""Run the paired lambda=0/lambda=0.1 ResidualMPC controller smoke."""

from __future__ import annotations

import argparse

import torch
from mjlab.envs import ManagerBasedRlEnv

from mc_mjlab.actions.residual_mpc_joint_torque_action import (
  ResidualMpcJointTorqueAction,
)
from mc_mjlab.tasks.residual_mpc import mdp
from mc_mjlab.tasks.residual_mpc.residual_mpc_env_cfg import residual_mpc_env_cfg


def main() -> None:
  """Run the deterministic live controller/action assertions."""
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument("--num-workers", type=int, default=2)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  cfg = residual_mpc_env_cfg(
    num_envs=2,
    num_workers=args.num_workers,
    randomization=False,
    pushes=False,
    fixed_twist=(0.1, 0.0, 0.0),
  )
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg, device=args.device)
  term = env.action_manager.get_term(mdp.ACTION_NAME)
  if not isinstance(term, ResidualMpcJointTorqueAction):
    raise TypeError(f"unexpected action term {type(term).__name__}")
  action = torch.zeros(2, env.action_manager.total_action_dim, device=env.device)
  try:
    env.reset()
    term.set_blend_factor(torch.tensor([0], device=env.device), 0.0)
    term.set_blend_factor(torch.tensor([1], device=env.device), 0.1)
    start = env.scene["robot"].data.root_link_pos_w.clone()
    peak_reference_velocity = 0.0
    for _ in range(args.steps):
      _, _, terminated, truncated, _ = env.step(action)
      assert not bool(term.controller_worker_failed.any())
      assert not bool(truncated.any())
      assert bool(torch.isfinite(term.final_effort).all())
      assert bool(torch.isfinite(term.nominal_torque).all())
      assert bool(torch.isfinite(term.residual_torque).all())
      assert bool(
        torch.isfinite(term.datastore_scalar_output(mdp.QP_OBJECTIVE_CALLBACK)).all()
      )
      support = term.datastore_scalar_output(mdp.SUPPORT_FOOT_CALLBACK)
      assert bool(((support == 0.0) | (support == 1.0)).all())
      phase = mdp.controller_contact_phases(env)
      assert bool(((phase >= 0.0) & (phase <= 1.0)).all())
      assert float(term.blended_residual_torque[0].abs().max()) == 0.0
      assert bool((term.final_effort.abs() <= term.effort_limit + 1.0e-5).all())
      peak_reference_velocity = max(
        peak_reference_velocity,
        float(term.controller_reference("alpha").abs().amax()),
      )
      if bool(terminated.any()):
        raise AssertionError("controller cohort terminated during live smoke")
    displacement = torch.linalg.vector_norm(
      env.scene["robot"].data.root_link_pos_w[:, :2] - start[:, :2], dim=1
    )
    assert bool((displacement > 0.05).all()) or peak_reference_velocity > 0.05
    assert float(term.blended_residual_torque[1].abs().max()) > 0.0
    print(
      "ResidualMPC live smoke: PASS "
      f"displacement={displacement.tolist()} "
      f"peak_reference_velocity={peak_reference_velocity:.3f}"
    )
  finally:
    term.close()
    env.close()


if __name__ == "__main__":
  main()
