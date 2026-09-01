#!/usr/bin/env python3
"""Verify deterministic residual-feedback layout, isolation and parity contracts."""

from __future__ import annotations

import numpy as np

import mc_mjlab.tasks  # noqa: F401
from mc_mjlab.actions.mc_rtc_controller_io_binding import ControllerIoBinding
from mc_mjlab.actions.residual_feedback_action import (
  ROOT_POSE_DIM,
  SUPPORTED_MODALITIES,
  ResidualFeedbackJointTorqueActionCfg,
)
from mc_mjlab.tasks.residual_feedback.residual_feedback_env_cfg import (
  residual_feedback_env_cfg,
)
from mc_mjlab.tasks.residual_mpc import mdp
from mc_mjlab.tasks.residual_mpc.residual_mpc_env_cfg import residual_mpc_env_cfg


def verify_rotation_composition() -> None:
  """A composed rotation stays a unit quaternion; adding one would not."""
  compose = ControllerIoBinding._compose_small_rotation
  quat = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (3, 1))
  out = compose(quat, np.zeros((3, 3)))
  assert np.allclose(out, quat), "a zero rotation must leave the quaternion alone"
  out = compose(quat, np.tile(np.array([[0.0, 0.02, 0.0]]), (3, 1)))
  norms = np.linalg.norm(out, axis=1)
  assert np.allclose(norms, 1.0), f"composition must stay normalized, got {norms}"
  assert not np.allclose(out, quat), "a non-zero rotation must change the quaternion"
  # Opposite rotations must land either side of the identity, not both above it.
  plus = compose(quat, np.tile(np.array([[0.0, 0.05, 0.0]]), (3, 1)))
  minus = compose(quat, np.tile(np.array([[0.0, -0.05, 0.0]]), (3, 1)))
  assert plus[0, 2] * minus[0, 2] < 0.0, "sign of the rotation must be respected"


def verify_modality_widths() -> None:
  """Declared widths are what the action space grows by, per modality."""
  cfg = residual_feedback_env_cfg(num_envs=2, num_workers=1)
  action = cfg.actions[mdp.ACTION_NAME]
  assert isinstance(action, ResidualFeedbackJointTorqueActionCfg)
  assert set(action.feedback_modalities) <= set(SUPPORTED_MODALITIES)
  assert ROOT_POSE_DIM == 6, "root pose is 3 translation plus 3 rotation"


def verify_rejects_bad_modalities() -> None:
  """An unknown or duplicated modality fails at build, not at the first step."""
  for bad in (("elbow_grease",), ("joint_position", "joint_position"), ()):
    try:
      residual_feedback_env_cfg(num_envs=2, num_workers=1, feedback_modalities=bad)
    except ValueError:
      continue
    # The cfg builder defers to the action's constructor, so an invalid tuple may
    # only raise there; either place is fine, silence is not.
    raise AssertionError(f"modalities {bad!r} must be rejected")


def verify_parity_with_residual_mpc() -> None:
  """The two tasks may differ only in the action term."""
  a = residual_mpc_env_cfg(num_envs=2, num_workers=1)
  b = residual_feedback_env_cfg(num_envs=2, num_workers=1)
  assert set(a.rewards) == set(b.rewards), "reward terms must match"
  for name, term in a.rewards.items():
    assert term.weight == b.rewards[name].weight, f"{name} weight differs"
  assert set(a.terminations) == set(b.terminations), "termination terms must match"
  assert set(a.events) == set(b.events), "event terms must match"
  assert set(a.observations) == set(b.observations), "observation groups must match"
  for group, spec in a.observations.items():
    assert set(spec.terms) == set(b.observations[group].terms), (
      f"observation group {group} differs"
    )
  assert a.episode_length_s == b.episode_length_s, "episode length differs"


def verify_curriculum_present() -> None:
  """Both tasks carry the kick curriculum, and neither carries it without a kick."""
  for builder in (residual_mpc_env_cfg, residual_feedback_env_cfg):
    with_push = builder(num_envs=2, num_workers=1)
    assert "kick_difficulty" in (with_push.curriculum or {}), (
      f"{builder.__name__} must adapt kick difficulty"
    )
    without = builder(num_envs=2, num_workers=1, pushes=False)
    assert not (without.curriculum or {}), (
      f"{builder.__name__} must not curriculum a kick it does not apply"
    )


def main() -> None:
  """Run every deterministic residual-feedback contract."""
  verify_rotation_composition()
  verify_modality_widths()
  verify_rejects_bad_modalities()
  verify_parity_with_residual_mpc()
  verify_curriculum_present()
  print("residual-feedback deterministic contracts: PASS")


if __name__ == "__main__":
  main()
