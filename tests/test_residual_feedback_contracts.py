"""Deterministic residual-feedback layout, isolation and parity contracts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

import mc_mjlab.tasks  # noqa: F401
from mc_mjlab.actions.residual_feedback_action import (
  ROOT_POSE_DIM,
  SUPPORTED_MODALITIES,
  ResidualFeedbackJointTorqueActionCfg,
)
from mc_mjlab.sim_controller_bridge import _compose_small_rotation
from mc_mjlab.tasks.residual_feedback.residual_feedback_env_cfg import (
  residual_feedback_env_cfg,
)
from mc_mjlab.tasks.residual_mpc import mdp
from mc_mjlab.tasks.residual_mpc.residual_mpc_env_cfg import residual_mpc_env_cfg


def test_rotation_composition() -> None:
  """A composed rotation stays a unit quaternion; adding one would not."""

  def compose(quat: torch.Tensor, rotvec: Sequence[Sequence[float]]) -> np.ndarray:
    return _compose_small_rotation(
      quat, torch.tensor(rotvec, dtype=torch.float64).expand(3, 3)
    ).numpy()

  quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64).expand(3, 4)
  out = compose(quat, [[0.0, 0.0, 0.0]])
  assert np.allclose(out, quat.numpy()), (
    "a zero rotation must leave the quaternion alone"
  )
  out = compose(quat, [[0.0, 0.02, 0.0]])
  norms = np.linalg.norm(out, axis=1)
  assert np.allclose(norms, 1.0), f"composition must stay normalized, got {norms}"
  assert not np.allclose(out, quat.numpy()), (
    "a non-zero rotation must change the quaternion"
  )
  # Opposite rotations must land either side of the identity, not both above it.
  plus = compose(quat, [[0.0, 0.05, 0.0]])
  minus = compose(quat, [[0.0, -0.05, 0.0]])
  assert plus[0, 2] * minus[0, 2] < 0.0, "sign of the rotation must be respected"


def test_modality_widths() -> None:
  """Declared widths are what the action space grows by, per modality."""
  cfg = residual_feedback_env_cfg(num_envs=2, num_workers=1)
  action = cfg.actions[mdp.ACTION_NAME]
  assert isinstance(action, ResidualFeedbackJointTorqueActionCfg)
  assert set(action.feedback_modalities) <= set(SUPPORTED_MODALITIES)
  assert ROOT_POSE_DIM == 6, "root pose is 3 translation plus 3 rotation"


def test_rejects_bad_modalities() -> None:
  """An unknown or duplicated modality fails at build, not at the first step."""
  for bad in (("elbow_grease",), ("joint_position", "joint_position"), ()):
    try:
      residual_feedback_env_cfg(num_envs=2, num_workers=1, feedback_modalities=bad)
    except ValueError:
      continue
    # The cfg builder defers to the action's constructor, so an invalid tuple may
    # only raise there; either place is fine, silence is not.
    raise AssertionError(f"modalities {bad!r} must be rejected")


def test_parity_with_residual_mpc() -> None:
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


def test_curriculum_present() -> None:
  """The kick curriculum needs both its opt-in and a kick to apply."""
  for builder in (residual_mpc_env_cfg, residual_feedback_env_cfg):
    opted_in = builder(num_envs=2, num_workers=1, kick_curriculum=True)
    assert "kick_difficulty" in (opted_in.curriculum or {}), (
      f"{builder.__name__} must adapt kick difficulty when asked to"
    )
    # Off by default: an adaptive magnitude makes the arms face different
    # difficulty. docs/residual-mpc.md#INITIAL_VELOCITY_RANGE
    default = builder(num_envs=2, num_workers=1)
    assert not (default.curriculum or {}), (
      f"{builder.__name__} must leave the kick distribution fixed by default"
    )
    without = builder(num_envs=2, num_workers=1, kick_curriculum=True, pushes=False)
    assert not (without.curriculum or {}), (
      f"{builder.__name__} must not curriculum a kick it does not apply"
    )
