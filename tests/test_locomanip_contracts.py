"""Wiring contracts the trainable locomanip task must not lose."""

from __future__ import annotations

from typing import cast

from mjlab.envs import ManagerBasedRlEnvCfg

import mc_mjlab.tasks  # noqa: F401
from mc_mjlab.actions.mc_rtc_residual_joint_position_actions import (
  McRtcResidualJointPositionActionCfg,
)
from mc_mjlab.tasks.locomanip import DEMO_TASK_ID, RESIDUAL_TASK_ID
from mc_mjlab.tasks.locomanip.locomanip_residual_env_cfg import (
  DECIMATION,
  FRAMESKIP,
  RESIDUAL_SCALE,
  make_locomanip_residual_env_cfg,
)
from mc_mjlab.tasks.locomanip.mdp import accessors


def _action(cfg: ManagerBasedRlEnvCfg) -> McRtcResidualJointPositionActionCfg:
  """Return the residual action cfg under its own type."""
  return cast(McRtcResidualJointPositionActionCfg, cfg.actions[accessors.ACTION_NAME])


def test_ids_are_distinct() -> None:
  """Verify the demo id is a prefix of the trainable one, so matching must be exact."""
  assert RESIDUAL_TASK_ID != DEMO_TASK_ID
  assert RESIDUAL_TASK_ID.startswith(DEMO_TASK_ID)


def test_action_name_matches_the_provenance_key() -> None:
  """Verify the runner's hardcoded action lookup resolves before the first step."""
  cfg = make_locomanip_residual_env_cfg()
  assert accessors.ACTION_NAME in cfg.actions


def test_control_rates_divide() -> None:
  """Verify decimation is a whole number of controller periods."""
  cfg = make_locomanip_residual_env_cfg()
  action = _action(cfg)
  assert cfg.decimation == DECIMATION
  assert action.frameskip == FRAMESKIP
  assert cfg.decimation % action.frameskip == 0


def test_residual_authority_covers_every_actuator() -> None:
  """Verify no actuator can fall through to scale 1.0 with no clip."""
  action = _action(make_locomanip_residual_env_cfg())
  assert action.scale == RESIDUAL_SCALE
  assert action.clip == {".*": (-RESIDUAL_SCALE, RESIDUAL_SCALE)}
  assert action.residual_actuator_names


def test_declared_datastore_outputs_cover_the_terms() -> None:
  """Verify every callback the manager terms read is collected by the action."""
  action = _action(make_locomanip_residual_env_cfg())
  assert set(action.datastore_vectors_outputs) == {
    accessors.OBJECT_REFERENCE_POSITION,
    accessors.OBJECT_REFERENCE_RPY,
  }
  assert set(action.datastore_scalar_outputs) == {
    accessors.LEFT_PHASE,
    accessors.RIGHT_PHASE,
    accessors.COMPLETE,
  }
  assert action.controller_objects == {"obj": accessors.OBJECT_ENTITY}
  assert action.required_controller == accessors.REQUIRED_CONTROLLER


def test_worker_failure_truncates_instead_of_penalising() -> None:
  """Verify infrastructure noise bootstraps rather than paying the fall penalty."""
  terminations = make_locomanip_residual_env_cfg().terminations
  assert terminations["controller_worker_failed"].time_out
  assert not terminations["controller_failed"].time_out


def test_scene_resets_joints_first() -> None:
  """Verify the only term that resets joints leads the reset events."""
  events = make_locomanip_residual_env_cfg().events
  assert next(iter(events)) == "reset_scene_to_default"


def test_cart_variation_hooks_are_opt_in() -> None:
  """Verify the pose and payload hooks add events only when asked for."""
  assert "reset_cart" not in make_locomanip_residual_env_cfg().events
  varied = make_locomanip_residual_env_cfg(
    cart_pose_range={"x": (-0.05, 0.05)}, cart_mass_scale=(1.0, 4.0)
  )
  assert "reset_cart" in varied.events
  assert "cart_payload" in varied.events


def test_sparse_jacobian_is_kept() -> None:
  """Verify the cart's extra velocities stay under MuJoCo Warp's dense limit."""
  assert make_locomanip_residual_env_cfg().sim.mujoco.jacobian == "sparse"
