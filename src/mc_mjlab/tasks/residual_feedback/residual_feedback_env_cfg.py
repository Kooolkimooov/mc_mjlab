"""ResidualMPC's task with the residual moved onto the controller's own feedback."""

from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg

from mc_mjlab.actions.residual_feedback_action import (
  ResidualFeedbackJointTorqueActionCfg,
)
from mc_mjlab.tasks.residual_mpc import mdp
from mc_mjlab.tasks.residual_mpc.residual_mpc_env_cfg import residual_mpc_env_cfg

#: Encoder offset at a saturated feedback action. docs/residual-feedback.md
FEEDBACK_SCALE = 0.02


def residual_feedback_env_cfg(
  play: bool = False,
  feedback_scale: float = FEEDBACK_SCALE,
  torque_channel: bool = True,
  **kwargs,
) -> ManagerBasedRlEnvCfg:
  """ResidualMPC's env with the action swapped for the residual-feedback one."""
  cfg = residual_mpc_env_cfg(play=play, **kwargs)
  # Only the action term differs, so the two tasks stay comparable.
  base = cfg.actions[mdp.ACTION_NAME]
  carried = {f.name: getattr(base, f.name) for f in fields(base)}
  cfg.actions[mdp.ACTION_NAME] = ResidualFeedbackJointTorqueActionCfg(
    **carried,
    feedback_scale=feedback_scale,
    torque_channel=torque_channel,
  )
  return cfg
