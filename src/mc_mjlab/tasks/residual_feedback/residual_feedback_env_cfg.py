"""ResidualMPC's task with the residual moved onto the controller's own feedback."""

from dataclasses import fields

from mjlab.envs import ManagerBasedRlEnvCfg

from mc_mjlab.actions.residual_feedback_action import (
  ResidualFeedbackJointTorqueActionCfg,
)
from mc_mjlab.tasks.residual_mpc import mdp
from mc_mjlab.tasks.residual_mpc.residual_mpc_env_cfg import residual_mpc_env_cfg

#: Feedback spaces and their saturated offsets. docs/residual-feedback.md
FEEDBACK_MODALITIES = ("joint_position", "root_pose")
FEEDBACK_SCALE = 0.02
ROOT_TRANSLATION_SCALE = 0.0025
ROOT_ROTATION_SCALE = 0.005


def residual_feedback_env_cfg(
  play: bool = False,
  feedback_modalities: tuple[str, ...] = FEEDBACK_MODALITIES,
  feedback_scale: float = FEEDBACK_SCALE,
  root_translation_scale: float = ROOT_TRANSLATION_SCALE,
  root_rotation_scale: float = ROOT_ROTATION_SCALE,
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
    feedback_modalities=tuple(feedback_modalities),
    feedback_scale=feedback_scale,
    root_translation_scale=root_translation_scale,
    root_rotation_scale=root_rotation_scale,
    torque_channel=torque_channel,
  )
  return cfg
