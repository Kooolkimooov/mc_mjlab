"""ResidualMPC disturbances."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from mc_mjlab import mdp as shared_mdp

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


from mc_mjlab.tasks.residual_mpc.mdp.accessors import (
  PLANAR_KICK_SPEED,
  ROBOT_CFG,
  YAW_KICK_RATE,
  residual_action,
)


def refresh_action_scaling(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | None
) -> None:
  """Sync randomized effort limits and PD gains into the action scale."""
  del env_ids
  residual_action(env).refresh_effort_limits_and_action_scale()


class initial_velocity_kick(shared_mdp.disturbances.push_and_record):
  """One base-velocity kick per episode, withheld until the FSM is walking."""

  scale: float = 1.0
  """Difficulty dial driven by survival_kick_curriculum."""

  def __call__(  # type: ignore[override]
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    warmup_s: float = 0.0,
    asset_cfg: SceneEntityCfg | None = None,
  ) -> None:
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    age = env.episode_length_buf[ids]
    # `last_push_step` is global while `age` is per-episode, so this is "the last
    # kick happened inside the current episode" rather than "ever".
    kicked = (env.common_step_counter - self.last_push_step[ids]) < age
    ids = ids[(age * env.step_dt >= warmup_s) & ~kicked]
    if ids.numel() == 0:
      return
    self._kick(env, ids, asset_cfg)

  def _kick(
    self,
    env: ManagerBasedRlEnv,
    ids: torch.Tensor,
    asset_cfg: SceneEntityCfg | None,
  ) -> None:
    """Sample uniformly inside the paper's norm ball and add it to the base."""
    asset = env.scene[(asset_cfg or ROBOT_CFG).name]
    # Fig. 4 bounds the *norms*, so sampling each axis independently would put
    # 21.6% of draws outside the ball. docs/residual-mpc.md#INITIAL_VELOCITY_RANGE
    vel_w = asset.data.root_link_vel_w[ids]
    delta = torch.zeros_like(vel_w)
    angle = 2.0 * torch.pi * torch.rand(len(ids), device=env.device)
    # sqrt(u) keeps the disc uniform; without it the samples crowd the rim.
    radius = (
      PLANAR_KICK_SPEED
      * self.scale
      * torch.sqrt(torch.rand(len(ids), device=env.device))
    )
    delta[:, 0] = radius * torch.cos(angle)
    delta[:, 1] = radius * torch.sin(angle)
    yaw = YAW_KICK_RATE * self.scale
    delta[:, 5] = torch.empty(len(ids), device=env.device).uniform_(-yaw, yaw)
    # `root_link_vel_w` comes from `cvel`, which MuJoCo does not recompute until
    # the next forward, so a before/after difference would read zero.
    asset.write_root_link_velocity_to_sim(vel_w + delta, env_ids=ids)
    self.last_push_vel[ids] = quat_apply_inverse(
      asset.data.root_link_quat_w[ids], delta[:, :3]
    )
    self.last_push_step[ids] = env.common_step_counter
