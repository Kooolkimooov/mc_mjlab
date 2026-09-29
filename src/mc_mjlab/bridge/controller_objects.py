"""Free-body object feedback in controller-local coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

#: Position, xyzw orientation, linear velocity, angular velocity, linear acceleration.
OBJECT_STRIDE = 3 + 4 + 3 + 3 + 3

#: MuJoCo's wxyz free-joint quaternion, reordered to the xyzw the layout carries.
_XYZW = [4, 5, 6, 3]


def world_angular_velocity(
  quat_wxyz: torch.Tensor, local: torch.Tensor
) -> torch.Tensor:
  """Rotate a free joint's body-frame angular velocity into world."""
  axis = quat_wxyz[:, 1:4]
  cross = 2 * torch.linalg.cross(axis, local)
  return local + quat_wxyz[:, :1] * cross + torch.linalg.cross(axis, cross)


class ControllerObjects:
  """Stage each controller object's free-body state, sensor channels included."""

  def __init__(self, env: ManagerBasedRlEnv, mapping: dict[str, str]) -> None:
    self.env = env
    self.addresses: list[tuple[int, int]] = []
    for name, entity_name in mapping.items():
      self.addresses.append(self._free_joint_addresses(env, name, entity_name))

  def fill(self, block: torch.Tensor, offset: int) -> None:
    """Sample each object's pose, velocity and acceleration at the dispatch point."""
    data = self.env.sim.data
    origins = self.env.scene.env_origins

    for i, (qpos_adr, dof_adr) in enumerate(self.addresses):
      qpos = data.qpos[:, qpos_adr : qpos_adr + 7]
      qvel = data.qvel[:, dof_adr : dof_adr + 6]
      start = offset + OBJECT_STRIDE * i

      block[:, start : start + 3] = qpos[:, :3] - origins
      block[:, start + 3 : start + 7] = qpos[:, _XYZW]
      block[:, start + 7 : start + 10] = qvel[:, :3]
      block[:, start + 10 : start + 13] = world_angular_velocity(
        qpos[:, 3:7], qvel[:, 3:6]
      )
      # The estimator's own acceleration channel, as mc_mujoco sends qacc.
      block[:, start + 13 : start + 16] = data.qacc[:, dof_adr : dof_adr + 3]

  @staticmethod
  def _free_joint_addresses(
    env: ManagerBasedRlEnv, name: str, entity_name: str
  ) -> tuple[int, int]:
    """``(qpos, dof)`` addresses of a passive entity's single free joint."""
    if not name or entity_name not in env.scene.entities:
      raise ValueError(f"invalid controller object mapping: {name!r}: {entity_name!r}")
    if env.scene[entity_name].cfg.articulation is not None:
      raise ValueError(f"controller object {entity_name!r} must be passive")

    model = env.sim.mj_model
    joints = [
      model.joint(i)
      for i in range(model.njnt)
      if model.joint(i).name.startswith(entity_name + "/")
    ]
    if len(joints) != 1 or joints[0].type[0] != mujoco.mjtJoint.mjJNT_FREE:  # ty: ignore[unresolved-attribute]
      raise ValueError(f"controller object {entity_name!r} must have one free joint")
    return int(joints[0].qposadr[0]), int(joints[0].dofadr[0])
