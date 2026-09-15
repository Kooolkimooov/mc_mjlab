"""Free-body object feedback in controller-local coordinates."""

import mujoco
import torch


def object_state(qpos, qvel, origins):
  """Pack xyzw pose and world velocities from MuJoCo free-joint state."""
  quat = qpos[:, 3:7]
  angular = qvel[:, 3:6]
  xyz = quat[:, 1:4]
  cross = 2 * torch.linalg.cross(xyz, angular)
  world_angular = angular + quat[:, :1] * cross + torch.linalg.cross(xyz, cross)
  return torch.cat(
    (qpos[:, :3] - origins, quat[:, [1, 2, 3, 0]], qvel[:, :3], world_angular),
    dim=1,
  )


class ControllerObjects:
  """Validate named free-body entities and stage their current joint state."""

  def __init__(self, env, mapping):
    self.env = env
    self.addresses = []
    model = env.sim.mj_model
    for name, entity_name in mapping.items():
      if not name or entity_name not in env.scene.entities:
        raise ValueError(
          f"invalid controller object mapping: {name!r}: {entity_name!r}"
        )
      entity = env.scene[entity_name]
      joints = [
        model.joint(i)
        for i in range(model.njnt)
        if model.joint(i).name.startswith(entity_name + "/")
      ]
      if (
        len(joints) != 1 or joints[0].type[0] != mujoco.mjtJoint.mjJNT_FREE  # ty: ignore[unresolved-attribute]
      ):
        raise ValueError(f"controller object {entity_name!r} must have one free joint")
      if entity.cfg.articulation is not None:
        raise ValueError(f"controller object {entity_name!r} must be passive")
      self.addresses.append((int(joints[0].qposadr[0]), int(joints[0].dofadr[0])))

  def fill(self, block, offset):
    """Sample fresh qpos/qvel at the robot's dispatch point."""
    data = self.env.sim.data
    for i, (qa, da) in enumerate(self.addresses):
      start = offset + 13 * i
      block[:, start : start + 13] = object_state(
        data.qpos[:, qa : qa + 7],
        data.qvel[:, da : da + 6],
        self.env.scene.env_origins,
      )
