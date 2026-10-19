"""The measured-ZMP memo must serve one step phase and never the next."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import mujoco
import torch
from mjlab.envs import ManagerBasedRlEnv

from mc_mjlab import mdp

FEET = ("LeftFootForceSensor", "RightFootForceSensor")

FEET_XML = """
<mujoco>
  <worldbody>
    <body name="root" pos="0 0 0.8">
      <freejoint/>
      <geom size="0.1" mass="1"/>
      <site name="LeftFootForceSensor" pos="0 0.1 -0.8"/>
      <site name="RightFootForceSensor" pos="0 -0.1 -0.8"/>
    </body>
  </worldbody>
  <sensor>
    <force name="LeftFootForceSensor_fsensor" site="LeftFootForceSensor"/>
    <torque name="LeftFootForceSensor_tsensor" site="LeftFootForceSensor"/>
    <force name="RightFootForceSensor_fsensor" site="RightFootForceSensor"/>
    <torque name="RightFootForceSensor_tsensor" site="RightFootForceSensor"/>
  </sensor>
</mujoco>
"""


class FeetEnv:
  """One environment standing on two loaded feet, CoM 0.8 m above the floor."""

  def __init__(self) -> None:
    self.mj_model = mujoco.MjModel.from_xml_string(FEET_XML)
    self.num_envs = 1
    self.device = "cpu"
    self.common_step_counter = 7
    self.scene = {"robot": SimpleNamespace(indexing=SimpleNamespace(root_body_id=1))}
    nsite = self.mj_model.nsite
    self.data = SimpleNamespace(
      site_xmat=torch.eye(3).reshape(1, 1, 9).repeat(1, nsite, 1),
      site_xpos=torch.zeros(1, nsite, 3),
      sensordata=torch.zeros(1, self.mj_model.nsensordata),
      subtree_com=torch.tensor([[[0.0, 0.0, 0.8], [0.0, 0.0, 0.8]]]),
    )
    self.sim = SimpleNamespace(mj_model=self.mj_model, data=self.data)
    for foot in FEET:
      adr = self.mj_model.sensor(f"{foot}_fsensor").adr[0]
      self.data.sensordata[0, adr + 2] = -500.0

  def place_feet_x(self, x: float) -> None:
    """Move both loaded soles to ``x`` relative to the CoM, as a reset would."""
    for foot in FEET:
      site = self.mj_model.site(foot).id
      self.data.site_xpos[0, site] = torch.tensor([x, 0.0, 0.0])


def test_the_critic_offset_is_the_post_reset_state() -> None:
  """A reward-phase read at the same counter must not reach the observation."""
  env = FeetEnv()
  sensors = mdp.sensors.zmp_sensors(cast(ManagerBasedRlEnv, env), FEET, "robot")

  env.place_feet_x(-0.2)
  terminal, _ = sensors.measured_offset(cast(ManagerBasedRlEnv, env))
  assert torch.allclose(terminal, torch.tensor([[-0.2, 0.0]]))

  env.place_feet_x(0.0)
  observed = mdp.sensors.measured_zmp_offset(cast(ManagerBasedRlEnv, env), FEET)
  assert torch.allclose(observed, torch.zeros(1, 2), atol=1e-6)
