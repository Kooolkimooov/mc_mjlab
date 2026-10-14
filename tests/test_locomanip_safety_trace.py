"""Native-input force delivery and terminal extrema survive episode recycling."""

from __future__ import annotations

from types import SimpleNamespace as NS
from typing import Any, cast

import numpy as np
import pytest
import torch
from evaluation.locomanip_trace import DispatchTrace, SafetyTrace
from mjlab.envs import ManagerBasedRlEnv

from mc_mjlab.tasks.locomanip.mdp import accessors, observations


def test_dispatch_trace_reads_actual_input_offsets(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  raw = torch.arange(12, dtype=torch.float32).reshape(2, 6)
  offsets = torch.tensor([[5.0, -25.0, 50.0, 0.0, 0.0, 0.0], [0.0] * 6])
  rows = np.zeros((2, 12))

  bridge = NS(
    fill_controller_input=None,
    layout=NS(
      input=NS(
        force_sensors_offset=lambda: 0,
        force_sensors=["LeftHandForceSensor", "RightHandForceSensor"],
      )
    ),
  )

  # The second hand occupies the next six native columns.
  def fill_with_stride(block: np.ndarray) -> None:
    block[:, :3] = raw[:, :3].numpy() + offsets[:, :3].numpy()
    block[:, 6:9] = raw[:, 3:].numpy() + offsets[:, 3:].numpy()

  bridge.fill_controller_input = fill_with_stride
  term = NS(_bridge=bridge, executed_physical_action=offsets)
  monkeypatch.setattr(accessors, "residual_action", lambda env: term)
  sensors = {
    name: NS(adr=[index])
    for index, name in enumerate(
      ("robot/LeftHandForceSensor_fsensor", "robot/RightHandForceSensor_fsensor")
    )
  }
  sensors["robot/RightHandForceSensor_fsensor"] = NS(adr=[3])
  env = cast(
    ManagerBasedRlEnv,
    NS(
      num_envs=2,
      sim=NS(mj_model=NS(sensor=sensors.__getitem__), data=NS(sensordata=raw)),
    ),
  )
  trace = DispatchTrace(env)
  bridge.fill_controller_input(rows)
  np.testing.assert_allclose(trace.last, offsets.numpy())
  np.testing.assert_allclose(trace.peak[0, :3], [5, 25, 50])
  np.testing.assert_allclose(trace.error, 0)
  assert trace.active.tolist() == [1, 0]
  trace.close()
  assert bridge.fill_controller_input is fill_with_stride


def test_max_lag_and_failure_snapshot_precede_reset(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  def entity(**data: Any) -> Any:
    return NS(data=NS(**data))

  robot = entity(
    root_link_pos_w=torch.tensor([[0.0, 0.0, 0.79]]),
    root_link_lin_vel_w=torch.zeros(1, 3),
    root_link_ang_vel_b=torch.zeros(1, 3),
    projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]),
  )
  cart = entity(
    root_link_pos_w=torch.tensor([[0.9, 0, 0.0]]), root_link_lin_vel_w=torch.zeros(1, 3)
  )

  class Scene(dict):
    """Expose entity lookup and environment origins to the trace."""

    env_origins = torch.zeros(1, 3)

  scene = Scene(robot=robot, cart=cart)
  term = NS(
    raw_action=torch.zeros(1, 6),
    cfg=NS(frameskip=2),
    datastore_vector_output=lambda name: torch.tensor([[1.0, 0, 0]]),
    controller_reference=lambda name: torch.zeros(1, 2),
  )
  env = cast(
    ManagerBasedRlEnv,
    NS(
      num_envs=1,
      device="cpu",
      step_dt=0.02,
      scene=scene,
      episode_length_buf=torch.tensor([1]),
      cfg=NS(sim=NS(mujoco=NS(timestep=0.001))),
      termination_manager=NS(
        active_terms=["fell_over"], get_term=lambda n: torch.tensor([True])
      ),
    ),
  )
  monkeypatch.setattr(accessors, "residual_action", lambda env: term)
  monkeypatch.setattr(
    accessors, "object_position_error", lambda env: torch.tensor([[0.12, 0, 0]])
  )
  monkeypatch.setattr(accessors, "hand_contact_forces", lambda env: torch.zeros(1, 6))
  monkeypatch.setattr(
    observations, "manipulation_phase", lambda env: torch.tensor([[4.0, 4.0]])
  )
  monkeypatch.setattr(
    "evaluation.locomanip_trace.metric_snapshot",
    lambda env, ids: {"forward_cart_lag": [0.12]},
  )
  dispatch = NS(
    last=np.zeros((1, 6)),
    total=np.zeros((1, 6)),
    square=np.zeros((1, 6)),
    peak=np.zeros((1, 6)),
    active=np.zeros(1),
    error=np.zeros(1),
    count=1,
  )
  trace = SafetyTrace(env, [{"mass_kg": 550}])
  first = trace.update(cast(DispatchTrace, dispatch))
  assert first["signed_cart_error_m"][0][0] == pytest.approx(0.12)
  monkeypatch.setattr(
    accessors, "object_position_error", lambda env: torch.tensor([[0.03, 0, 0]])
  )
  robot.data.root_link_pos_w[0, 2] = 0.55
  env.episode_length_buf[0] = 2
  last = trace.update(cast(DispatchTrace, dispatch))
  trace.finish(torch.tensor([0]), last)
  cart.data.root_link_pos_w[0, 0] = 0.9
  result = trace.results[0]
  assert result is not None
  assert result["max_forward_lag_m"] == pytest.approx(0.12)
  assert result["metrics"]["forward_cart_lag"] == pytest.approx(0.12)
  assert result["min_height_m"] == pytest.approx(0.55)
  assert result["terminal_position_error_m"] == pytest.approx(0.03)
  assert result["combined_failure"]
