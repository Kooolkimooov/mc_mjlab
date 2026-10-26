"""Exact PD sums, stepped-only windows, partial resets and unchanged motors."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch
from conftest import requires_controller
from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.pd_actuator import IdealPdActuator, IdealPdActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.registry import load_env_cfg

from mc_mjlab.robots.pd_telemetry import (
  FIELDS,
  InstrumentedPdActuator,
  PdDemand,
  instrument_pd,
  instrumented_pd_cfg,
)
from mc_mjlab.tasks.locomanip.profiles import HRP5P
from mc_mjlab.tasks.locomanip_study import CONTROL_TASK_ID

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHAIN = """
<mujoco><worldbody>
  <body name="a"><joint name="j0" axis="0 0 1"/><geom size="0.1" mass="1"/>
    <body name="b" pos="0 0 0.2"><joint name="j1" axis="0 1 0"/><geom size="0.1"/>
      <body name="c" pos="0 0 0.2"><joint name="j2" axis="1 0 0"/><geom size="0.1"/>
      </body></body></body>
</worldbody></mujoco>
"""
PARTS = (
  IdealPdActuatorCfg(
    target_names_expr=("j0",),
    stiffness=50.0,
    damping=2.0,
    effort_limit=3.0,
    armature=0.01,
    frictionloss=0.1,
  ),
  IdealPdActuatorCfg(
    target_names_expr=("j1",), stiffness=80.0, damping=4.0, effort_limit=5.0
  ),
  IdealPdActuatorCfg(
    target_names_expr=("j2",), stiffness=30.0, damping=1.0, viscous_damping=0.3
  ),
)
MOTOR_FIELDS = (
  "actuator_trnid",
  "actuator_gear",
  "actuator_forcerange",
  "actuator_ctrlrange",
  "dof_armature",
  "dof_frictionloss",
  "dof_damping",
)


def command(*values: torch.Tensor) -> ActuatorCmd:
  zeros = torch.zeros_like(values[0])
  return ActuatorCmd(*values, *(zeros,) * (5 - len(values)))


def chain(actuators: tuple, num_envs: int = 2) -> tuple[Entity, Simulation]:
  cfg = EntityCfg(
    spec_fn=lambda: mujoco.MjSpec.from_string(CHAIN),
    articulation=EntityArticulationInfoCfg(actuators=actuators),
  )
  entity = Entity(cfg)
  model = entity.compile()
  sim = Simulation(num_envs=num_envs, cfg=SimulationCfg(), model=model, device=DEVICE)
  entity.initialize(model, sim.model, sim.data, DEVICE)
  return entity, sim


def test_sample_matches_the_ideal_law_and_peak_contributions() -> None:
  generator = torch.Generator().manual_seed(123)
  shape = (3, 5)
  demand = PdDemand(torch.zeros(shape), 0.001)
  raw = []
  for _ in range(20):
    cmd = command(*(torch.randn(shape, generator=generator) for _ in range(3)))
    params = {
      "stiffness": torch.rand(shape, generator=generator) * 100,
      "damping": torch.rand(shape, generator=generator) * 10,
      "force_limit": torch.rand(shape, generator=generator) * 30,
    }
    sample = demand.evaluate(cmd, *params.values())
    assert torch.equal(
      sample[FIELDS.index("clipped_nm")], IdealPdActuator.control_law(params, cmd)
    )
    demand.record(sample)
    raw.append(sample[FIELDS.index("raw_nm")])

  trace = torch.stack(raw)
  window = demand.snapshot()
  assert torch.equal(window["peak_abs_nm"], trace.abs().amax(dim=0))
  parts = ("p_nm", "d_nm", "feedforward_nm")
  contributions = torch.stack([window[f"at_total_peak/{name}"] for name in parts])
  assert torch.allclose(contributions.sum(dim=0), window["at_total_peak/raw_nm"])
  assert torch.allclose(
    window["at_total_peak/time_s"], trace.abs().argmax(dim=0).float() * 0.001
  )


def test_saturation_runs_cross_windows_and_reset_clears_selected_rows() -> None:
  ones = torch.ones(2, 1)
  demand = PdDemand(ones, 0.01)
  for value in (2.0, 3.0):
    demand.record(demand.evaluate(command(ones * value), ones, ones, ones))
  demand.begin_interval()
  demand.record(demand.evaluate(command(ones * 7), ones, ones, ones))
  window = demand.snapshot()
  assert torch.allclose(window["saturated_s"], ones * 0.01)
  assert torch.allclose(window["longest_continuous_s"], ones * 0.03)

  demand.reset(torch.tensor([0]))
  assert torch.equal(demand.terminal["peak_abs_nm"], torch.tensor([[7.0], [0.0]]))
  assert demand.steps[0] == 0 and demand.steps[1] == 3
  assert demand.peak[0] == 0 and demand.peak[1] == 7
  assert torch.equal(window["peak_abs_nm"], ones * 7)


@pytest.mark.parametrize("parts", [PARTS, PARTS[::-1]], ids=["joint", "reversed"])
def test_instrumented_chain_keeps_motors_and_trajectory_bitwise(parts: tuple) -> None:
  reference, reference_sim = chain(parts)
  probe, probe_sim = chain((instrumented_pd_cfg(parts, 0.002, 2),))
  for field in MOTOR_FIELDS:
    assert np.array_equal(
      getattr(reference_sim.mj_model, field), getattr(probe_sim.mj_model, field)
    ), field
  actuator = probe.actuators[0]
  assert isinstance(actuator, InstrumentedPdActuator)
  assert actuator.telemetry is not None
  # Live gains are what the action term overwrites at init.
  for entity in (reference, probe):
    owner = next(a for a in entity.actuators if "j2" in a.target_names)
    assert isinstance(owner, IdealPdActuator) and owner.stiffness is not None
    owner.stiffness[:, owner.target_names.index("j2")] = 400.0

  generator = torch.Generator().manual_seed(7)
  for _ in range(6):
    target = torch.randn(2, 3, generator=generator).to(DEVICE)
    for entity, sim in ((reference, reference_sim), (probe, probe_sim)):
      entity.set_joint_position_target(target)
      entity.write_data_to_sim()
      sim.step()
      entity.update(0.002)
    assert torch.equal(reference_sim.data.ctrl, probe_sim.data.ctrl)
    assert torch.equal(reference_sim.data.qpos, probe_sim.data.qpos)

  telemetry = actuator.telemetry
  assert torch.equal(telemetry.samples, torch.full((2, 3), 2.0, device=DEVICE))
  assert torch.all(telemetry.last[FIELDS.index("kp")][:, -1] == 400.0)
  applied = probe_sim.data.ctrl[:, actuator.ctrl_ids]
  assert torch.equal(telemetry.last[FIELDS.index("clipped_nm")], applied)


def test_reset_writes_are_not_samples_and_resets_are_partial() -> None:
  probe, sim = chain((instrumented_pd_cfg(PARTS, 0.002, 4),))
  actuator = probe.actuators[0]
  assert isinstance(actuator, InstrumentedPdActuator)
  telemetry = actuator.telemetry
  assert telemetry is not None
  probe.set_joint_position_target(torch.ones(2, 3, device=DEVICE))
  for _ in range(3):
    probe.write_data_to_sim()
    sim.step()
    probe.update(0.002)

  probe.reset(torch.tensor([0], device=DEVICE))
  probe.write_data_to_sim()
  assert telemetry.terminal["samples"][0].eq(3).all()
  assert telemetry.samples[0].eq(0).all() and telemetry.samples[1].eq(3).all()
  probe.write_data_to_sim()
  sim.step()
  probe.update(0.002)
  assert telemetry.steps[0].eq(1).all() and telemetry.steps[1].eq(4).all()


def test_mismatched_parts_are_rejected() -> None:
  delayed = IdealPdActuatorCfg(
    target_names_expr=("j2",), stiffness=1.0, damping=1.0, delay_max_lag=2
  )
  with pytest.raises(ValueError, match="delay"):
    instrumented_pd_cfg((*PARTS[:2], delayed), 0.002, 2)
  overlapping = IdealPdActuatorCfg(
    target_names_expr=("j[01]",), stiffness=1.0, damping=1.0
  )
  with pytest.raises(ValueError, match="claimed|matched"):
    chain((instrumented_pd_cfg((*PARTS, overlapping), 0.002, 2),))


@requires_controller(HRP5P.mc_rtc_yaml)
def test_study_robot_motors_are_unchanged() -> None:
  original = load_env_cfg(CONTROL_TASK_ID)
  instrumented = load_env_cfg(CONTROL_TASK_ID)
  instrument_pd(instrumented)
  models = [
    Entity(cfg.scene.entities["robot"]).compile() for cfg in (original, instrumented)
  ]
  assert models[0].nu == models[1].nu > 0
  for field in MOTOR_FIELDS:
    assert np.array_equal(getattr(models[0], field), getattr(models[1], field)), field
  names = [
    [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    for m in models
  ]
  assert names[0] == names[1]
