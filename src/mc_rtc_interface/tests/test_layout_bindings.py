"""Configuration objects exposed at the manager's Python boundary."""

import pytest

import mc_rtc_interface as native


def test_controller_access_is_through_the_manager():
  assert hasattr(native, "ControllersManager")
  assert not hasattr(native, "ControllersHost")
  assert not hasattr(native, "ControllerInstance")


@pytest.mark.parametrize("num_joints", [0, 1, 5])
def test_layout_joint_channels_do_not_overlap(num_joints):
  layout = native.IoLayout()
  names = [f"joint_{i}" for i in range(num_joints)]
  layout.set_joint_order(names)
  assert layout.input.joint_order == layout.output.joint_order == names
  assert layout.input.q_offset() == 0
  assert layout.input.qd_offset() == num_joints
  assert layout.input.tau_offset() == 2 * num_joints
  assert layout.input.root_offset() == 3 * num_joints
  assert layout.input.reset_offset() == 3 * num_joints + 10
  assert layout.input_size == layout.input.reset_offset() + 1
  assert layout.output.q_offset() == 0
  assert layout.output.qd_offset() == num_joints
  assert layout.output.tau_offset() == 2 * num_joints
  assert layout.output.status_offset() == 3 * num_joints
  assert layout.output_size == 3 * num_joints + 1


def test_nested_layout_retains_its_parent():
  layout = native.IoLayout()
  nested = layout.input
  del layout
  nested.body_sensors = ["FloatingBase"]
  assert nested.force_sensors_offset() == 16
