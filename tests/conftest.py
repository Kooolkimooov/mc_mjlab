"""Initialize plugin registration before importing individual contract modules."""

from __future__ import annotations

from pathlib import Path

import mjlab  # noqa: F401
import pytest

from mc_mjlab.bridge.config import get_controller_name
from mc_mjlab.utils.controller_install import controller_is_installed


def requires_controller(mc_rtc_yaml: Path) -> pytest.MarkDecorator:
  """A `pytestmark` skipping a file unless that profile's base controller is installed."""
  controller_name = get_controller_name(mc_rtc_yaml)
  # A mark, not `pytest.skip(allow_module_level=True)`: ROS's launch_testing
  # plugin imports every test file during collection and that raise escapes it.
  return pytest.mark.skipif(
    not controller_is_installed(controller_name),
    reason=f"{controller_name} is not installed in the sourced mc_rtc workspace",
  )
