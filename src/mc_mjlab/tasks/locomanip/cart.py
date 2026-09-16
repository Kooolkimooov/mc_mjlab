"""Locomanip's passive cart and directional floor contact."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg

from mc_mjlab.robots.robot_module import get_robot_module


def cart_xml_path() -> Path:
  """Resolve the installed module's source asset or an explicit XML override."""
  override = os.environ.get("MC_MJLAB_CART_XML")
  if override:
    path = Path(override).expanduser()
  else:
    module_path = get_robot_module("LMC/Cart").path
    if isinstance(module_path, bytes):
      module_path = module_path.decode()
    path = Path(module_path).parent / "mujoco/model/Cart.xml"
  if not path.is_file():
    raise FileNotFoundError(f"cart XML missing: {path}; set MC_MJLAB_CART_XML")
  return path


def cart_spec() -> mujoco.MjSpec:  # ty: ignore[unresolved-attribute]
  """Load the original 10 kg cart without its embedded ground plane."""
  root = ET.parse(cart_xml_path()).getroot()
  world = root.find("worldbody")
  if world is not None:
    world.attrib.pop("name", None)
  spec = mujoco.MjSpec.from_string(  # ty: ignore[unresolved-attribute]
    ET.tostring(root, encoding="unicode")
  )
  for pair in list(spec.pairs):
    spec.delete(pair)
  spec.delete(spec.geom("local_ground"))
  for i, geom in enumerate(spec.geoms):
    if not geom.name:
      geom.name = "handle" if i == 1 else f"handle_support_{i}"
  return spec


#: The grasped handle sits 0.35 m BEHIND this origin, so the robot's hands reach
#: to `CART_INIT_X - 0.35`, not to `CART_INIT_X`. docs/locomanip.md#cart_init_x
CART_INIT_X = 0.90


def cart_cfg() -> EntityCfg:
  """Create a passive free body in front of the robot's initial stance."""
  return EntityCfg(
    spec_fn=cart_spec, init_state=EntityCfg.InitialStateCfg(pos=(CART_INIT_X, 0, 0))
  )


def cart_floor_contact(spec: mujoco.MjSpec) -> None:  # ty: ignore[unresolved-attribute]
  """Restore the asset's directional friction against the scene's sole plane."""
  planes = [
    g
    for g in spec.geoms
    if g.type == mujoco.mjtGeom.mjGEOM_PLANE  # ty: ignore[unresolved-attribute]
  ]
  if len(planes) != 1:
    raise ValueError("Locomanip requires exactly one scene ground plane")
  spec.add_pair(
    name="cart_floor",
    geomname1="cart/cart_box",
    geomname2=planes[0].name,
    condim=3,
    friction=[0.5, 0.02, 0.005, 0.0001, 0.0001],
  )
