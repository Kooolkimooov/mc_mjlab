"""Locomanip's passive cart, its floor contact and the hand-contact sensors."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from mc_mjlab.robots.robot_module import get_robot_module
from mc_mjlab.tasks.locomanip.mdp.accessors import OBJECT_BODY


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
  """Load the cart asset without the ground plane it embeds."""
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


def cart_nominal_mass_kg() -> float:
  """Read the asset's own mass, which a drawn payload is absolute against."""
  # Never assume 10 kg: the mc_mujoco sweep harness rewrites this file in place.
  # docs/locomanip.md#cart_mass_range_kg
  return cart_spec().body(OBJECT_BODY).mass


def cart_cfg(init_x: float) -> EntityCfg:
  """Create a passive free body in front of the robot's initial stance."""
  # The handle sits 0.35 m BEHIND the cart's origin, so the hands reach to
  # `init_x - 0.35`. docs/locomanip.md#cart_init_x
  return EntityCfg(
    spec_fn=cart_spec, init_state=EntityCfg.InitialStateCfg(pos=(init_x, 0, 0))
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


def hand_cart_contact_sensors(
  hand_bodies: Mapping[str, str],
) -> tuple[ContactSensorCfg, ...]:
  """Report each hand's contact against the cart, for terms that read the grasp."""
  return tuple(
    ContactSensorCfg(
      name=f"{side}_hand_cart",
      primary=ContactMatch(mode="subtree", pattern=body, entity="robot"),
      secondary=ContactMatch(mode="subtree", pattern="Body", entity="cart"),
      fields=("found", "force"),
      reduce="netforce",
      history_length=2,
    )
    for side, body in hand_bodies.items()
  )
