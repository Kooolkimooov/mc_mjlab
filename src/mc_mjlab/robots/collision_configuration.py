"""Collision handling for the mc_rtc robot entities

The mc_mujoco robot XMLs mark collision geoms through MJCF default classes
(``class="collision"``) but leave them unnamed, and everything in mjlab that
selects geoms does so by regex on their names. So the collision lifecycle is:

1. ``name_remaining_collision_geoms`` / ``is_collision_geom`` -- give the
   unnamed geoms stable names, so presets and randomization can address them.
2. ``group_and_disable_collision_geoms`` -- stamp the geom-group convention
   (visual 2 / collision 3 / sites 4) and turn every collision off, so a
   consumer re-enables a chosen set rather than inheriting the XML's.
3. re-enable, one of two ways:
   - ``get_collision_presets`` -- ``CollisionCfg`` sets for a robot whose geoms
     are named (mjlab applies them by regex), or
   - ``enable_all_collision_geoms`` -- the blanket fallback for a robot whose
     geoms are still unnamed, keyed on the group-3 mark from step 2.

Keeping the whole story in one file keeps the group-3 convention defined and
consumed in the same place: ``group_and_disable_collision_geoms`` sets it,
``enable_all_collision_geoms`` reads it, and they must not drift.
"""

from __future__ import annotations

import mujoco
from mjlab.utils.spec_config import CollisionCfg

##
# Geom naming.
##


def is_collision_geom(geom: mujoco.MjsGeom) -> bool:
  """Whether the geom takes part in contacts *as authored in the MJCF*.

  Only meaningful before ``group_and_disable_collision_geoms`` blanks
  ``contype``/``conaffinity``; call the namers ahead of that.
  """
  return bool(geom.contype or geom.conaffinity)


def _geom_name_stem(geom: mujoco.MjsGeom) -> str:
  meshname = getattr(geom, "meshname", "")
  if meshname:
    return meshname[:-5] if meshname.endswith("_mesh") else meshname
  # Primitive geoms carry no mesh to name them after, so the parent body is
  # the only stable handle. They are not an edge case worth skipping: on
  # RHPS1 the only two are the sole boxes, i.e. exactly the geoms presets
  # most need to address.
  body = geom.parent
  name = body.name if body is not None else ""
  return name or "geom"


def name_foot_collision_geoms(spec: mujoco.MjSpec, foot_bodies: dict[str, str]) -> None:
  """Give each foot's collision geom a semantic name.

  ``foot_bodies`` maps a foot body name to the name to assign its collision
  geom, e.g. ``{"L_ANKLE_P_LINK": "left_foot_collision"}``. Run this *before*
  ``name_remaining_collision_geoms`` so the feet land in the foot bucket: a
  mesh/body derived name would match the body regex instead, and the feet would
  then get body contact params. The assigned name also encodes left/right, which
  no derived name could.

  Raises if a foot body does not hold exactly one unnamed collision geom -- a
  silently unnamed sole makes the foot preset match nothing, and the robot then
  walks on whatever the fallback happens to enable.
  """
  found: dict[str, list[mujoco.MjsGeom]] = {body: [] for body in foot_bodies}
  for geom in spec.geoms:
    body = geom.parent
    body_name = body.name if body is not None else ""
    if body_name in foot_bodies and not geom.name and is_collision_geom(geom):
      found[body_name].append(geom)

  for body_name, geom_name in foot_bodies.items():
    geoms = found[body_name]
    if len(geoms) != 1:
      raise ValueError(
        f"expected exactly one unnamed collision geom on {body_name} to name "
        f"'{geom_name}', found {len(geoms)}. The foot geometry changed; update "
        "the robot's foot-body mapping."
      )
    geoms[0].name = geom_name


def name_remaining_collision_geoms(spec: mujoco.MjSpec, prefix: str) -> tuple[str, ...]:
  """Name every unnamed collision geom ``<prefix>_collision_<stem>``.

  Already-named geoms are left alone, so a caller can name the few geoms it
  wants to address semantically (feet) before handing the rest over here.
  Returns the names assigned, in spec order.
  """
  assigned: list[str] = []
  taken = {geom.name for geom in spec.geoms if geom.name}
  for geom in spec.geoms:
    if geom.name or not is_collision_geom(geom):
      continue
    name = f"{prefix}_collision_{_geom_name_stem(geom)}"
    if name in taken:
      # One body can hold several collision geoms, and one mesh can be
      # instanced more than once.
      suffix = 2
      while f"{name}_{suffix}" in taken:
        suffix += 1
      name = f"{name}_{suffix}"
    geom.name = name
    taken.add(name)
    assigned.append(name)
  return tuple(assigned)


##
# Geom-group convention + default-off.
##

COLLISION_GROUP = 3
"""Geom group the collision geoms are stamped with; the fallback keys on it."""


def group_and_disable_collision_geoms(spec: mujoco.MjSpec) -> None:
  """Apply the geom-group convention, then turn all collisions off.

  Visual geoms -> group 2, collision geoms -> ``COLLISION_GROUP`` (3), sites ->
  group 4, read from the MJCF's ``class="visual"/"collision"`` split (which
  sets contype/conaffinity). Collisions are then disabled so consumers
  re-enable a chosen set: named-geom presets, or ``enable_all_collision_geoms``.
  """
  for geom in spec.geoms:
    geom.group = 2 if (geom.contype == 0 and geom.conaffinity == 0) else COLLISION_GROUP
  for site in spec.sites:
    site.group = 4
  for geom in spec.geoms:
    geom.contype = 0
    geom.conaffinity = 0


def enable_all_collision_geoms(spec: mujoco.MjSpec) -> None:
  """Re-enable every collision geom wholesale, by group.

  The fallback for a robot whose geoms are unnamed: name-based presets match
  nothing, so without this the robot has no collisions and sinks through the
  floor. Keyed on the group-3 mark ``group_and_disable_collision_geoms`` left.
  """
  for geom in spec.geoms:
    if geom.group == COLLISION_GROUP:
      geom.contype = 1
      geom.conaffinity = 1


##
# Named-geom presets.
##


def get_collision_presets(
  prefix: str, foot_expr: str
) -> tuple[CollisionCfg, CollisionCfg, CollisionCfg]:
  """``(feet_only, full, full_without_self)`` for a robot whose feet match
  ``foot_expr`` and whose other collision geoms are named ``<prefix>_collision_*``.

  Contact parameters follow mc_mujoco (condim 3, MuJoCo's default friction),
  since the stabilizer is a force-feedback loop tuned against it; ``priority=1``
  on the feet keeps the foot/floor mix from depending on the terrain's friction.

  - ``feet_only``: only the feet touch anything; the body passes through.
  - ``full``: every named geom live, self-collisions included (contype 1 /
    conaffinity 1) -- the safe default, matching the XML authors.
  - ``full_without_self``: contype 0 so body geoms never initiate a pair, while
    the terrain's own contype still finds them, so a fallen robot lands.
  """
  core = foot_expr.removeprefix("^").removesuffix("$")
  all_expr = rf"^({core}|{prefix}_collision_.*)$"
  feet_only = CollisionCfg(
    geom_names_expr=(foot_expr,),
    condim=3,
    priority=1,
    disable_other_geoms=False,
  )
  full = CollisionCfg(
    geom_names_expr=(all_expr,),
    contype=1,
    conaffinity=1,
    condim=3,
    priority={foot_expr: 1},
    disable_other_geoms=False,
  )
  full_without_self = CollisionCfg(
    geom_names_expr=(all_expr,),
    contype=0,
    conaffinity=1,
    condim=3,
    priority={foot_expr: 1},
    disable_other_geoms=False,
  )
  return feet_only, full, full_without_self
