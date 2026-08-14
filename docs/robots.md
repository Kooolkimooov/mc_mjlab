# Robot configuration

The three robots (HRP5P, JVRC1, RHPS1) are parallel by construction: each
`robots/<ROBOT>/<robot>_constants.py` is thin and delegates to the shared helpers
in `robots/*.py`, differing only in robot-specific names — root body, foot bodies,
deactivated joints.

`etc/mc_rtc.yaml`'s `MainRobot` is the single source of truth for which robot
runs. Both the demo and the RL task resolve it through `robots_registry`, so the
two sides cannot drift, and the host raises if the entity's joints do not exist
on the controller's robot.

## Assets

Robot assets (MJCF, meshes, PD gains) are **not tracked** in this repo. Each robot
package symlinks them in on first use from
`$HOME/workspace/install/share/mc_mujoco/<ROBOT>` (`mc_mujoco_assets`).

## mc_rtc_robot_configuration

The controller's `RobotModule` owns the ground truth for refJointOrder, the
half-sitting stance, the default floating-base attitude, and (via `bounds`) the
nominal torque limits. Reading them from the module keeps the mjlab side from
carrying hand-copied transcriptions that drift from the robot the controller
actually drives.

Everything is lazy and cached on purpose: importing this module — and through it
the registry and the per-robot constants — must not require a sourced mc_rtc
workspace, and must not pay the module's construction cost until something needs
controller-derived data.

**Binding quirk:** `stance()` keys come back as `bytes` while `bounds()` keys come
back as `str`; `_decode_joint_key` normalises both.

`_Tree` is built from `joints()`/`bodies()` rather than `jointIndexByName`,
because that binding throws a C++ `std::out_of_range` on a missing name which
terminates the process uncatchably — the same trap `ControllerHost.joint_index`
avoids.

Limbs are derived from mc_rtc's standard sensor names rather than a per-robot
table of body names. The stabilizer is written against those names, so a humanoid
module that runs a walking controller has them by construction.
`get_limb_bodies` raises if one is missing rather than guessing.

## collision_configuration

The mc_mujoco robot XMLs mark collision geoms through MJCF default classes
(`class="collision"`) but leave them **unnamed**, and everything in mjlab that
selects geoms does so by regex on their names. Hence the lifecycle:

1. `name_remaining_collision_geoms` / `is_collision_geom` — give the unnamed geoms
   stable names, so presets and randomization can address them.
2. `group_and_disable_collision_geoms` — stamp the geom-group convention
   (visual 2 / collision 3 / sites 4) and turn every collision off, so a consumer
   re-enables a chosen set rather than inheriting the XML's.
3. Re-enable, one of two ways:
   - `get_collision_presets` — `CollisionCfg` sets for a robot whose geoms are
     named (mjlab applies them by regex), or
   - `enable_all_collision_geoms` — the blanket fallback for a robot whose geoms
     are still unnamed, keyed on the group-3 mark from step 2.

Keeping the whole story in one file keeps the group-3 convention defined and
consumed in the same place: `group_and_disable_collision_geoms` sets it,
`enable_all_collision_geoms` reads it, and they must not drift.

**The flag matters.** `RobotSpec.names_collision_geoms` records that a robot named
its geoms, and `prepare_cfg_for_mc_rtc` keeps the presets only then. A robot that
has *not* named its geoms must fall back to enabling group 3 wholesale, because a
preset matching nothing drops the robot through the floor. The flag cannot be
inferred from a non-empty `EntityCfg.collisions` for exactly that reason.
`prepare_cfg_for_mc_rtc` also always deletes the XMLs' own motors, since mjlab
adds its own.

Primitive geoms carry no mesh to name them after, so the parent body is the only
stable handle. They are not an edge case worth skipping: on RHPS1 the only two are
the sole boxes, i.e. exactly the geoms presets most need to address.

Each robot names its soles semantically so presets can address the feet apart
from the body — one primitive box on the last leg link (HRP5P), one collision mesh
on the ankle-pitch link (JVRC1), one box per foot (RHPS1, whose ankle collision
mesh is commented out in the XML).

## pd_actuator_configuration

Every robot drives its motorized joints with the same natural-frequency PD model,
with per-joint gains derived from the reflected rotor inertia the MJCF already
carries. Those defaults are then overwritten by `PDgains_sim.dat` at action-term
init — see [coupling.md](coupling.md#invariants-and-traps).

Actuators are **unclamped**, like mc_mujoco's PD torque (its motors set
`forcelimited=false`): with the real gains, nominal limits would saturate
constantly and change the stabilizer's behaviour.
`mc_rtc_robot_configuration.get_effort_limits` has the real per-joint torque
scale, kept out of the actuators on purpose. That is also why nothing in the sim
clamps a too-large residual — see
[residual-authority.md](residual-authority.md#residual_scale).

## additional_sensors_configuration

The mc_mujoco XMLs ship the force/IMU sensors the stabilizer needs, but not the
sole velocimeters and root angular-momentum sensor the RL rewards read. Those are
added here, uniformly, keyed on frames every robot already has: the
`lf_force`/`rf_force` sole sites the foot force sensors sit on, and the
floating-base body.

The root angular-momentum sensor is a subtree sensor, which is what makes
`mj_subtreeVel` run — and therefore what makes `subtree_linvel` available to
`com_velocity_tracking`.

## Actuated joint sets

**HRP5P** — every joint in the mc_rtc refJointOrder (all 53, fingers included) is
actuated, matching the 53 motors its MJCF declares. Gains come from
`PDgains_sim.dat` at action-term init, so the fingers get their real gains without
a special case.

**RHPS1** — all 30 rotary joints; no fingers, and the 8 passive slide linkages are
zero-DoF in the module and excluded. Matches the 30 motors the MJCF declares.

**JVRC1** — the hands are an underactuated grasp: five finger joints per side
follow `<side>_UTHUMB` through active `mjEQ_JOINT` equalities with fixed gear
ratios (+/-1, +/-3). That leaves two coherent configurations, and the spec and the
actuator set must pick the same one:

- `True` (default) — keep the couplings and leave the five slaves per hand
  unactuated. One commanded grasp DoF per hand, the fingers following it; this is
  the joint set mc_mujoco motorizes.
- `False` — delete the couplings and actuate all 44, giving mc_rtc the per-finger
  control its refJointOrder and its 44 gain rows imply. Pick this for a task that
  actually manipulates something.

**Actuating a slave while its equality is live** puts the PD and the constraint
solver on one DoF, pulling against each other. The whole finger stance is zero and
zero satisfies every ratio, so the conflict stays invisible until something
commands a finger away from zero — which is why it can hide.

To leave a joint fully passive on any robot, pass it to `get_actuated_joints`'s
`non_actuated`; nothing needs it today.
