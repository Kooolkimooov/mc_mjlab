# The mc_rtc coupling

## McRtcResidualActionBase

**Current:** the action owns one native `ControllersManager` and two Python
shared-memory blocks described by `WorkerStartMessage`. The manager is closed
before either block is unlinked, including initialization failure and finalization.
Python retains simulation wiring, real PD gains, residual subsets, safety
projection and interpolation. Native workers execute controllers.

Control solves are dispatched for every row at shared control boundaries and
collected one control period later. Native `collect()` returns failed rows, not
completed rows; Python tracks whether a dispatch is pending. Environment
decimation must be divisible by positive `frameskip`. Failed rows never publish
new targets or datastore values. Unserviced output statuses are marked failed
before dispatch so stale successful payloads cannot be mistaken for fresh output.

**Re-measure if:** frameskip, environment decimation or native collection changes.

**History:** the ten legacy Python backend/helper modules were removed during the
native migration. [Earlier measurements](coupling-history.md) retain their sample
sizes and original values; they do not validate this integration.

## SimControllerBridge

**Current:** input and output columns use native offset methods. Python scatters
all simulated reference-order joints, including passive joints, and retains
RobotModule stance positions for unsimulated entries. Encoders include bias;
velocities and measured actuator efforts use the corresponding simulation joints.
Action outputs gather only target joints; public `alpha` reads native `qd`.

Root position is environment-local. MuJoCo's wxyz quaternion is converted to
native xyzw after feedback offsets. The root block has ten values: position,
quaternion and linear velocity. Every body sensor, including `FloatingBase`, has
a separate six-value gyro/acceleration slot. FloatingBase receives raw free-joint
angular velocity and linear acceleration; other sensors use named IMU readings.
Force/torque slots hold MuJoCo readings; native code performs the sign conversion.
Root feedback rotates the IMU readings and root quaternion; wrench feedback uses
the full native force-sensor ordering.

**Re-measure if:** RobotModule joint/sensor metadata or MuJoCo conventions change.

**History:** the legacy layout used target-order joints and a sixteen-value root
block. Native mapping and geometry are covered by `tests/test_native_action_contracts.py`.

## Reset pose seeding

**Current:** reset drains outstanding work before shared inputs can change. Selected
rows queue reset flags, discard staged targets and seed interpolation from biased
joint positions (velocity and torque start at zero). Other rows retain their
staged targets and control phase. Native reset seeds the real robot, resets the
controller plan and observer pipelines, then steps using current sensor inputs.
Episode failure latches clear on the Python episode reset; queued reset state
persists until a collected native result acknowledges it.

**Re-measure if:** native controller reconstruction or observer reset order changes.

**History:** [pose-seeding measurements](coupling-history.md#reset-pose-seeding).

## controller logs

**Current:** `ControllerInstance::finish_reset` explicitly removes
`perf_UpdateContacts` before its second `MCController::reset`. Global
initialization has already called that reset once. The second call is needed
after measured poses and sensors are seeded, but the installed logger's
`addLogEntry(..., overwrite=true)` erases the old callback and appends a new
`KeyAddedEvent` without a `KeyRemovedEvent`. The binary reader then has one more
key than the writer has values, shifting all subsequent columns. A type mismatch
can make those columns appear entirely NaN in both `mc_bin_to_flat` and
`mc_log_ui`; they share the same reader and are not independent checks.

Removing the entry first emits the missing event without changing controller
execution. `test_host_logging` uses the real base reset and checks a vector
registered during `run()` through `mc_rtc::log::FlatLog`, after both initialization
and an episode reset. It fails without the removal and passes with it.

**Re-measure if:** mc_rtc fixes overwrite event serialization, the second reset
is removed, or another reset callback starts overwriting log entries.

**History:** 2026-09-29 -- traced the 251 NaN manager/task channels reported in
[the JVRC1 blend investigation](locomanip.md#jvrc1-hand-force-blend-in-mc_mjlab)
to a duplicate `perf_UpdateContacts` add in sample zero. The raw MessagePack
values were finite. Inserting a remove event immediately before that duplicate
in a copy of `mjlab25-LocomanipController-2026-09-29-14-11-57.bin` let the standard
converter recover **12,498 / 12,500** finite `FootManager_refZmp_x` and
`CentroidalManager_ZMP_measured_x` samples; the first two precede registration.
A new 15 s JVRC1 zero-residual demo with the bridge fix gave **7,498 / 7,500**
finite samples on both channels, **7,500 / 7,500** finite `perf_UpdateContacts`
samples, minimum root z **0.691 m**, and both hands in Hold at the end. All 17
native checks passed. This repairs observability, not the blend's stability.

Old `.flat` exports remain corrupt. Recover the key events in a copy of the
original `.bin` before converting again; do not replace NaNs with zeros or infer
that the controller's live managers held NaNs.

## Worker failure is a truncation

**Current:** native `OutputLayout.Status` is `OK=0`, `QP_FAILED=1`,
`WORKER_FAILED=2`. The action maintains separate `controller_failed` and
`controller_worker_failed` episode latches. Only the infrastructure failure term
has `time_out=True`; it bootstraps and avoids the fall penalty. Failed worker
payloads are excluded even if their old status is successful. Recovery belongs
to the native manager; Python queues reset flags for affected rows.

Native recovery is implemented in `ControllersManager`. Deterministic manager
doubles exercise the Python handshake, while native tests exercise live process
replacement. A dispatch send failure, reply error,
receive failure or timeout quarantines that worker's complete row slice. The
manager collects the other replies, kills and reaps every failed generation, and
returns the failed row ids immediately. The action marks those rows
`WORKER_FAILED`, retains their reset requests and ignores their payloads.

The action asks the manager to respawn from its episode-reset callback. A worker
with multiple rows waits until every failed row in its slice has reset, then its
replacement starts on a generation-specific IPC endpoint. Startup binds the
shared-memory spans without writing outputs or initializing controllers; the
next reset-bearing step initializes them from current simulation state. A
replacement that cannot start is a fatal manager error: the manager closes every
worker rather than retrying forever or leaving a partially usable pool.

**Re-measure if:** native quarantine/recovery, startup cost or termination
configuration changes.

**History:** the old Python pool implemented kill/respawn quarantine. Its retained
measurements and failure evidence are in [the historical notes](coupling-history.md).

## datastore_scalar_inputs

**Current:** `datastore_scalar_inputs` and `datastore_vectors_inputs` name
`double` and `Eigen::Vector3d` datastore *setters*. They are the unconditional
setter path: the native step writes every declared input column into the controller's datastore
before `run()`, on every control period, with no usage flag, no baseline and no
restore — which is why they work on the current native interface, while the
gated pairs still need usage flags it does not expose.

The action term owns one value buffer per kind, zero until a task writes it.
`set_datastore_scalar_input(setter, values)` takes `(num_envs,)` and
`set_datastore_vector_input(setter, values)` takes `(num_envs, 3)`;
`datastore_scalar_input` / `datastore_vector_input` read the fed value back.
A value holds until set again, across episode resets, because a fed setter is a
property of the task and not of the episode — the rule the removed scalar holds
followed. docs/controller-timing.md#removed-datastore_scalar_input_commands

One hazard: a declared setter is
written from the very first controller step, before any task code has run, so
its zero default reaches the controller unless the task feeds it during
construction.

**Re-measure if:** the native input block gains usage flags, or `apply_input`
stops writing datastore columns every step.

**History:**
- 2026-09-15 — both fields existed on the cfg with output-shaped docstrings and
  no reader; wired to the layout, the value buffers and the per-period write.
  The walking reference moved onto them when the gated pairs were removed.

## required_controller

**Current:** a cfg field on `McRtcResidualActionCfg`, `None` by default, naming
the controller a term's calls only exist in; `_validate_cfg` compares it to the
config's `Enabled` (`bridge.config.get_controller_name`) before a worker
starts, and its error names the field so a controller known to provide the same
calls can be accepted. Only `WalkingReferenceActionCfg` redeclares it,
defaulting to `LogisticController_ismpc`, the sole provider of the
`ismpc_walking::get_ref_vel` / `set_ref_vel` pair; every other action term
leaves it `None` and runs on whatever controller is enabled. Without the check a
mismatch surfaces as the native "controller initialization failed" raise from
the missing callbacks, which names the datastore entries rather than the
controller.

**Re-measure if:** another term starts calling controller-specific entries, or
the walking reference moves to a differently named controller.

**History:**
- 2026-09-15 -- added `required_controller`, replacing the walking-reference
  cfg's own `walking_controller` field and its `__post_init__` raise. That raise
  fired during task registration, where mjlab's task loader turns an exception
  into a `[WARN]` and a traceback: the run then failed as an unknown task id
  rather than a wrong controller. `_validate_cfg` sees the final cfg, after any
  `--env.*` override, and stops the run itself.

## controller_is_installed

**Current:** `utils/controller_install.py` answers whether the sourced
workspace ships a named controller, and `tests/conftest.py`'s
`require_controller` skips a whole test module when it does not. A controller
counts as installed if the workspace holds **either** its module
(`<lib>/mc_controller/<name>.so`) **or** a configuration for it
(`<lib>/mc_controller/etc/<name>.{conf,yaml,yml}`, or a per-robot file under
`<lib>/mc_controller/<name>/`). Both halves are needed: a variant like
`LogisticController_ismpc` is configuration over `LogisticController`'s module
and has no `.so` of its own, while a plain controller may ship only the module
and take its whole configuration from the caller.

The guard is a `pytestmark` skipif, never `pytest.skip(allow_module_level=True)`:
a sourced ROS workspace puts `launch_testing`'s pytest plugin on the path, and
its `pytest_pycollect_makemodule` imports *every* `test_*.py` in the directory
looking for a launch entrypoint. A `Skipped` raised while that hook imports a
module escapes it and ends the whole session -- `1 skipped` and
`found no collectors for tests/test_policy.py`, with the other 96 tests never
run. A mark is evaluated after the import and cannot do that.

The mark only works because `tasks/locomanip/__init__.py` registers nothing when
`CONTROLLER_NAME` is absent, warning instead. Its cfgs read the cart asset while
they are built (`cart_nominal_mass_kg`), so registering unconditionally made
`import mc_mjlab.tasks` raise `FileNotFoundError` -- and every test module starts
with that import, so a missing Locomanip took the residual_balance,
residual_mpc and residual_feedback tests down with it.

`rl/controller_provenance.py` reads the same `controller_config_paths`, so what
a checkpoint records as its base controller and what the tests check for being
installed cannot drift apart.

**Re-measure if:** mc_rtc changes where controllers or their configurations
install, or a task starts loading an asset from a controller package the guard
does not name.

**History:** 2026-09-18 -- added after locomanip registered on a second robot;
until then every controller the tests touch happened to be installed on the one
machine that ran them.

## user_controller_config_paths

**Current:** provenance records the installed configuration *and* the three
files mc_rtc would merge over it from the user's own tree:
`~/.config/mc_rtc/controllers/<name>.{conf,yaml,yml}`, plus every per-robot file
already sitting under `~/.config/mc_rtc/controllers/<name>/`. The first three are
recorded whether or not they exist, because an absent one is a file that takes
effect the moment somebody creates it -- and creating it can move a hand
transform or a gain while the checkpoint's own YAML is untouched.

This list is deliberately **not** part of `controller_config_paths`, which also
backs `controller_is_installed`: a user override alone does not make a
controller loadable, and folding it in would report a controller the workspace
never shipped as installed.

`provenance_signature` now drops absent records instead of encoding them. Both
sides of the comparison are reduced by the same function, so this is not a
weakening: a file present at training and missing now still loses its record and
still mismatches, and a file that *appears* still adds one. What it buys is that
a role added to provenance later -- these three, or the next ones -- does not by
itself invalidate every checkpoint taken before it existed.

**Re-measure if:** mc_rtc changes its user configuration search path, or a
controller starts reading configuration from a third location.

**History:** 2026-09-28 -- added with the review that found provenance searching
installed directories only. `~/.config/mc_rtc/controllers/` was empty on the
development machine, so nothing had yet diverged.

## object-state-feedback

**Current:** the object block is 16 doubles per controller object -- position, xyzw
orientation, linear velocity, world angular velocity, linear acceleration -- and
`apply_input` both places `realRobot(object)` and sets that object's
`FloatingBase` body sensor from them, matching what mc_mujoco's `mj_sim.cpp`
does for every floating-base robot.

**The two simulators deliver the measured object by different routes**, which is
why their logs disagree. Measured at 10 kg on JVRC1:

| channel | mc_mjlab | mc_mujoco |
| --- | --- | --- |
| `obj_ff_tx` (reference robot) | 0.750 -> 1.164 | 0.750 -> 1.750 |
| `obj_ff_real_tx` (measured robot) | 0.750 -> 1.399 | **0.7500, every sample** |
| `obj_FloatingBase_position_x` (sensor) | **0, before this change** | 0.750 -> 1.848 |

mc_mujoco writes only the sensor and leaves the measured robot frozen at its init
pose; before this change mc_mjlab wrote only the robot. Neither is starved: both
carry the pose. `realObj()` is `realRobot("obj")`, and inside the controller its
only writers are ManipManager's ROS callbacks, which `enableManagerUpdate`/
`enableRos: false` leaves idle -- so mc_mujoco runs the whole cart demo with a
stale measured object and completes it. Readers are `ConfigManipState`'s
`preUpdateObj` branch (default false), a GUI button, and two log entries.

**No consumer of the sensor is identified.** The change closes a real parity gap
and it does something measurable -- on the JVRC1 10 kg gate it lifted bilateral
contact from 14.1 s to 20.6-26.0 s and forward travel from 0.59 m to 0.75-1.06 m
-- but the robot still falls to the same 0.549 m in 3 of 3 cycles, so it is not
the fix for docs/locomanip.md#JVRC1-hand-force-blend-in-mc_mjlab.

**It also makes mjlab logs readable by the sweep tooling.**
`logs/locomanip/mass-sweep/analyze_log.py` reads the object position from
`obj_FloatingBase_position_x`, so before this change it reported a stationary
cart for every mc_mjlab log while working correctly on mc_mujoco ones. That trap
cost a wrong diagnosis; see the History below.

**Re-measure if:** mc_rtc starts running an observer pipeline for object robots,
or a controller begins reading `realObj()` on a live path.

**History:** 2026-09-29 -- added while chasing the JVRC1 blend failure, on the
theory that the controller was planning against an object at the world origin.
That theory was wrong: `obj_ff_real_tx` had always been correct and only the
sensor channel read zero. The parity is still worth having, and the acceleration
column is new, but the justification in the first version of this section did not
survive contact with the logs.

## Datastore callbacks

**Current:** there is no alias layer. A task names native callbacks directly in
`datastore_vectors_outputs` / `datastore_scalar_outputs`, and
`datastore_vector_output` / `datastore_scalar_output` read them back under the
same names. The adapter's four
getters are constants in `mc_mjlab/bridge/controller_datastore.py` (`PLANNED_ZMP`,
`CONTROL_COM`, `CONTROL_COM_VEL`, `SUPPORT_FOOT`); a controller's own getters are
constants in the task that reads them, as `residual_mpc/mdp.py` holds the
`ismpc_walking::` ones. `SUPPORT_FOOT` is numeric (right=0, left=1) — ismpc's own
`ismpc_walking::support_foot_name` returns a string the layout cannot carry, so
the plugin converts it. Every configured name must validate in native
initialization; missing callbacks are errors.

**History:**
- 2026-09-15 — the two alias maps were deleted with the walking-reference move.
  They mapped four vector and two scalar public names onto these callbacks, and
  the one mapping that was not cosmetic (`ismpc_walking::support_foot_name` →
  `mc_mjlab::support_foot`) is now written out at its single use site.

**Removed 2026-09-15: the gated `DatastoreCommands` pairs.** They carried one
usage flag per callback per environment, captured a baseline on activation, sent
absolute baseline-plus-offset values and restored the baseline once on
deactivation. None of it ever ran here: the flags needed
`use_datastore_scalar_offset()` / `use_datastore_vector3_offset()`, which the
native `InputLayout` does not define, so constructing a non-empty pair list
raised — including for the walking reference, whose action term therefore failed
in `__init__`. Setters now go through `datastore_scalar_inputs` /
`datastore_vectors_inputs` above, and a term that wants a baseline latches the
getter itself (docs/walking-reference.md#_feed_walking_reference).

The native interface accepts only `double` and `Eigen::Vector3d` callbacks.
Getters return values or const references; setters accept values or const
references. Unsupported inventory:

| Type or signature | Examples |
|---|---|
| `bool` | Walking-state getters, `set_disturbance` |
| Integer types | `set_n_step(int)` |
| `std::string` | Support-foot and swing-foot names |
| Other numeric types and Eigen shapes | `float`, Vector2, VectorXd, matrices, quaternions |
| Spatial types | `sva::PTransformd`, `MotionVecd`, `ForceVecd` |
| Custom objects and handles | `ControllerConfiguration`, `WalkingInterface` shared pointers |
| Containers and raw datastore values | Footstep arrays; entries without getter/setter callbacks |
| Other callable signatures | Zero-argument commands, multiple arguments, mutable-reference getters |

An external controller adapter must expose required values using supported
numeric callbacks. The `mc_mjlab::` names are supplied in-process by
[instance_datastore_plugin](#instance_datastore_plugin) rather than by a
controller.

**Re-measure if:** callback signatures, usage-flag semantics or adapter values change.

**History:** baselines used to live in Python controller hosts. They now belong to
the simulation-side command state; native inputs always contain absolute values.

## instance_datastore_plugin

**Current:** `src/mc_rtc_interface/{hpp,cpp}/instance_datastore_plugin.*` is the
numeric adapter, in-process rather than in a controller. A layout name carrying
`IoLayout::plugin_prefix` (`mc_mjlab::`) is registered on the controller's own
datastore by `ControllerInstance::finish_reset`, bound to the namespace function
whose name is the rest of the entry: `control_com`, `control_com_vel`,
`planned_zmp` and `support_foot`. Registration happens on every controller build
because `MCGlobalController::reset()` destroys and rebuilds the controller with
its datastore. A prefixed name matching no function throws before validation, as
does a name the controller already defines: two definitions of one quantity with
no way to tell which is live.

`support_foot` converts the controller's `std::string`
`ismpc_walking::support_foot_name`, which the numeric callback contract cannot
carry, to right=0 / left=1 by its `Left`/`Right` prefix, and throws on a name
that is neither. Measured live under `LogisticController_ismpc`: values `{0, 1}`
with 3 switches in 400 policy steps (4 s), consistent with the gait period.

The plugin reads `MCController::robot()`, the robot the QP integrates. Measured
live with a zero residual on `residual_balance`/HRP5P: `control_com` z≈0.98 m,
`|planned_zmp - control_com|` up to 0.019 m over 120 steps — non-zero, so the
control robot really is forward-accelerated. Zero there would mean the value has
silently collapsed to the CoM. `validate_dcm_objective` reads `control_com_vel`
as its commanded speed and separates the regimes: 0.1096 m/s mean walking
(`targetCmdVel: [0.1, 0, 0]`) against exactly 0.0000 standing, 17600 grounded
samples each.

**Re-measure if:** the prefix, the function set, or the registration point in
`finish_reset` changes.

**History:** these four names were declared in `controller_datastore.py` and
provided by nothing, so any layout requesting them failed
`utils::datastore::validate` at controller init. A controller-side alternative
was rejected: the doubles previously patched into `ismpc_walking` are reverted
by a workspace rebuild, and they only ever covered that one controller.

## planned_zmp

**Current:** `instance_datastore_plugin::planned_zmp` publishes the
control-centroid ZMP, using the QP control robot's CoM and commanded
acceleration. For gravity `g=9.81`, horizontal
ZMP is `com_xy - com_z * acceleration_xy / (acceleration_z + g)`, with z=0;
when `abs(acceleration_z + g) < 1e-3`, retain the historical `com_xy, z=0`
free-fall convention. Never substitute ISMPC's delay-compensated reachable
`zmp_target` or canonical output-robot dynamics. The canonical robot's dynamic
fields can be zero because the converter copies only joint channels.

**Re-measure if:** the adapter's centroid calculation or ZMP plane changes.

**History:** [definition and delay-compensation evidence](coupling-history.md#planned_zmp).
Checkpoint validation remains intact and must reject changed effective contracts.

## Console output

**Current:** `controller_timeout_ms=60000` retains the old operational collection
timeout. `console_output` writes native row log flags: none, environment zero
(`single`), or all. Python worker-file logging and in-process execution switches
have been removed. `MC_MJLAB_PRINT_RESIDUAL` still controls the Python residual
print interval.

Native fd guards do **not** silence C++ spdlog, and cannot — mc_rtc's loggers
are asynchronous, so the scoped guard races the drain, and worker startup logs
before any row exists. `console_output="none"` leaks on both paths. See
[OutputGuard](process-workers.md#outputguard).

**Re-measure if:** timeout semantics or native row logging changes.

**History:**
- the native binding's standalone timeout default is 5 ms; the action passes its
  explicit 60000 ms default.
- 2026-09-11 — "Native fd guards silence C++ spdlog" was wrong; corrected with
  the reproduction in [OutputGuard](process-workers.md#outputguard).

## IoLayout

**Current:** `src/mc_rtc_interface/hpp/io_layout.hpp` defines all offsets;
`hpp/ipc_socket.hpp` defines the worker protocol. The architecture generator
extracts these sources into the shared-memory and protocol pages. Python never
assumes the status precedes datastore output or that alpha is a native channel.

**Re-measure if:** native layout or IPC declarations change.

**History:** [legacy column map](coupling-history.md#iolayout).

## Position control law

**Current:** interpolate controller q/alpha; add the selected position residual;
subtract encoder bias; project to hardware bounds. Non-residual joints track
controller output and use the real reference-order PD gains.

**Re-measure if:** encoder or gain conventions change.

**History:** [position rationale](coupling-history.md#position-control-law).

## Torque control law

**Current:** interpolate q/alpha/tau. A joint with exactly zero controller torque
uses PD tracking of biased encoders; other joints use controller torque. Add the
selected torque residual and apply safety projection. Real PD gains are copied
before entity gains are zeroed. ResidualMPC and feedback variants retain their
existing blending, authority and actuator-subset logic.

**Re-measure if:** torque fallback, actuator gains or blending changes.

**History:** [torque rationale](coupling-history.md#torque-control-law).
