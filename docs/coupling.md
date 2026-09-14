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

## ControllerIoBinding

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
block. Native mapping and geometry are covered by `verify_native_action_contracts`.

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

## DatastoreCommands

**Current:** public scalar/vector accessors retain their names; configuration maps
aliases to native callback names. Default vector aliases are `planned_zmp` →
`mc_mjlab::planned_zmp`, `control_com` → `mc_mjlab::control_com`,
`control_com_vel` → `mc_mjlab::control_com_vel`, and `walking_ref_vel` →
`ismpc_walking::get_ref_vel`. Both `support_foot` and the historical public
`ismpc_walking::support_foot_name` alias select numeric `mc_mjlab::support_foot`
(right=0, left=1). Unknown names are treated as explicit callback names and must
validate successfully in native initialization. Missing callbacks are errors.

Setters require one usage flag per callback per environment, addressed by
`use_datastore_scalar_offset()` and `use_datastore_vector3_offset()`. Missing
methods raise a clear prerequisite error before workers start. Python captures
the latest collected getter value on activation and sends absolute
baseline-plus-residual values. Deactivation restores that baseline once, then
clears usage. Each callback is independently gated; getters remain populated
while setters are inactive. After reset, relative commands wait for fresh getter
output from the first step and apply in the following control period. Explicit
absolute commands may apply immediately. Held scalar offsets survive resets.

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
controller; native usage flags remain a prerequisite for the setter path.

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
