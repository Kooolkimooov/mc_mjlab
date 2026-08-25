# External controller API boundary

The deployment-side mc_rtc interface implied by the residual policy. This
repository consumes the current Python bindings and does not modify the external
controller library; the items below define the versioned boundary a controller
implementation should expose before hardware deployment.

## recovery_state

**Current:** expose command-relative DCM offset, base angular velocity, gravity
tilt, left/right vertical foot load, calibrated detector score, filtered
authority, burst-active state, and rearm state at the controller period. Values
must carry explicit SI units and a controller monotonic timestamp. Authority is
read-only from the policy's perspective.

**Re-measure if:** detector features, state estimator, force sensors, or control
period changes.

**History:**
- 2026-08-24 — defined from the accepted deployable recovery detector; push
  schedule and time since push are deliberately absent.

## joint_residual_interface

**Current:** accept a vector keyed by stable joint names, control mode
(`position` or `torque`), normalized request, physical request, and sequence
number. Return the requested physical residual, authority-scaled residual,
feasibility-projected residual, final controller target, per-joint projection
flags, and the position/velocity/effort bounds used. Reject unknown joints,
duplicate sequence numbers, stale timestamps, unit mismatches, and controller
mode mismatches atomically.

The interface must apply authority and feasibility inside the real-time
controller boundary; a client-side clamp is useful defense in depth but cannot
be the hardware safety contract. Zero authority must yield bit-exact zero
executed residual.

**Re-measure if:** actuator mode, robot module bounds, residual transform, or
real-time transport changes.

**History:**
- 2026-08-24 — mirrors the repository's requested/executed accounting and
  hardware-bound projection without assuming datastore access in today's
  bindings.

## controller_references

**Current:** publish joint position and velocity references plus planned ZMP,
control CoM, and control CoM velocity under stable names. Each sample carries
the controller step sequence that generated it so the one-period asynchronous
pipeline cannot pair a residual with the wrong reference. Reference validity
and controller failure are explicit status fields, never inferred from a zero
vector.

**Re-measure if:** controller output channels or asynchronous pipeline latency
changes.

**History:**
- 2026-08-24 — records the minimum actor inputs already consumed by this
  repository and the sequence relationship required for deployment parity.

## controller_parameter_modulation

**Current:** the local mc_rtc binding exposes `MCController.datastore()` plus a
generic `DataStore.call()` over zero-argument getters and one-argument setters
whose values are already supported by the Python binding. Unsupported callback
signatures fail with their C++ type instead of being invoked speculatively. The
external source commit is `aadbd7cebd`.

The `Position-Velocity` task appends a recovery-gated `(vx, vy, yaw_rate)`
offset to the position-residual action. It uses
`ismpc_walking::get_ref_vel`/`set_ref_vel`, captures the live nominal command at
activation, slew-limits changes, restores that exact value at zero authority,
and reads the applied value back through the output block. The host resolves the
controller and datastore every step because reset rebuilds both. Missing
callbacks fail during host configuration.

The scalar transport also carries paired `double` callbacks with applied-value
and activation-baseline readback. Step duration was safe to manipulate and
restore but failed its causal promotion gate, so it is available only to the
probe. CoM height and torso pitch remain excluded because neither has a getter
that can restore the live controller state exactly.

**Re-measure if:** the installed walking controller changes its callbacks,
nominal velocity, recovery detector, or control period.

**History:**
- 2026-08-25 — an eight-world-per-arm confirmation improved two-second recovery
  DCM error only 2.247% at `+0.10 s` and 1.519% at `+0.20 s`; both were below
  the 5% gate, so the provisional timing task was removed.
- 2026-08-25 — implemented and live-tested generic getter/setter invocation,
  gated shared-memory command transport, exact nominal restoration, applied
  readback, and the registered `Position-Velocity` task.
- 2026-08-25 — inspected the installed controller and bindings. The controller
  has the required runtime callbacks, but type erasure at the Python boundary
  blocks an in-repository prototype without an external binding change.

## compatibility

**Current:** negotiate a semantic API version and robot-module digest before
enabling nonzero authority. The digest covers joint order, position/velocity/
effort limits, PD gains, control mode, detector calibration, and residual scales.
On mismatch, missing data, stale data, or controller failure, latch authority to
zero and require an explicit healthy rearm interval.

**Re-measure if:** checkpoint provenance or controller configuration hashing
changes.

**History:**
- 2026-08-24 — extends the checkpoint's existing base-controller provenance
  check to the live deployment handshake; implementation remains outside this
  repository.
