# The mc_rtc coupling

How one mc_rtc whole-body controller per environment gets stepped from a batched
GPU sim. Code in `src/mc_mjlab/actions/`: the action term owns the control law,
`ControllerPool` the transport, `ControllerIoBinding` the sim-side wiring, and
`ControllerHost` the worker-side mc_rtc calls.

The coupling replicates mc_mujoco's fidelity — real PD gains, force/IMU sensor
feeds, substep target interpolation — so mc_mujoco is the reference when
behaviour is in question.

## Dispatch and interpolation

The one deliberate deviation from mc_mujoco: controller steps are dispatched
asynchronously and collected one control period later, so targets lag their
source state by one period in exchange for overlapping the solve with the GPU
sim. One async step is outstanding at a time — `dispatch_controller_step` sends
without blocking and `collect` awaits it, so each worker holds at most one
command.

Freshly collected outputs are promoted to `next` (`previous <- next`,
`next <- staged`) at each env's period start, keeping the ramp continuous one
period behind. Envs without a collected output yet — startup, or just reset —
hold their seeded value.

On reset, a step may still be in flight from the last `apply_actions`. It has to
be drained before the I/O binding overwrites the input block or the pool sends
reset commands, because the workers must be done reading it. Outputs for envs
*not* being reset are staged in `staged_control` and still applied at their next
period start.

Rates: the sim runs at 1 kHz and the controller at 500 Hz (`frameskip=2`, the
mc_mujoco pairing), while the policy acts at 50 Hz (`decimation=20`); the residual
is therefore held across 10 controller periods.

## Reset pose seeding

**This is load-bearing and it is invisible when it breaks** — it shows up as
nothing but a survival rate.

`MCGlobalController`'s attitude-taking `init`/`reset` overloads iterate
`controller().robots()` — the *control* robots — and never touch `realRobot()`.
The real robot therefore keeps the pose the `MCController` constructor gave it,
which is the controller config's `init_pos`. That is the robot the observer
pipeline estimates on and the stabilizer feeds back from, so an episode that
starts anywhere else begins with its state estimate in the wrong frame.

Measured on HRP5P before `_seed_real_robot` existed: with the sim's reset yaw
drawn over +/-pi against a config assuming 0, **39% of episodes died 4-7 s in,
before any disturbance**, with the controller chasing a motion it had not
commanded (measured CoM speed 0.91 m/s against a 0.1 m/s walk target). Failure
rose with the disagreement — 0% at 0.05 and 0.75 rad, 12.9% at 1.55, 81% at 3.05
— and setting the config's heading to pi inverted it exactly.

Yaw is what makes it bite: `KinematicInertial` takes attitude from the
accelerometer, which observes gravity and therefore roll and pitch but *not*
heading, so a wrong initial yaw is never corrected.

**`reset_base`'s `pose_range` depends on this working.** It was emptied while the
bug was live, and re-enabled (`x`/`y` +/-0.1 m, `yaw` over +/-pi) once the
teleport reconciled the frames per episode. If the seeding ever silently
degrades — a binding without `realRobot()`, a workspace rebuild — the
randomisation turns straight back into the 39% failure, and the only symptom is
a survival rate. That is what `_warn_seeding_unavailable` exists to shout about,
and why re-enabling was checked against a measured baseline rather than assumed.

`velocity_range` stays empty regardless: `reset()` takes encoders and a pose but
no velocity, so the controller would start believing something false.

### _seed_real_robot

Base pose only, deliberately. Copying the joints as well (`mbc.q` plus forward
kinematics, on the theory that the observers build their anchor frame from foot
poses) was tried and measured: it did not reduce the near-pi failure rate, so it
is not carried for a hypothesis the data declined to support.

Two steps, and measured to need exactly these two. Ablated at |yaw| >= 1.57,
where the bug used to kill 54-64% of episodes:

| Treatment | Failed |
| --- | --- |
| `posW` alone | 13/38 |
| `posW` + observer reset | 20/41 |
| `MCController::reset` alone | 24/40 |
| **`posW` + `MCController::reset`** | **0/27** |

Zeroing `velW`/`accW`, moving the control robot, and resetting the observer
pipeline all turned out to be unnecessary — the control robot is already placed
correctly by `MCGlobalController::reset`, and the observers are reset by the
controller reset.

The controller reset is re-run *after* the teleport because
`MCGlobalController::reset()` already called it once from inside
`initController`, i.e. **before** the caller can place `realRobot()`, so
`Walking_controller::reset()` re-derived its world references and reset the
stabilizer task against the estimate's stale pose. Move first, rebuild second, as
the BaselineWalkingController demo's teleport does. `fsm::Controller::reset`
guards `startIdleState()` behind a one-shot `first_reset_`, so the re-entry
re-seeds the controller without restarting the FSM.

Seeding only, never per step: this writes an estimate the observers own, and
doing it every step would hand the controller ground truth and quietly delete the
state estimation this coupling exists to reproduce.

### The warnings, and where they go

`_warn_seeding_unavailable` fires on every degraded path — the two early returns
and the missing-`ControllerResetData` case that warns and carries on — because
degrading silently is the whole problem.

`_warn_if_pose_not_taken` reads back the **real** robot, the one
`_seed_real_robot` writes. Reading the control robot instead cannot detect
anything: `MCGlobalController::reset` has already placed that one at `pose`
whatever happened, so the comparison is against the write we did not make. What
the read-back does catch is `realRobot()` handing back a copy, where
`posW(pose)` writes to a temporary and the seeding is a silent no-op. Measured on
the current bindings: 9 calls, `dp = 0.000000`, `dyaw = 0.000000` — a live
reference, no false positives.

Both warn once per worker; they are build-level faults, identical for every env.

**Where the warning lands:** worker stderr, which under the default
`console_output="none"` is an fd-level redirect into the capture file. It reaches
a terminal only with `console_output` of "single"/"all", with
`MC_MJLAB_WORKER_LOG_DIR` set, or attached to an error reply. Same as every other
diagnostic the host prints — silencing mc_rtc's C++ spdlog has to be fd-level,
and that takes ours with it.

### _assumed_base_pose

`(x, y, z, yaw)` a freshly built controller places its robot at, read *before*
`init()` so it reflects the controller config rather than anything the simulation
feeds. `posW().rotation()` is SpaceVecAlg's world-to-body matrix, so the robot's
forward axis in world coordinates is its first row — hence
`atan2(E[0, 1], E[0, 0])` for the heading.

`controller().robot()` and **not** `controller.robot()`: the latter is
`outputRobot`, whose base pose never reflects the config's `init_pos` — it reads
as identity whatever the config says. Reading it made this check silently unable
to detect the one disagreement it exists for.

Returns `None` on a binding without the `controller()` accessor. This runs in
`ControllerHost.__init__`, so raising would fail every worker before
`await_ready()` — a hard startup failure in place of a run that merely goes
unseeded (and warns).

## VECTOR_OUTPUTS

Per-env 3-vectors a host can publish; `IoLayout.output_vectors` names a subset
and the action term reads them back by the same name. Unlike the per-joint
channels these are not interpolated across substeps.

Each reader takes the **control** robot (`MCController.robot()`), not
`MCGlobalController.robot()`, which every other read goes through. That one is
the *canonical output* robot (`MAKE_ROBOTS_ACCESSOR(robot, outputRobot)`), which
`RobotConverter` fills by copying `q`/`alpha`/`alphaD`/`jointTorque` across and
nothing else. It never runs `forwardVelocity`/`forwardAcceleration`, so its
`comVelocity`/`comAcceleration` read **exactly zero** — silently wrong rather
than absent. Canonical is right for the joint channels (it is what you send to
the actuators, as mc_mujoco does) and wrong for anything dynamic.

Re-resolved every step and never cached: `MCGlobalController::reset()` erases the
controller and builds a new one, so any handle into it dies at the next env
reset. Caching one segfaults the worker a reset later, far from the cache.

### _planned_zmp

`rbd::computeCentroidalZMP` (RBDyn/src/RBDyn/ZMP.cpp) with the ground as the ZMP
plane, applied to the QP's own solution: `TasksQPSolver` runs
`forwardAcceleration` after every solve, so the control robot's `comAcceleration`
is the commanded one, and inverting the LIPM relation on it recovers the ZMP that
motion implies.

This is the plan as the QP hands it down, not the walking MPC's raw `zmpTarget`.
The two differ by ismpc's ZMP-delay compensation (it builds the stabilizer's
CoM-acceleration target from `admittanceTarget`,
`Walking_controller.cpp:765-793`) and by whatever the QP traded away. `zmpTarget`
itself is reachable only through the controller's datastore, which the mc_rtc
Python bindings do not expose; this one needs nothing beyond stock
`mc_rbdyn.Robot`, and it is arguably the more apt thing to reward — it is what
the controller is asking of the world *now*.

`control_com` must be subtracted from both sides before comparing with the sim's
ZMP: the controller places its plan against the *estimated* state, which drifts
from MuJoCo's ground truth, and that drift would otherwise read as tracking
error. CoM rather than base because the LIPM relation is written on the
CoM-to-ZMP offset. `control_com_vel` needs no such correction — the observers
integrate position, so position is what drifts; velocity is differential.

## ControllerPool

Controllers live in worker processes because construction (~570 ms, ~70 MB each,
serial-only) and Cython marshalling are GIL-bound; only the patched binding's
`run()` releases the GIL. Env count is memory-bound in practice.

**Forkserver**, not spawn or fork. Spawn would pull torch/mjlab into each worker
(~20% slower startup); fork is unsafe. Any non-empty preload keeps the server
from importing `__main__`; it must not pull in numpy or the bindings, since a
forked server must stay single-threaded and numpy's import starts OpenBLAS
threads. An env var makes worker numpy skip its thread pool too — workers do no
BLAS.

**Quarantine, not fatal.** mc_rtc can wedge *permanently inside* `reset()` of a
controller whose MPC has collapsed (observed in training: worker unresponsive, no
output, no crash), and `run()` can keep returning True after
`[error] MPC result is too far from stability condition, stopping` — so neither
"run() returned false" nor "reset() returns" can be relied on for a fallen robot.
A worker that dies or goes unresponsive is killed and respawned, its controllers
rebuilt, and its envs reported failed via the status column so the trainer ends
those episodes and re-inits on reset. **The timeout plus quarantine is the
containment; do not remove it.**

The per-command timeout is a budget: a step is milliseconds and a reset not much
more, so it only fires on a genuinely stuck worker. Construction gets its own,
much larger budget in `await_ready`.

Cleanup is registered early so it runs even if the owner's construction raises;
the lists are captured by reference, covering the shm blocks and any respawned
workers, since revival mutates the list slots in place.

### Controller failure is one episode, not the run

mc_mujoco stops the whole sim when `run()` reports failure. A trainer cannot: the
QP giving up is the normal end of a fall, and it must cost one episode. The host
latches it and lets the trainer terminate the env; the last good outputs stay in
the block for the substeps still to come.

## Console output

mc_rtc's terminal logging is hardwired C++ spdlog, so silencing requires
fd-level redirection, not `sys.stdout` swaps. `console_output` picks "none"
(silence all, the default), "single" (env 0 only, in a dedicated worker) or
"all". Both tasks' play variants override it to "single".

Because fd redirection is process-global, per-env guards cannot run under
threads: "none" silences the whole batch, and "single" is only honoured serially
(the host guards per env in `step_envs`). Workers use a capture file rather than
`/dev/null` so error replies can attach mc_rtc's own error text; `reply_ok`
truncates it to keep it from growing.

`play` exposes no `--env.*` overrides (only `train` does), so the escape hatch for
a run already going is `MC_MJLAB_WORKER_LOG_DIR=<dir>`, which redirects each
worker's output to a file there whatever the cfg says, and enables `faulthandler`.

The residual printout during `play` is flushed deliberately: it is meant to be
read live next to the viewer, and Python block-buffers into a pipe while mc_rtc's
spdlog writes straight to fd 1 — unflushed, the two interleave wrongly or vanish
entirely if the session is killed rather than exited.

## IoLayout

Column layout of the shared input/output blocks, one row per env.

Input row:

```
[0, T)           target-joint positions (encoders)
[T, 2T)          target-joint velocities
[2T, 3T)         target-joint torques (qfrc_actuator)
[3T, 3T+16)      root block; the first 7 are always pos(3) + quat wxyz(4):
                   named routing:    qpos7, qvel6, qacc3
                   singular routing: pos3, quat4, linvel3, omega_body3, accel3
[imu_off, ...)   6 per IMU body sensor: gyro(3), accel(3)
[wrench_off, ..) 6 per force sensor: force(3), torque(3) as MuJoCo reads them
```

Output row: one T-wide block per entry of `output_channels`, in order — the
default `("q", "alpha")` gives q in `[0, T)` and alpha in `[T, 2T)` — followed by
a single status column at `status_off` carrying 1.0 once the controller has
failed, then 3 columns per entry of `output_vectors` from `vector_off`.

### Sensor routing

Named routing (mc_mujoco parity: raw base state to "FloatingBase", IMU readings
to the other body sensors) needs the extended binding's name-keyed setters; the
singular fallback only reaches `bodySensors[0]`.

Encoders are fed **biased**: these are the robot's encoders, so the controller's
own state estimate should carry the same calibration error the policy observes,
rather than being handed ground truth the real one never sees.

The stabilizer is a force-feedback loop, so foot/hand wrenches and the IMU must
be fed. That is automatic when the model has sensors named
`<ForceSensor>_fsensor`/`_tsensor` and `<BodySensor>_gyro`/`_accelerometer`.

## Position control law

mc_rtc's `q` and the position residual live in encoder coordinates. The action
therefore sends `q + residual - encoder_bias` to MuJoCo's position servo, matching
mjlab's standard joint-position action. Without that subtraction a simulated
encoder bias changes what the controller observes but not where the actuator
moves, an unrealistically easy plant that the real controller does not have.

Interpolation seeds from `joint_pos_biased`, not ground-truth position. The first
target after reset is consequently the current physical stance after bias
compensation, with no one-step jump induced by the random calibration error.

## Torque control law

`McRtcResidualJointTorqueAction` reproduces mc_mujoco's `--torque-control` law
(`MjRobot::sendControl`): `q`, `alpha` and `jointTorque` are each interpolated
across `frameskip`, and per joint the interpolated torque drives the actuator
unless it is exactly zero, in which case the joint falls back to PD tracking of
the interpolated `q`/`alpha`.

The fallback is not optional. mc_rtc only fills `mbc.jointTorque` for robots
whose solver has a `DynamicsConstraint`; a kinematics-only controller leaves it
zero, and without the fallback those joints would go limp.

The fallback error is `q_reference - joint_pos_biased`, for the same encoder
semantics as the controller and position action. A nonzero direct torque remains
a torque command and needs no position-bias adjustment.

Because the term computes that whole law itself, it takes over the entity's PD:
the configured gains (`pd_gains_path` when given) are copied out at construction
and then zeroed, leaving mjlab's actuators as pass-through motors fed by
`set_joint_effort_target`.

At reset, position ramps from the current biased encoder stance while velocity
and torque ramp from zero. A zero torque seed puts every joint on the PD fallback
for the first control period, so the robot holds its stance instead of going limp.

## Invariants and traps

- mc_rtc vectors are indexed by the robot module's `ref_joint_order()`, which may
  include joints mjlab does not simulate or actuate; the host expands to and from
  it, filling unsimulated slots with the default stance.
- `PDgains_sim.dat` (one `kp kd` row per refJointOrder joint) overwrites the
  actuator configs' armature-derived default gains at action-term init. **Without
  the real gains a walking controller falls.**
- `Robot.jointIndexByName` on a missing joint throws a C++ `std::out_of_range`
  that terminates the process uncatchably — always probe `hasJoint` first.
- Never cache the `MCController` from `controller()`, or a `Robot` from it, across
  steps. `MCGlobalController::reset()` does `controllers.erase(...)` +
  `AddController(...)`: it destroys and rebuilds the controller, so no handle
  survives an env reset. Re-resolve every step; it costs a wrapper allocation.
- mc_rtc's ROS plugin must not autoload into controller-hosting processes: its
  background threads corrupt the heap. See the README for how autoload is
  disabled machine-side. `ROS.so` merely being *mapped* is fine; only
  initialisation spawns the corrupting threads.
