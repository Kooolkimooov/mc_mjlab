# Historical coupling implementations

These are retained pre-migration notes and measurements. Descriptions labelled
Current below describe their original implementation, not the native action
integration. Use [coupling.md](coupling.md) for current behavior.

# The mc_rtc coupling

How one mc_rtc whole-body controller per environment gets stepped from a batched
GPU sim. Code in `src/mc_mjlab/actions/`: the action term owns the control law,
`ControllerPool` the transport, `SimControllerBridge` the sim-side wiring, and
`ControllerHost` the worker-side mc_rtc calls.

The coupling replicates mc_mujoco's fidelity — real PD gains, force/IMU sensor
feeds, substep target interpolation — so mc_mujoco is the reference when
behaviour is in question.

## ControllerInstance

**Current:** the native instance in `src/mc_rtc_interface/` consumes one CPU
double row and publishes canonical `q`, `qd` (`mbc.alpha`), and `tau` channels
in fixed `q`, `qd`, `tau` order, whether Python consumes them or not.
Missing joints publish zero; unobserved encoder positions retain the robot's
initial stance. A failed solve preserves the last
outputs and latches `QP_FAILED` until reset. An uninitialized instance reports
`WORKER_FAILED`. The instance returns its status to the host; only the host writes
the public status column. Layout metadata is copied once at initialization, so
rebuilding the host's routing maps cannot mutate a late worker's layout.

The root block contains position, an **xyzw** quaternion, and linear velocity
(10 doubles), which feed the required `FloatingBase` sensor. The host checks
for it at construction. `body_sensors` follows the robot module's full list,
including `FloatingBase`; each slot holds gyro then linear acceleration
(six doubles). All sensors use the same loop, with no duplicated root gyro or
acceleration fields. Wrench slots contain force then torque, both negated on
delivery to match mc_mujoco. Initialization and
reset share one pose path: initialize the control robot, seed the real robot,
then reset the controller's plan with its current joint configuration.

`datastore_scalar` and `datastore_vector3` append one or three doubles per name
after the sensor inputs or the output status. Input names select absolute
setters, called before the solve; output names select getters, read afterward.
Supported types are `double` and `Eigen::Vector3d`, passed/returned by value or
const reference. Signatures are checked after initialization and reset, and
callback handles are never retained across controller reconstruction. These
fields contain no activation gates, baseline deltas, or task-specific output
adapters; callers must supply the intended absolute command each step.

Set actuated joints and datastore selections before allocating the rows and
calling `initialize`. Joint offsets are `q_offset()`, `qd_offset()`, and
`tau_offset()` on both layouts; output status follows all three joint blocks. The
native host is still separate from the Python action/pool integration, which
owns interpolation, dispatch timing, and process-failure containment.

**Re-measure if:** sensor conventions, robot conversion, reset ordering, callback
types, or the MuJoCo actuator setup change.

**History:**
- 2026-09-07 — fixed joint output blocks to `q`, `qd`, `tau`, removing channel
  selection. All 25 native tests passed. The same 6,000-step, one-environment
  walking check, consuming only `q` and `qd`, gave root z
  `0.7584888097877681–0.7782477721964856 m`, displacement
  `0.8503900949037692 m`, and median reference-velocity norm
  `0.8241782103324028 rad/s`, with no failed solves.
- 2026-09-07 — moved floating-base gyro and acceleration into the regular
  `body_sensors` slots and reduced the root block to pose and linear velocity.
  All 25 native tests passed. The same 6,000-step, one-environment walking
  check gave root z `0.7584932343928974–0.7782477721964856 m`, displacement
  `0.8505204522775239 m`, and median reference-velocity norm
  `0.8207467160489267 rad/s`, with no failed solves.
- 2026-09-07 — separated `FloatingBase` from `imu_sensors`, removing its six
  unused input doubles and the per-sensor branch. All 25 native tests passed,
  including layout checks for HRP5P, JVRC1 and RHPS1_MuJoCo. Repeating the
  6,000-step, one-environment walking check below gave root z
  `0.7584886017545118–0.7782477721964857 m`, displacement
  `0.8504390710589433 m`, and median reference-velocity norm
  `0.823870095647426 rad/s`, with no failed solves.
- 2026-09-07 — native controller tests cover sensor values, reordered channels,
  untouched stance, callback signatures, latched failure, and repeated resets.
  A separate one-environment HRP5P/LogisticController_ismpc check ran 6,000
  synchronous controller steps with raw mc_mujoco XML, real PD gains and two
  interpolated MuJoCo substeps per solve (12 s simulation). Over the samples
  after 2 s, root z was `0.7584880835166774–0.7783122851585075 m`, planar
  displacement was `0.8502725091058962 m`, and median joint-reference velocity
  norm was `0.8240023537456057 rad/s`. All solves succeeded. This checks the
  instance, not mjlab's asynchronous pipeline or checkpoint equivalence.
  Test configuration excluded global-plugin autoload because the machine's
  ROS marker had reappeared; the installed plugin files were not changed.

## ControllersHost

**Current:** the native host owns one `ControllerInstance` per environment
and runs all instances sequentially on the calling thread. Its constructor is
`ControllersHost(configuration_path, num_controllers)`; there is no worker
count or timeout argument. The Python bindings keep the GIL. Use one calling
thread, preferably the main thread of a future worker process; native callers
must also serialize all access. This host provides no hang or crash containment.

`initialize()`, `reset()` and `step()` operate directly on the bound arrays.
The host publishes per-row statuses. Completed step/reset exceptions replace
the affected instance, preserve its print setting and leave the buffers usable;
reset initializes the replacement. Initialization errors propagate after row
statuses are published. Input and output blocks must not overlap. Set layout
selections before initialization and leave them unchanged during step/reset;
instances snapshot layout metadata at initialization.

The native host has no process supervisor yet. The old Python process pool
under `src/mc_rtc_interface_py/` is retained as reference code, not wired into
this native host. See [the implementation guide](process-workers.md).

**Re-measure if:** process dispatch is added, layout ownership changes, or the
controller dependency set changes.

**History:**
- 2026-09-08 — removed the native thread pool, slice jobs, timeout retirement
  and output-guard epochs. All 38 native Python tests and the standalone C++
  output-guard test passed, including repeated host destruction and rebinding.
  The one-environment HRP5P/LogisticController_ismpc harness completed 6,000
  synchronous steps (12 s simulation) with no failed solves. Across the 4,999
  samples after 2 s, root z was `0.7585712120743928–0.7782477721964857 m`, planar
  displacement was `0.8510976445982837 m`, and median reference-velocity norm
  was `0.8329993376356719 rad/s`. This checks native walking, not process
  recovery, mjlab's asynchronous pipeline or training throughput.
- 2026-09-07 — implemented, and measured on HRP5P with the `Posture` controller,
  16 instances, 300 steps after one warm-up, `OMP_NUM_THREADS=1`, on 32 cores.
  Per controller: `87.8 us` at 1 thread, `45.0` at 2, `23.0` at 4, `11.8` at 8,
  `9.5` at 16 — `9.24x` at 16 threads. All 26 native tests passed. The measured
  controller is `Posture`, not `LogisticController_ismpc`, for the reason below;
  ismpc's solve is roughly 8x heavier (738.3 us/controller,
  `MC_MJLAB_PROFILE_WORKERS`), so this scaling is a floor rather than a
  prediction. Nothing measured here says anything about the trainer's end-to-end
  throughput: the Python action term and pool are not yet on this path.
- 2026-09-07 — `reset()` and then `initialize()` moved onto the pool; the
  latter needed the never-exiting workers described under `WorkerPool`. Every
  timing in this section was taken with the ROS autoload marker directory
  absent.

### num_threads and num_controllers

**Current:** on this 32-core box the step batch stops improving at **16 to 24
threads**, whatever the controller count, and 30 is sometimes slower than 24.
Resident memory is flat at **74-82 MB per controller**, so instance count is
bounded by the control-period deadline long before it is bounded by RAM.

Batch milliseconds per control period, HRP5P / `LogisticController_ismpc`,
300 steps after a warm-up:

| controllers \ threads | 1 | 2 | 4 | 8 | 16 | 24 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 5.728 | 2.752 | 1.381 | 0.804 | **0.509** | 0.539 | 0.541 |
| 32 | 13.480 | 6.725 | 3.512 | 1.989 | **0.991** | 0.960 | 1.021 |
| 64 | 32.647 | 15.948 | 8.415 | 4.186 | 2.378 | **2.020** | 2.125 |
| 128 | 66.576 | 34.139 | 17.873 | 10.065 | 6.338 | **5.038** | 4.913 |
| 256 | 133.955 | 68.400 | 36.084 | 21.281 | 13.773 | **11.878** | 11.316 |

Speedup against one thread peaks at `11.3x` (16 controllers), `14.0x` (32),
`16.2x` (64), `13.5x` (128) and `11.8x` (256): parallel efficiency falls as
controllers are added, from 71% to 39%, so a bigger batch is not a cheaper one
per controller. Per-controller time is flat near `31 us` to 64 controllers and
rises to `46 us` at 256.

**This benchmark's controller is idle, not walking.** Real training profiling
measured `738.3 us` per controller in `run()`
(`MC_MJLAB_PROFILE_WORKERS`), roughly 24x these numbers, because the synthetic
harness gives the robot a static root and no walk command. Take the thread-count
knee and the memory figures as transferable and the absolute instance ceiling as
not: it has to be re-measured against a real walk.

Past the core count there is nothing to gain. Sweeping 16/24/30/32/40/48/64
threads on 32 physical cores, the best thread count is **30** at 256, 384 and
512 controllers, and oversubscribing to 64 buys `+0.0%` there; only the smallest
batch measured (64 controllers) showed anything, `+9.1%`, which is within this
benchmark's run-to-run noise — single 300-step runs vary by 10-20% at small
controller counts, while 256 and above reproduce to about 2%.

Memory per controller is dead flat at **74-76 MB** from 64 to 512 controllers:
`4.74 GB` at 64, `9.35` at 128, `18.55` at 256, `27.72` at 384, `36.91` at 512.
On a 59 GB machine that puts the RAM ceiling near 650 controllers, which the
control-period deadline reaches long before.

**Re-measure if:** the robot, controller, core count, or the control period
changes — and before trusting any instance count, under a walking controller.

**History:**
- 2026-09-07 — a synthetic harness cannot hold the controller in its walking
  state, so the walking-load figure is still owed. Feeding the real refJointOrder
  stance, closed-loop encoder feedback, `+g` on the body sensors and half the
  robot's `104.8 kg` on each foot sensor does drive the FSM through
  `Walking::Initial::ISMPC_` -> `Logistic::Demo` -> `Pause` ->
  `Walking::WalkCmdVelImpl` at about 4 s. It then collapses with `MPC failed,
  stopping` and `MPC result is too far from stability condition`, with or
  without the base carried forward at the commanded `0.1 m/s`, because the feet
  never land where the plan puts them. Measuring walking load needs MuJoCo in
  the loop, which is what produced the existing `738.3 us` figure.
- 2026-09-07 — swept 64/128/256/384/512 controllers against 16/24/30/32/40/48/64
  threads for the thread ceiling and the memory curve. One point (64
  controllers, 24 threads) reported a failed controller.
- 2026-09-07 — swept 16/32/64/128/256 controllers against 1/2/4/8/16/24/30
  threads, `OMP_NUM_THREADS=1`, `nice -n 5`, 32 cores. One point (32
  controllers, 24 threads) reported a failed controller, which the static-root
  input makes likely; every other point was clean. Parallel `initialize` for the
  same sweep ran 1.0 s (16 controllers) to 11.1 s (256) at the knee.

### ismpc aborts the process from its own thread

**Current:** `LogisticController_ismpc` runs `Walking_controller::WalkingTrajectoryLoop`
on a `std::thread` of its own, one per instance, which computes the walking
trajectory off the control thread and parks on a condition variable between
solves. A throw on that thread escapes a thread entry point and calls
`std::terminate`, so the **whole process aborts**. No `catch(...)` on our step
path can see it, and no in-process containment can cover it. Each instance also
carries that thread, so `num_threads` is not the process's thread budget.

The way to trigger it is to disable mc_rtc's plugins.
`ComputeWalkingTrajectory` (`Walking_controller.cpp:316`) calls
`datastore().assign<>("footsteps_planner::input_vel", ...)`, and `assign`
requires the key to exist already; the key is *created* by the FootSteps_Planner
plugin (`FootSteps_Planner/src/plugin.cpp:17`, `datastore().make<>`). A config
carrying `ClearGlobalPluginPath: true` with `Plugins: []` therefore leaves ismpc
assigning into a key nobody made, and the resulting
`[DataStore] No key "footsteps_planner::input_vel"` takes the process down.

So a harness that silences plugins to keep the ROS plugin out must not silence
them wholesale — that also removes the footsteps planner ismpc depends on. With
the ROS autoload marker directory removed machine-side (README "ROS plugin"),
plugins can simply be left alone.

**Re-measure if:** the installed walking controller changes its trajectory
thread, or the footsteps planner moves where it creates its datastore keys.

**History:**
- 2026-09-07 — first read as ismpc needing a walk command it had not been given.
  That was wrong: the trigger is the missing plugin, and enabling plugins runs
  ismpc through a stepped benchmark with no abort.

### Deferred: per-environment step and reset

`step()` and `reset()` act on every instance. The pipeline needs a subset of
both: mjlab resets envs as they terminate, and `run_indices` comes from per-env
`_steps_since_run % frameskip == 0`, which goes sparse as soon as envs reset at
different times. The host will own the mechanics, but the index set stays
Python's to supply — it owns termination and the decimation, and the host can
derive neither.

Both entry points should take a sorted `int64` index array rather than a mask
column in the input block, which would put control flow into the layout.
Threads find their sub-range with `lower_bound`/`upper_bound` over their
instance slice: no allocation, no scan of skipped envs. The caller must pass the
array itself — `mc_rtc_residual_action.py`'s `.tolist()` is a GPU sync plus a
Python list build every control period, and dropping it is a win independent of
threading.

### step_timeout

**Current:** removed from the native host along with the thread pool.
Deadlines must be implemented by a future parent process, covering startup,
initialization, reset and step. A synchronous native call can block indefinitely.

**Re-measure if:** a process supervisor is implemented.

<details>
<summary>Retired thread-backend contract (not the current API)</summary>

**Former thread-backend contract:** `ControllersHost`'s fourth argument is seconds, `0.0` (off) by
default. On timeout the pool cancels outstanding slices and retires their worker
epochs under the dispatch mutex before returning. It removes the old dispatch
without advancing the work generation, so a completed helper cannot interpret
the timeout as new work. Late completions cannot decrement a newer batch's
pending count. Replacement helpers accept subsequent dispatches; retired threads
park forever rather than exit through mc_rtc's dangling TLS destructors.

Every environment in an outstanding slice receives a fresh, uninitialized
instance and `WORKER_FAILED`. It remains unstepped until reset initializes its
replacement. An abandoned worker owns its original `ControllerSlice`, including
the controller handles and row spans, and checks cancellation between instances.
The host never reads that slice's mutable results after timeout. The old slice
is released when its worker unwinds; a permanent wedge retains that slice's
controllers (`~75 MB` each) and one retired thread.

Only the host writes the public status column. A late controller can still write
payload columns in the old output block, but it cannot overwrite `WORKER_FAILED`.
After timeout the caller may read statuses and payloads from completed slices;
it must not read failed-row payloads, copy the whole output block, or modify old
inputs. Both old blocks remain pinned for the host's lifetime. `step()` and
`reset()` refuse until fresh blocks are supplied through `set_blocks()` or
`initialize()`. Both entry points reject overlap with retired storage, including
offset NumPy views, and require non-overlapping input and output blocks.

Completed step/reset exceptions are different: their workers have unwound, so
only the affected instance is replaced and released, and existing blocks remain
usable for subsequent steps and reset. Both recovery paths preserve the
per-environment terminal-print setting. Initialization errors still propagate.
Only a timeout calls `OutputGuard::release_all()`, once for the dispatch.

There is one active host per process, calls are serialized, and the host must
outlive its abandoned workers because their controller configuration still
belongs to it. Configure layout selections before initialization; keep them
unchanged during stepping and reset. Initialization snapshots layout metadata
for each instance. No input or output rows are copied per step; only handles,
spans and small result metadata are prepared for dispatch.

**What it cannot cover:** a controller that aborts the process from a thread of
its own (see "ismpc aborts the process from its own thread"), or a segfault.
Both need a process boundary; no in-process guard reaches them.

</details>

**History:**
- 2026-09-07 — timeout fixes preserve direct I/O: 41 native Python tests and two
  standalone C++ tests passed. The standalone pool and output-guard tests also
  passed with ThreadSanitizer, with no reported races. A one-environment
  HRP5P/LogisticController_ismpc run with `step_timeout=2.0` completed 6,000
  controller steps (12 s simulation), with no failed solves. Across the 4,999
  samples after 2 s, root z was `0.7584897708222923–0.7782477721964854 m`, planar
  displacement was `0.8502215240595052 m`, and median reference-velocity norm was
  `0.8305167007616873 rad/s`. This checks the native synchronous harness, not
  mjlab's asynchronous pipeline or training throughput.
- 2026-09-07 — the `accept`/`publish` copies cost nothing measurable. Widening
  the rows sevenfold at a fixed controller workload (one unmapped joint,
  `in=49 out=4`, against 53, `in=205 out=160`) moved the batch by `-1.0%` at 128
  controllers and `+1.0%` at 512. Against the points measured before the copies
  existed, 24 threads: 64 controllers `2.038 -> 1.969 ms`, 128 `5.018 -> 5.253`,
  256 `11.713 -> 11.943`, 512 `24.818 -> 25.115`. All inside this benchmark's
  noise band.
- 2026-09-07 — dropped the per-instance `accept`/`publish` copies and the
  shared-owned batch in favour of the single-host contract above, replaced the
  quarantine list with a fresh controller in the slot, recovering the environment
  on its next reset, and fixed the pool's width at one worker per core instead of
  growing it per host. Nothing is copied per step any more; the earlier copy
  measurement stands only as evidence that it was not the copies that cost.
- 2026-09-07 — added. A two-controller probe with one instance sleeping 5 s past
  a `0.4 s` timeout returns promptly with status `[OK, WORKER_FAILED]`, and three
  further steps complete in `0.00 s`. Parallel construction still soaks clean at
  6 controllers x 6 threads x 10 host rounds.

## ControllerSlice

**Current:** removed together with `SliceJob`; the synchronous host iterates
its uniquely owned instances directly. There are no abandoned dispatches.

**Re-measure if:** an execution backend is introduced.

**History:**
- 2026-09-07 — separated from host state to contain late exceptions and avoid
  accessing replacement controller handles from a retired worker.

## OutputGuard

**Current:** a synchronous RAII redirect of fds 1 and 2 to `/dev/null`
around silent instance calls. Each guard saves and restores its own descriptors;
lexical nesting and ordinary exception unwinding work without global state.
There are no mutexes, epochs or forced-release operations. Calls must not overlap
across application threads because the descriptors remain process-global.

The host's configuration banner is outside the instance guard. The mechanism
also redirects unrelated output in the same process while active. A future
process supervisor should configure worker logging before host construction.

**Re-measure if:** concurrent calls are permitted or worker logging is added.

**History:**
- 2026-09-07 — added with the per-instance `print_to_terminal` flag, defaulting
  to silent to match the Python side's `console_output` default of "none".
  Policy stays in Python; this is only the mechanism.
- 2026-09-11 — the mechanism does not hold: mc_rtc's loggers are asynchronous,
  so a scoped redirect races the drain and leaks regardless of the flag. The
  "future process supervisor" note above is the fix, and the configuration
  banner is only the second of two leaks. Current note:
  [OutputGuard](process-workers.md#outputguard).

## WorkerPool

**Current:** removed. No host-owned worker threads are created or retained.
The loader observations below explain why future process workers should execute
controller calls on their main thread.

**Former implementation:** one pool per process, sized once by hardware concurrency, and its
threads **never exit** — the instance is allocated once and deliberately leaked,
and the destructor is `= delete`d so nobody can undo that. `initialize()`,
`reset()` and `step()` all run on it.

The threads must not exit, and that is the whole reason this class exists rather
than a pool owned by each host. mc_rtc's loader opens and closes controller
libraries constantly — 47 `dlclose` calls in a two-controller, two-round run —
and something in that set calls `pthread_key_create` on load without a matching
`pthread_key_delete` on unload. Every open/close cycle therefore strands a
pthread key whose destructor address points into a mapping that is about to go
away. Interposing `pthread_key_create` showed five such keys sharing one
destructor offset at five different addresses, one per load, and at crash time
those addresses were unmapped or already reused by another library.

A stranded key only hurts a thread that both set a value for it and then exits,
because that is when glibc walks the key table and calls the dangling
destructor. Serial construction never hit this: the only thread that touches
controller TLS is the main thread, which outlives every unload. A per-host pool
did hit it, on the first host teardown after a concurrent build:

```
#0  0x00007fff9a7a6660 in ??? ()        <- registered by an unloaded library
#1  __GI___nptl_deallocate_tsd at ./nptl/nptl_deallocate_tsd.c:73
#2  start_thread at ./nptl/pthread_create.c:462
```

Workers that never exit never reach `__nptl_deallocate_tsd`, so the stranded
keys stay harmless. This is a containment, not a repair — the keys are still
leaked, and a thread created by anything else that touches them can still crash
on exit. The repair belongs in whichever library registers them.

Pinning libraries with `RTLD_NODELETE` is the other lever and was not usable:
`libqpOASES`, the six libraries registering keys during a controller load, and
the three `lttng-ust` libraries were each pinned in turn and none of them
stopped the crash, because the guilty registration is in a library the trace
could not attribute. Keeping an unrelated host alive for the whole run *does*
work, since it holds the libraries open, but no caller can be asked to
guarantee that.

**Re-measure if:** mc_rtc's loader stops closing libraries between uses, or a
library in the controller's dependency set learns to delete its pthread keys.

**History:**
- 2026-09-07 — parallel `initialize()` on HRP5P/`LogisticController_ismpc`:
  128 controllers went from `105.60 s` serial to `10.17 s` on 16 threads,
  `10.4x`. At 32 controllers, `26.29 s` at 1 thread, `6.54` at 4, `4.13` at 16.
  Resident memory is the cost: 128 controllers took `11.24 GB` on 16 threads
  against `9.26 GB` serial, and 32 took `4.45 GB` against `2.52 GB`. Soaked at
  8 controllers x 8 threads x 15 host create/destroy rounds with no crash, where
  the per-host pool died after the first round.
- 2026-09-07 — diagnosed by interposing `pthread_key_create`, `dlopen` and
  `dlclose` with `LD_PRELOAD` and matching the faulting address against the
  process's mappings at crash time.

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

Measured on HRP5P before `seed_real_robot` existed: with the sim's reset yaw
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

### seed_real_robot

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
`seed_real_robot` writes. Reading the control robot instead cannot detect
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
legacy worker-file capture enabled, or attached to an error reply. Same as every other
diagnostic the host prints — silencing mc_rtc's C++ spdlog has to be fd-level,
and that takes ours with it.

### assumed_base_pose

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

### planned_zmp

`rbd::computeCentroidalZMP` (RBDyn/src/RBDyn/ZMP.cpp) with the ground as the ZMP
plane, applied to the QP's own solution: `TasksQPSolver` runs
`forwardAcceleration` after every solve, so the control robot's `comAcceleration`
is the commanded one, and inverting the LIPM relation on it recovers the ZMP that
motion implies.

This is the plan as the QP hands it down, not the walking MPC's raw `zmpTarget`.
The two differ by ismpc's ZMP-delay compensation (it builds the stabilizer's
CoM-acceleration target from `admittanceTarget`,
`Walking_controller.cpp:765-793`) and by whatever the QP traded away. The local
binding now reaches `zmpTarget` through `DataStore.call()`, but this quantity is
kept for checkpoint compatibility and because it is arguably the more apt thing
to reward — it is what the controller is asking of the world *now*.

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

`ManagerBasedRlEnv.close()` closes only its renderer and recorder in the current
mjlab release; it does not close action terms. A process that constructs several
envs in sequence must therefore call `McRtcResidualActionBase.close()` before
`env.close()`. `ControllerPool.close()` is idempotent, drains its workers and
shared memory, and detaches the fallback finalizer. `await_ready()` also closes
the pool on any startup exception.

Measured on 2026-08-25: the first multi-scenario qualifier omitted that explicit
action close, retaining six generations of 12 workers. Active anonymous memory
reached 51.5 GB on a 59 GB machine, all but 220 KB of 8 GB swap was consumed,
and the kernel invoked the OOM killer at 10:58 and 11:32. The terminal error was
a later 300 s worker-startup timeout caused by the memory pressure, not the root
failure. A two-checkpoint same-process live check after the fix left no
qualification or forkserver worker process behind.

### MC_MJLAB_PROFILE_WORKERS

**Current:** setting `MC_MJLAB_PROFILE_WORKERS=<dir>` accumulates nanosecond
timings inside each worker and writes one `worker-<pid>.json` at shutdown. The
phases are controller input marshalling, `controller.run()`, output extraction,
and the whole multi-environment batch; the file also carries the worst batch.
Profiling is disabled by default and performs no clock reads on that path.

A five-iteration run on 2026-09-04 used 128 environments and 30 workers and
measured 1,638,400 controller steps. Of the summed phase time, input marshalling
was 3.29% (26.5 us/controller), `controller.run()` was 91.66% (738.3
us/controller), and output extraction was 5.04% (40.6 us/controller). Mean batch
time ranged from 3.10 to 3.97 ms across workers; the maximum observed batch was
42.84 ms. The workers with five environments formed the slow tail, while those
with four were near the lower end.

**Re-measure if:** the controller, worker count, controller frequency, robot, or
host marshalling changes.

**History:**
- 2026-09-04 — added after trainer-only profiling could attribute collection
  waits to workers but could not separate Python binding work from the C++ solve.

### Controller failure is one episode, not the run

mc_mujoco stops the whole sim when `run()` reports failure. A trainer cannot: the
QP giving up is the normal end of a fall, and it must cost one episode. The host
latches it and lets the trainer terminate the env; the last good outputs stay in
the block for the substeps still to come.

### Worker failure is a truncation

**Current:** the status column carries three values, not two —
`STATUS_OK`, `STATUS_QP_FAILED` and `STATUS_WORKER_FAILED` (`mc_rtc_interface.io_layout`).
The action term latches them into `controller_failed` and
`controller_worker_failed`, and the task maps them to two termination terms of
which only the second is `time_out=True`.

**Why they must not be one flag.** A dead or wedged worker takes down every env
assigned to it at once, for reasons the policy did not cause and cannot avoid.
Folded into `controller_failed` it was a normal terminal state: the whole batch
paid `termination_penalty` (`-200`) and the value target was cut to the reward,
teaching the critic that some states are worth -200 for reasons not in the
observation. As a truncation the value bootstraps off the final observation and
no penalty is charged, which is the standard treatment of an exogenous time limit
(Pardo et al. 2018, *Time Limits in Reinforcement Learning*).

Both are still counted, separately, as `Episode_Termination/controller_failed`
and `Episode_Termination/controller_worker_failed`. A run where the second is not
~0 is a run whose comparison needs that number quoted beside it: those episodes
are shorter for infrastructure reasons.

Which code writes which value:

| writer | value | when |
| --- | --- | --- |
| `ControllerInstance::step` | `QP_FAILED` | `run()` returned false, and every step after until reset |
| `ControllerInstance::step` | `WORKER_FAILED` | the instance is not initialized yet |
| `McRtcResidualActionBase._collect_controller_output` | `WORKER_FAILED` | the manager returned rows from a failed worker generation |

## ControllersManager

**Current:** Python owns the shared-memory blocks. The manager's constructor
accepts `timeout_ms` (default 5 ms), stored in `m_timeout_ms` and used by
`collect()` for each worker's reply. Zero polls without waiting; negative values
are rejected. Startup keeps its separate timeout. Per-row logging flags travel
in shared input rather than through a manager constructor option.

Dispatch send errors, reply errors, receive failures and timeouts quarantine the
affected worker's row slice. Collection harvests other replies before killing
and reaping failed processes, then returns the failed rows without launching a
replacement. The failed command is not retried. Python truncates those rows and
requests a respawn from the episode-reset callback. Once every row assigned to a
failed worker has reset, its replacement starts on an endpoint carrying a new
generation number and binds its shared spans without initializing controllers or
writing outputs. The next reset-bearing step initializes it. Failed replacement
startup closes the entire manager and raises.

`close()` is idempotent and also runs from the destructor. It drains any pending
operation before sending Stop, waits for the acknowledgment and process exit,
and kills and reaps workers that do not finish. Each reply and process-exit wait
has a 1 s budget; sending Stop has a 100 ms budget. Python can then release its
shared memory. Dispatch and collection after close raise an error.

**Re-measure if:** worker teardown exceeds these budgets or the collection
deadline needs to vary by operation.

**History:** The collection timeout was previously a hardcoded 5 ms; shutdown
only happened through destruction and could mistake an operation reply for Stop.
Native process recovery was added on 2026-09-11; the earlier Python pool's
quarantine behavior remains under `## ControllerPool`.

## Python controller bindings

**Current:** Python interacts with controllers through `ControllersManager`;
`ControllersHost` and `ControllerInstance` are internal C++ worker components.
Python prepares an `IoLayout` with `set_joint_order()` and the input
sensor names and datastore callbacks before construction. Joint inputs follow the
robot module's complete reference order. The binding exposes the current native
layout, including the input reset flag; it no longer exposes actuator-subset maps
or datastore metadata from the old host interface.

`SharedMemoryDescription(file_name, offset, size)` uses byte offsets and sizes.
`WorkerStartMessage(layout, input, output)` copies those descriptions and layout
into `ControllersManager(configuration_path, num_controllers, num_workers,
configuration, timeout_ms=5)`. Blocking manager calls release the GIL.
`dispatch(Command.Initialize/Reset/Step)` and `collect()` preserve the native
command protocol; collection returns failed row indices without writing memory.
The manager supports `with` and explicit `close()`. The Python owner must keep
the shared-memory blocks alive until the manager has closed. Calls on the same
manager must be serialized even though other Python threads can run.

**Re-measure if:** shared-memory ownership or the command protocol changes.

**History:** The old bindings referenced removed layout members and a removed
host constructor, preventing the extension from building.

## DIRECT_JOINTS

**Current:** Python probe tests compare the first 17 HRP5P joints (legs, torso,
head) directly with encoder inputs. They supply the full reference-order layout;
fixed and coupled finger outputs are not identity copies of those inputs.

**Re-measure if:** HRP5P's reference order or the robot converter changes.

**History:** Before the supplied-layout API, the probe tests selected these same
17 joints through the host's actuator-subset mapping.

## build_timeout_ms

**Current:** `max(300 s, 30 s * controllers_per_worker)` — the window
`ControllersManager`'s constructor waits for a worker's startup `Reply`, which the
worker sends only after building all of its `ControllerInstance`s. Mirrors the
Python `ControllerPool.await_ready` budget it replaces. Construction is ~570 ms
per controller and serial within a worker (see `## ControllerPool`), so the
per-controller term dominates above ten controllers and the 300 s floor covers
process start plus mc_rtc's dynamic linking.

**Re-measure if:** controller construction cost changes, the robot module grows,
or workers start hosting many more controllers each.

**History:**
- 2026-08-25 — a 300 s worker-startup timeout fired as a *symptom* of OOM
  pressure, not a genuine build overrun; see the retained-worker measurement in
  `## ControllerPool`. The floor was left unchanged.

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
a run already going is the removed worker-file capture setting, which redirects each
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

`output_channels` takes keys of `MBC_ATTR_BY_CHANNEL` and must match the action
term's own `output_channels`. `output_vectors` takes keys of `VECTOR_OUTPUTS`:
whole-controller 3-vectors, not per-joint, and not interpolated across substeps.
`output_scalars` names read-only datastore getters, one column each.
`wrenches` and `imu` list the force and body sensors in input-block order.
`datastore_vector_commands` and `datastore_scalar_commands` are `(getter, setter)`
pairs for the gated delta commands; each gets an active flag plus its value in the
input block, and `datastore_vector_command_is_absolute` picks, per command, whether
the setter takes an absolute target or a baseline-relative delta.

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
