# Controller timing modulation

## callback_kind

**Current:** `scripts/inspect_controller_datastore.py` builds one disposable
controller and classifies datastore entries from their reported C++ types. It
does not invoke callbacks. The installed `LogisticController_ismpc` publishes
31 entries under `ismpc_walking::`, including 10 one-argument setters.

| setter | paired getter | transient-screen status |
| --- | --- | --- |
| `set_ref_vel` | `get_ref_vel` | safe; screened separately |
| `set_ts` | `get_ts_target` | safe numeric pair; screened, not promoted |
| `set_tds` | `get_tds` | excluded; ignored while ratio mode is enabled |
| `set_com_height` | none | excluded; reconfigures the stabilizer |
| `set_torso_pitch` | none | excluded; no exact restoration value |
| `set_ref_pose` | none | excluded; also leaves velocity-control mode |
| `set_disturbance` | none | excluded; controller mode switch |
| `set_n_step` | none | excluded; planner state |
| `tds_by_ratio` | none | excluded; no getter for restoring the mode |
| `configure` | `get_config` | excluded; configuration object and side effects |

Commands such as `start/stop`, support-foot switching, and arm-swing toggles
are state transitions rather than bounded residual channels and are excluded.

**Re-measure if:** the installed controller library, selected controller, or
datastore binding changes.

**History:**
- 2026-08-25 — enumerated 43 total datastore entries and 31 walking entries
  from the live installed controller without invoking a setter.

## datastore_scalar_commands

**Current:** the shared-memory transport accepts paired scalar getter/setters.
Each active command is a delta from a baseline captured inside its worker. The
host checks that the getter is finite and the setter takes `double`, publishes
both the applied value and captured baseline, and restores the baseline when
the command becomes inactive. Reset discards every cached controller handle and
baseline because `MCGlobalController.reset()` rebuilds the controller.

This is generic probe infrastructure. A scalar pair does not change the policy
action space unless a task explicitly promotes it.

**Re-measure if:** scalar callback types, shared-memory layout, or controller
reset semantics change.

**History:**
- 2026-08-25 — the smoke probe changed step duration and measured
  `max_restore_error_s = 0.000000000` without a controller or worker failure.

## controller_timeout_ms

**Current:** one 60000 ms collect timeout covers both plain and reset-bearing
steps. It is a liveness check, not a latency budget: a step that has not
acknowledged by then is treated as dead, and the cost of being generous is only
a slower wedge detection, while the cost of being tight is a needless worker
respawn.

Measured on HRP5P / `LogisticController_ismpc`, zero-residual, 16 envs, idle
box, 300 steps per arm, first 20 steps excluded from the steady-state figures:

| ctrl/worker | step median | step p99 | step max | reset max |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 2.51 ms | 3.22 ms | 4.32 ms | 728 ms |
| 8 | 3.66 ms | 5.48 ms | 6.73 ms | 2812 ms |
| 16 | 8.18 ms | 10.14 ms | 10.33 ms | 5588 ms |

The two scale ~800x apart per controller: **0.41 ms** per controller for a step
against **347 ms** for a reset. Resets do not arrive as `Command::Reset` — they
ride inside a Step through the row's reset flag, which
`controllers_host.cpp` reads from shared memory — so a single timeout has to
cover the reset, and 60000 ms is sized for that. Extrapolated, a reset at 136
controllers per worker needs ~47 s.

Splitting the two was considered and rejected: the manager cannot see reset
flags (it never maps the input block), so it would need the action to pass a
per-dispatch hint, and normal steps acknowledge three orders of magnitude
inside the existing budget regardless.

**Re-measure if:** the robot, the controller, or the QP formulation changes, or
controllers per worker rises far above 16.

## probe_step_duration

**Current:** `scripts/probe_step_duration.py` runs all cohorts concurrently from
identical reset states and encoder bias, applies one fixed finite impulse, and
reports two-second command-relative DCM error, hazards, worker failures, applied
timing, and restoration error. It probes the generic scalar channel without
changing any registered task or checkpoint dimension.

An initial seed-42 map used two worlds per cohort, a 0.35 m/s-equivalent
sagittal impulse at 0.20 m height, and a 2 s recovery window:

| requested delta (s) | hazards / worker failures | DCM error (m) | vs zero | mean \|applied\| (s) | restore error (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -0.40 | 0 / 0 | 0.057126 | -0.2% | 0.144661 | 0.000000000 |
| -0.20 | 0 / 0 | 0.050803 | +10.9% | 0.071861 | 0.000000000 |
| 0.00 | 0 / 0 | 0.057004 | baseline | 0.000000 | 0.000000000 |
| +0.20 | 0 / 0 | 0.047379 | +16.9% | 0.089911 | 0.000000000 |
| +0.40 | 0 / 0 | 0.054654 | +4.1% | 0.185250 | 0.000000000 |

The apparent pass did not replicate through the proposed bounded, gated action
path. Its first two-world run measured `-7.312%`, `-1.332%`, baseline, `+3.081%`,
and `+1.948%` at `-0.20`, `-0.10`, `0`, `+0.10`, and `+0.20 s`. Because the
zero arm also shifted from `0.057004` to `0.050205 m`, the two-world result was
too noisy to promote.

The confirmation used eight worlds per arm through the exact proposed action
semantics:

| requested delta (s) | hazards / worker failures | DCM error (m) | vs zero | mean \|applied\| (s) | restore error (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0 / 0 | 0.051069 | baseline | 0.000000 | 0.000000000 |
| +0.10 | 0 / 0 | 0.049922 | +2.247% | 0.045732 | 0.000000000 |
| +0.20 | 0 / 0 | 0.050293 | +1.519% | 0.081434 | 0.000000000 |

Neither value clears the predeclared 5% gate. No timing action or training task
is retained; the generic scalar transport remains available for other paired
callbacks.

**Re-measure if:** gait timing limits, detector calibration, walking speed,
impulse profile, robot, or controller period changes.

**History:**
- 2026-08-25 — rejected the timing action after its eight-world confirmation
  improved DCM error by at most 2.247%, below the 5% gate.
- 2026-08-25 — added after the walking-reference velocity law failed its 5%
  causal promotion gate.
