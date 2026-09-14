# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

# What this is

mc_rtc controller integration for mjlab: an mjlab action term that steps one
mc_rtc whole-body controller per environment and adds an RL residual on top,
plus the robot assets (HRP5P, JVRC1, RHPS1) it drives. The coupling replicates
mc_mujoco's fidelity (real PD gains, force/IMU sensor feeds, substep target
interpolation), so mc_mujoco is the reference when behavior is in question.
The one deliberate deviation: controller steps are dispatched asynchronously
and collected one control period later, so targets lag their source state by
one period in exchange for overlapping the solve with the GPU sim.

# Environment prerequisites

- The mc_rtc Python bindings and controller libraries come from the sourced
  ROS workspace (`PYTHONPATH`/`LD_LIBRARY_PATH`). Run from a shell with it
  sourced; a missing workspace fails at import.
- The bindings are interpreter-specific. `requires-python` pins the matching
  interpreter and moves with whichever one the workspace builds for; a
  mismatch fails at import or segfaults.
- Robot assets (MJCF, meshes, PD gains) are not tracked: they symlink on
  first use from `$HOME/workspace/install/share/mc_mujoco/<ROBOT>` (see
  `robots/mc_mujoco_assets.py`).
- `~/.config/mc_rtc/mc_rtc.yaml` is merged into every controller's config
  before this repo's `etc/mc_rtc.yaml`, so its own `Enabled:` entry is
  overridden — but anything it sets that the repo file does not will apply
  silently. Check it before blaming the repo config.
- Residual-balance checkpoints embed the project/user mc_rtc YAML, the selected
  installed controller YAML, and PD gains. The custom runner rejects a mismatch;
  do not bypass that check and call the result an evaluation of the same policy.
- The mjlab dependency source is a per-machine choice (README "mjlab
  dependency"): PyPI by default, or an editable `../mjlab` checkout via a
  `[tool.uv.sources]` block that must NOT be committed. `uv.lock` is
  untracked for the same reason. Note uv ignores upper bounds on
  dependencies' `Requires-Python`, so the PyPI release resolves even when
  its cap excludes this project's interpreter.

# Commands

Always use `uv run`, never plain python.

Launch long-running training inside tmux. Reuse an existing tmux session/window
when one is available; create a named session only when none exists. Do not leave
a training process owned only by an agent exec session.

```sh
uv sync                                          # after choosing the mjlab source
scripts/demos/run_test_mc_rtc.sh                 # viser viewer (1 env)
uv run list-envs                                 # task ids (ours + mjlab's)
# Ids are Mc-Mjlab-<Residual-Balance|Zero-Residual>-<Enabled>-<MainRobot>-<Position|Torque>,
# built by utils/task_naming.py, which reads Enabled/MainRobot from
# etc/mc_rtc.yaml and then `.title().replace("_", "-")`s the whole string -- so
# LogisticController_ismpc/HRP5P become Logisticcontroller-Ismpc/Hrp5P, not the
# spelling in the yaml. Never hand-assemble one: run list-envs.
uv run train Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position
uv run play  Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position \
  --checkpoint-file <model.pt>
# Whether a checkpoint actually beat mc_rtc -- the training curves cannot say,
# see the Episode_Reward gotcha below. Both default --num-workers low so they
# can run beside a training job.
uv run python scripts/compare_to_baseline.py --checkpoint <model.pt>
uv run python scripts/probe_residual_authority.py --level 1.0
# The DCM objective's own gate: standing must not outscore walking (~1 min/regime).
uv run python scripts/validate_dcm_objective.py
uv run python scripts/verify_improvement_contracts.py
# Regenerate docs/architecture/ from the source; --check fails on drift.
uv run python scripts/generate_architecture_docs.py
uv run ruff format && uv run ruff check --fix    # format + lint
uv run ty check                                  # type check (56 pre-existing
                                                 # diagnostics: unresolvable
                                                 # mc_rtc bindings + mujoco stubs)
python3 .codex/hooks/check_prose.py src scripts # prose budget + docs/ links
```

The deterministic improvement-contract suite does not replace a live controller
check; the demo is the simulation verification. A healthy run holds a steady
root height (HRP5P z≈0.79, JVRC1 z≈0.83, RHPS1 z≈0.84) — a dropping z means the
robot is falling.

For a *walking* controller (`LogisticController_ismpc`), height alone is not
enough: a robot standing still holds a perfect z. Check that it is walking, by
base displacement (the reliable check: ~0.88 m over 12 s at
`targetCmdVel: [0.1, 0, 0]`). Under the native path `action_term.controller_reference("alpha")`
reads ~0 -- the canonical output robot's `mbc.alpha` is not populated the way the
pre-migration host reported it, so the old "≈0.4 rad/s median" heuristic no
longer holds; use displacement. The installed config now walks indefinitely:
`Logistic::FSMMoveBoxTableToLeftShelf` begins with `Walking::WalkCmdVelImpl`
(`targetCmdVel: [0.1, 0, 0]`, `timeout: 1000.0`) and the 1 m `Logistic::GoToTable`
path is commented out. That override lives in the *installed workspace* file
(`~/workspace/install/lib/mc_controller/etc/LogisticController_ismpc.yaml`), not
in this repo — a workspace rebuild can revert it, and the top-level
`transitions:` map does not show it either way, since the walk is inside that
Meta state's own transitions. `tasks/residual_balance` explains why the task's
episode length is what it is given an unbounded walk.

# Comments, docstrings and notes

Two hard rules, enforced by `.codex/hooks/check_prose.py` (which also runs as a
PostToolUse hook, so a violation comes back in the same turn):

1. **Every docstring is one line.** What the thing does, never the evidence for
   it. No exceptions — the checker errors on a second line.
2. **Comments stay under 10% of a file.** Docstrings are not counted against
   this; one line per definition is already the bound on them.

A comment earns its place in the code only if someone editing **that line**
would break something without it: a hazard, an ordering requirement, a unit, a
non-obvious invariant. Two lines, three at most.

Everything else — measurements, tuning history, runs that failed, alternatives
considered, ablation tables — goes to `docs/`, under a `##` heading that **is**
the identifier it concerns, so `grep -rn PUSH_VELOCITY docs/` finds it. Leave a
one-line comment behind that shares the terse reason with the link:

```python
# Difficulty dial; the baseline should almost always fail. docs/difficulty.md#push-velocity
PUSH_VELOCITY = 0.4
```

The link is a convenience, not the mechanism — grep by identifier is. So **skip
the link where the docs heading is already the function's own name** (`grep`
finds `zmp_tracking` either way); spend the line only where the connection is
not guessable.

If a file cannot meet 10%, it is too big or doing too many jobs — split it
rather than shaving the notes. That is why the PPO config sits in its own module
beside the env cfg.

Where each kind of writing lives:

| Where | Holds |
| --- | --- |
| `README.md` | how to use the repo |
| `AGENTS.md` | how to work in it; hazards needed *before* touching anything |
| `docs/` | why a specific number or design is what it is |
| memory dir | facts about the user, machine and workflow -- not repo facts |

`docs/README.md` carries the index and the section template
(`**Current:**` / `**Re-measure if:**` / `**History:**`). Keep measurements
verbatim when moving them: a paraphrase that drops the sample size is worth much
less than the original.

# Commit messages

Keep them concise: a subject line plus a 2-4 line body carrying the one number
or reason the diff does not show. Everything longer belongs in `docs/` under a
grep-able `##` heading.

**Never** put a `Codex-Session:` line, a session id, a `Codex.ai` URL, or a
"Generated with Codex" line in a commit message — they outlive the session
and stay in `git log` forever. Co-author trailers are wanted and stay. For Codex,
use `Co-Authored-By: Codex MODEL <noreply@openai.com>`, replacing `MODEL` with
the most precise current model identity available in the session, including its
variant or exact model ID when known. Never copy a model identity from an earlier
commit or assume a fixed model from these instructions. If the exact identity is
unavailable, use only what is known; do not invent a version or variant.
This holds for messages carried through a history rewrite too: strip session
lines and correct inaccurate attribution for the work being amended, while
preserving other contributors' trailers.

# Architecture

`docs/architecture/` is **generated from the source** and carries no
hand-written sentences: import graph, class hierarchy, the shared-memory
column map, the control step's real ordering, and one `task-*` page per
sub-package of `tasks/` with its manager terms, weights and guards. A new
task package grows its own page with no wiring. Never edit those files; edit
`scripts/generate_architecture_docs.py` and rerun it. `--check` fails when
they no longer match the code. Explanation belongs in the notes below and in
`docs/`, which the generated pages link to from the source's own comments.

From mjlab down to mc_rtc:

- `actions/mc_rtc_residual_action.py` — `McRtcResidualActionBase(Cfg)`, an
  mjlab `BaseAction`: per-substep interpolation of controller targets across
  `frameskip` (mc_mujoco parity) and the one-period-behind dispatch pipeline.
  The RL residual applies only to `residual_actuator_names`; other joints
  track raw mc_rtc output. Subclasses pick the controller output channels and
  how they reach the actuators:
  `mc_rtc_residual_joint_position_actions.py` →
  `McRtcResidualJointPositionAction(Cfg)` (channels `q`/`alpha` → position +
  velocity targets, residual on position); and
  `mc_rtc_residual_joint_torque_actions.py` →
  `McRtcResidualJointTorqueAction(Cfg)` (adds channel `tau` → effort targets,
  residual on torque).
- `mc_mjlab/controller_io.py` — simulation-side reference-order scatter/gather,
  biased encoders, measured effort, local root coordinates, wxyz-to-xyzw
  conversion and named sensors. Use native layout offset methods throughout.
- `mc_mjlab/controller_datastore.py` — numeric output aliases and independently
  gated setters. Relative commands capture collected baselines, restore once on
  deactivation, and wait one control period after reset for fresh getters.
- `mc_rtc_interface/cpp/` — native `ControllersManager`, worker, `ControllersHost`
  and `ControllerInstance`. The action owns the manager and two shared-memory
  blocks; close the manager before unlinking memory, including startup failure.
  `collect()` returns failed rows, so the action tracks pending dispatch itself.
  Reset flags are separate from episode failure latches. The action dispatches
  whole batches; environment decimation must be divisible by `frameskip`.
- `mc_rtc_interface/hpp/io_layout.hpp` and `ipc_socket.hpp` define the layout and
  protocol. Root input is ten values (position, xyzw quaternion, linear velocity);
  every body sensor, including FloatingBase, has its own gyro/acceleration slot.
  Public `alpha` maps to native `qd`. Python retains `utils/shared_memory.py`.
- Native worker recovery kills and reaps a failed generation during collection,
  then starts its replacement from the episode reset on a fresh endpoint. Its
  rows truncate, then the next reset-bearing step initializes the bound
  replacement. Missing usage methods and callbacks must raise. The numeric
  adapter is no longer missing: `mc_rtc_interface/cpp/instance_datastore_plugin.cpp` provides
  every `mc_mjlab::`-prefixed layout entry from a function of the same name,
  registered on the controller's datastore at each build. Never replace its
  control-centroid ZMP or return zero. See `docs/coupling.md` for the supported
  numeric callback contract and the plugin's prefix rule.
- `tasks/` — follows mjlab's own task layout, which is why this repo ships no
  train/play scripts: mjlab's console scripts drive it and tyro generates the
  `--env.*` / `--agent.*` overrides from the cfg dataclasses. `tasks/__init__.py`
  walks its sub-packages (`import_packages`, as `mjlab/tasks/__init__.py` does)
  and each `<task>/__init__.py` calls `register_mjlab_task` at module level;
  mjlab reaches it via the `mjlab.tasks` entry point. Builders take
  `play: bool` and return the play variant, per mjlab. Two gotchas: only
  sub-*packages* are walked, so a task added as a bare module never registers;
  and `register_mjlab_task` takes built cfgs, so `import mjlab` now builds this
  repo's env cfgs — without a sourced mc_rtc workspace mjlab's loader reports
  that as a `[WARN]` plus traceback rather than failing. Only six supported ids
  register by default; `MC_MJLAB_REGISTER_ARCHIVED_TASKS=1` restores ten
  historical residual ablations for old-checkpoint compatibility.
- `tasks/residual_balance/residual_balance_runner.py` — snapshots external base
  controller inputs into the run directory and every checkpoint, and validates
  them on load. Its effective-training manifest records live resolved manager
  terms, callable defaults and source hashes: full resumes enforce the semantic
  training contract and immediately recompute curricula after restoring the
  global counter, while actor-only loads enforce the narrower observation/action
  interface. Position and torque registrations use distinct full task ids as
  experiment names, so automatic resume cannot cross control modes.
- `robots/<ROBOT>/<robot>_constants.py` — per-robot constants: spec loading
  (collisions disabled by default, geom groups 2=visual/3=collision/4=sites),
  actuator configs, stance initial state, PD-gains path. The three are
  parallel by construction; each is thin, delegating to the shared
  `robots/*_configuration.py` helpers below, and differing only in the
  robot-specific names (root body, foot bodies, deactivated joints).
- `robots/*.py` — the shared machinery those constants files call, one
  concern per module: `mc_rtc_robot_configuration` (joint order, stance, base
  pose and torque limits read lazily from the mc_rtc `RobotModule`, so nothing
  is hand-transcribed), `collision_configuration` (geom naming + the
  `CollisionCfg` presets), `pd_actuator_configuration` (gains from the MJCF's
  armature), `additional_sensors_configuration` (the RL-only sole velocimeters
  and root angular-momentum sensor), `mc_mujoco_assets` (first-use symlinks),
  and `robots_registry` (`MainRobot` → `RobotSpec`, plus `prepare_cfg_for_mc_rtc`).
  `etc/mc_rtc.yaml`'s `MainRobot` is the single source of truth for which robot
  runs: the demo reads it and loads the matching mjlab entity, and the host
  raises if the entity's joints don't exist on the controller's robot.

Cross-cutting invariants:

- mc_rtc vectors are indexed by the robot module's `ref_joint_order()`
  (may include joints mjlab does not simulate or actuate); Python scatters/gathers
  to/from it, filling unsimulated slots with the default stance.
- `PDgains_sim.dat` (one `kp kd` row per refJointOrder joint) overwrites the
  actuator configs' armature-derived default gains at action-term init.
  Without the real gains a walking controller falls. The torque action then
  copies those gains out and zeroes the actuators' (`read_pd_gains` /
  `zero_pd_gains`), since it applies the PD fallback itself.
- mc_rtc only fills `mbc.jointTorque` for robots whose solver has a
  `DynamicsConstraint`; a kinematics-only controller leaves it zero. That is
  why the torque action keeps mc_mujoco's per-joint `tau != 0` fallback to PD
  — without it those joints would go limp.
- The robot XMLs' collision geoms are unnamed, so mjlab's name-based collision
  presets would match nothing. Each robot's `get_spec` therefore names them
  (`collision_configuration`) before disabling them by group, and ships
  presets; `RobotSpec.names_collision_geoms` records that it did, and
  `prepare_cfg_for_mc_rtc` keeps the presets. A robot that has *not* named its
  geoms falls back to enabling group 3 wholesale — the presets cannot be left
  in place there, since a preset matching nothing drops the robot through the
  floor. The flag cannot be inferred from a non-empty `EntityCfg.collisions`
  for exactly that reason. `prepare_cfg_for_mc_rtc` also always deletes the XMLs'
  own motors (mjlab adds its own).
- The stabilizer is a force-feedback loop: foot/hand wrenches and the IMU
  must be fed (automatic when the model has sensors named
  `<ForceSensor>_fsensor`/`_tsensor`, `<BodySensor>_gyro`/`_accelerometer`).

# Gotchas

- mc_rtc's ROS plugin must not autoload into the controller-hosting
  processes: its background threads corrupt the heap (reproducible: 15/15
  processes SIGSEGV/SIGABRT at teardown, `munmap_chunk(): invalid pointer`,
  and a week of kernel-log segfaults at ip ending 0x82d across python3 and
  mc_mujoco). Autoload is disabled machine-side by removing
  `$HOME/workspace/install/lib/mc_plugins/autoload/` (README "ROS plugin");
  a workspace rebuild can restore it, so if workers start dying again check
  that dir first. `ROS.so` still being *mapped* is fine (the loader dlopens
  every plugin-path .so during discovery); only initialization spawns the
  corrupting threads.
- mc_rtc can wedge *permanently inside* `reset()` of a controller whose MPC
  has collapsed (observed in training: worker unresponsive, no output, no
  crash), and `run()` can keep returning True after `[error] MPC result is
  too far from stability condition, stopping` — so neither "run() returned
  false" nor "reset() returns" can be relied on for a fallen robot. Native
  timeout and manager respawning are the required containment.
- `Robot.jointIndexByName` on a missing joint throws a C++ `std::out_of_range`
  that terminates the process uncatchably — always probe `hasJoint` first
  (the host's `joint_index` helper does).
- `MCGlobalController.robot()` is *not* the control robot: it is
  `MAKE_ROBOTS_ACCESSOR(robot, outputRobot)`, the canonical output robot that
  `RobotConverter` fills by copying `q`/`alpha`/`alphaD`/`jointTorque` across
  and nothing else. It never gets `forwardVelocity`/`forwardAcceleration`, so
  its `comVelocity`/`comAcceleration` (and `bodyVelB`/`bodyAccB`) read exactly
  **zero** — silently wrong rather than absent. That is right for the joint
  channels (canonical = what you send to the actuators, as mc_mujoco does), and
  wrong for anything dynamic: The external adapter callbacks must take
  `controller().robot()` instead, the robot the QP integrates.
- `MCGlobalController::reset()` does `controllers.erase(...)` + `AddController(...)`:
  it **destroys and rebuilds** the controller. No handle into it survives an env
  reset, so never cache the `MCController` from `controller()` (or a `Robot`
  from it) across steps — the worker segfaults one reset later, far from the
  cache. Re-resolve every step; it costs a wrapper allocation.
- The locally patched bindings expose `MCController.datastore()` and generic
  `DataStore.call()` for zero-argument getters and one-argument setters over
  the binding's supported scalar/vector/spatial types. Callback lookup is
  runtime-checked and must be re-resolved after reset like every controller
  handle. `mdp.zmp_tracking` deliberately keeps the control-centroid plan for
  checkpoint compatibility: it is the QP-commanded ZMP, while ismpc's reachable
  `zmp_target` differs by delay compensation before the stabilizer builds its
  CoM-acceleration target.
- mc_rtc terminal output is C++ spdlog. Native per-row log flags implement
  `console_output="none"`, `"single"` (environment zero), or `"all"`; play uses
  `"single"`. `controller_timeout_ms` defaults to 60000. Native workers own
  fd redirection; Python no longer redirects or captures worker output.
- The residual itself is printed during `play`: the action cfg's
  `print_residual_every` (residual_balance's play variant sets 10, i.e. 5 Hz)
  writes one `[residual]` line per interval with env 0's per-joint residual in
  rad or Nm, `*` marking a joint at its clip, and the vector norm last. The
  print override is `MC_MJLAB_PRINT_RESIDUAL=<n>` retunes the interval,
  0 silences it. The viewers cannot show this themselves: they only surface
  *reward* and *metrics* manager terms, never actions.
- A controller *worker* dying is not a controller failure: the status column
  carries three values, and `controller_worker_failed` is configured
  `time_out=True` so those episodes truncate and bootstrap instead of paying the
  `-200` termination penalty for the pool's bad luck. Dropping that flag silently
  teaches the critic that infrastructure noise is a fall.
  docs/coupling.md#worker-failure-is-a-truncation
- `fell_over` and `collapsed` are mutually exclusive labels (tilt wins), so their
  shares add up; their *union* is unchanged, and hazard is still computed from the
  union rather than by summing the two.
- Neither logged family of curves means what it looks like. Every
  `Episode_Reward/*` is an episode *sum*, and those correlate with episode
  length at r = +0.98 — they move when the robot survives longer, not when it
  tracks better. `Episode_Metrics/zmp_error` is the length-independent answer,
  in metres, but it is a `sum / step_count` average over *every* step and a step
  with the feet unloaded contributes 0 m (no centre of pressure to place), so on
  its own it *falls* when the robot spends more time off the ground. Read it as
  `zmp_error / zmp_grounded`; that companion metric exists to be the denominator
  (`MetricsTermCfg.reduce` offers no masked mean).
- `[tool.ruff] target-version` is pinned one interpreter below
  `requires-python` on purpose: otherwise ruff rewrites `except (A, B):`
  into PEP 758 syntax that older interpreters cannot parse. Keep the pin.
- Style: 2-space indent (ruff `indent-width = 2`), 88-column lines.
