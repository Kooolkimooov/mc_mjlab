# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
- The mjlab dependency source is a per-machine choice (README "mjlab
  dependency"): PyPI by default, or an editable `../mjlab` checkout via a
  `[tool.uv.sources]` block that must NOT be committed. `uv.lock` is
  untracked for the same reason. Note uv ignores upper bounds on
  dependencies' `Requires-Python`, so the PyPI release resolves even when
  its cap excludes this project's interpreter.

# Commands

Always use `uv run`, never plain python.

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
uv run ruff format && uv run ruff check --fix    # format + lint
uv run ty check                                  # type check (56 pre-existing
                                                 # diagnostics: unresolvable
                                                 # mc_rtc bindings + mujoco stubs)
python3 .claude/hooks/check_prose.py src scripts # prose budget + docs/ links
```

There is no test suite; the demo is the verification. A healthy run holds a
steady root height (HRP5P z≈0.79, JVRC1 z≈0.83, RHPS1 z≈0.84) — a dropping z
means the robot is falling.

For a *walking* controller (`LogisticController_ismpc`), height alone is not
enough: a robot standing still holds a perfect z. Check that it is walking, by
base displacement or by the controller's own joint-velocity reference
(`action_term.controller_reference("alpha")`, ≈0.4 rad/s median while walking
against ≈0.003 standing). The installed config now walks indefinitely:
`Logistic::FSMMoveBoxTableToLeftShelf` begins with `Walking::WalkCmdVelImpl`
(`targetCmdVel: [0.1, 0, 0]`, `timeout: 1000.0`) and the 1 m `Logistic::GoToTable`
path is commented out. That override lives in the *installed workspace* file
(`~/workspace/install/lib/mc_controller/etc/LogisticController_ismpc.yaml`), not
in this repo — a workspace rebuild can revert it, and the top-level
`transitions:` map does not show it either way, since the walk is inside that
Meta state's own transitions. `tasks/residual_balance` explains why the task's
episode length is what it is given an unbounded walk.

# Comments, docstrings and notes

Two hard rules, enforced by `.claude/hooks/check_prose.py` (which also runs as a
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
| `CLAUDE.md` | how to work in it; hazards needed *before* touching anything |
| `docs/` | why a specific number or design is what it is |
| memory dir | facts about the user, machine and workflow -- not repo facts |

`docs/README.md` carries the index and the section template
(`**Current:**` / `**Re-measure if:**` / `**History:**`). Keep measurements
verbatim when moving them: a paraphrase that drops the sample size is worth much
less than the original.

# Architecture

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
- `actions/mc_rtc_controller_pool.py` — `ControllerPool` owns the worker
  processes (forkserver; spawn is slower and fork is unsafe), their pipes and
  two shared-memory blocks for batched I/O (layout in `IoLayout`). A worker
  that dies or wedges mid-run is quarantined, not fatal: killed, respawned
  (controllers rebuilt, envs re-init on their next reset) and its envs
  reported failed via the status column.
- `actions/mc_rtc_controller_io_binding.py` — `ControllerIoBinding` resolves
  model addresses and sensor routing at init, builds the `IoLayout`, and
  fills the shared input block each step.
- `actions/mc_rtc_controller_host.py` — `ControllerHost` (one per worker) holds per-env
  `MCGlobalController`s and marshals encoder/root/IMU/wrench inputs and
  position/velocity outputs through the shared blocks. Controllers live in
  worker processes because construction (~570 ms, ~70 MB each, serial-only)
  and Cython marshalling are GIL-bound; only the patched binding's `run()`
  releases the GIL. Env count is memory-bound in practice. Beside the
  per-joint `output_channels` it can publish whole-controller 3-vectors
  (`VECTOR_OUTPUTS` → the action cfg's `controller_vectors`, read back with
  `controller_vector(name)`), which is how `mdp.zmp_tracking` gets the
  controller's planned ZMP; those are not interpolated across substeps.
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
  that as a `[WARN]` plus traceback rather than failing.
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
  (may include joints mjlab does not simulate or actuate); the host expands
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
  false" nor "reset() returns" can be relied on for a fallen robot. The
  pool's timeout + worker quarantine is the containment; do not remove it.
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
  wrong for anything dynamic: `VECTOR_OUTPUTS` readers take
  `controller().robot()` instead, the robot the QP integrates.
- `MCGlobalController::reset()` does `controllers.erase(...)` + `AddController(...)`:
  it **destroys and rebuilds** the controller. No handle into it survives an env
  reset, so never cache the `MCController` from `controller()` (or a `Robot`
  from it) across steps — the worker segfaults one reset later, far from the
  cache. Re-resolve every step; it costs a wrapper allocation.
- mc_rtc bindings expose no datastore access, so a controller's own planned
  quantities (ismpc's `zmpTarget` under `ismpc_walking::zmp_target`) are out of
  reach without patching them. `mdp.zmp_tracking` therefore derives the plan
  from the control robot's centroid instead — `rbd::computeCentroidalZMP` on
  `com`/`comAcceleration`, i.e. the ZMP the QP's commanded motion implies. It
  differs from `zmpTarget` by ismpc's ZMP-delay compensation (the stabilizer's
  CoM-acceleration target comes from `admittanceTarget`, not `zmpTarget`).
- mc_rtc's terminal logging is hardwired C++ spdlog; silencing requires
  fd-level redirection (`suppress_mc_rtc_output`), not `sys.stdout` swaps.
  The cfg's `console_output` picks "none" (silence all, the default),
  "single" (env 0 only, in a dedicated worker) or "all"; both tasks' play
  variants override it to "single". `play` exposes no `--env.*` overrides
  (only `train` does), so the escape hatch for a run that is already going is
  `MC_MJLAB_WORKER_LOG_DIR=<dir>`, which redirects each worker's output to a
  file there whatever the cfg says.
- The residual itself is printed during `play`: the action cfg's
  `print_residual_every` (residual_balance's play variant sets 10, i.e. 5 Hz)
  writes one `[residual]` line per interval with env 0's per-joint residual in
  rad or Nm, `*` marking a joint at its clip, and the vector norm last. Same
  escape hatch as above — `MC_MJLAB_PRINT_RESIDUAL=<n>` retunes the interval,
  0 silences it. The viewers cannot show this themselves: they only surface
  *reward* and *metrics* manager terms, never actions.
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
