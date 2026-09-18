# AGENTS.md

This file provides guidance to coding agents working in this repository.
`CLAUDE.md` is a symlink to it, so both names read the same bytes; edit this file.

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
# Ids are Mc-Mjlab-<task dir>-<Enabled>-<MainRobot>-<control suffix>,
# built by tasks/naming.py, which reads Enabled/MainRobot from
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
# Regenerate docs/architecture/ from the source; --check fails on drift.
uv run python scripts/generate_architecture_docs.py
uv run ruff format && uv run ruff check --fix    # format + lint
uv run ty check                                  # type check (117 pre-existing
                                                 # diagnostics: unresolvable
                                                 # mc_rtc bindings + mujoco stubs)
uv run pytest                                    # tests/: bindings + action contracts
cd build && ctest                                # C++ tests, worker recovery, and pytest
                                                 # (--target check-native skips pytest)
python3 scripts/check_prose.py src scripts tests # prose budget + docs/ links
python3 scripts/check_prose.py --strict src scripts tests  # ... as pre-commit runs it
```

`pytest` carries the improvement contracts; they moved out of a standalone
script into `tests/`, grouped by subject, so they are run rather than merely
runnable. They still do not replace a live controller check; the demo is the simulation verification. A healthy run holds a steady
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

Two hard rules, enforced by `scripts/check_prose.py` (which also runs as a
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
finds `dcm_stability` either way); spend the line only where the connection is
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

# Python style

The prose budget above means the code itself has to carry the explanation, so
write it to be read top to bottom.

- **Sequence a function as steps, not as a puzzle.** Do the work in the order
  someone would describe it, and put a blank line between the steps — wherever
  a reader would take a breath, and always around a block that does something
  different from the line before it.
- **Early returns over nesting.** Guard clauses first, then a flat body. A
  second level of indentation inside a method is usually a helper waiting to be
  extracted.
- **Order class methods by importance.** The lifecycle and the methods callers
  actually use come first (`__init__`, `process_actions`, `apply_actions`,
  `reset`, `close`), then the public accessors, then the `_`-prefixed helpers,
  and the `@property` definitions last, as one block at the bottom.
  `McRtcResidualActionBase` is the worked example.
- **Privates go below the publics, in call order.** The public block reads
  what-before-how; the `_`-prefixed block underneath follows the order those
  publics call into it, so the file still reads as one pass. A helper called
  from several places sits below the first of them.
- **One job per module.** When a module grows a second job, split it rather
  than sectioning it with comments — that is why `mdp/` is seven submodules and
  why the PPO config sits beside the env cfg rather than inside it. The 10%
  comment budget is the tripwire for this, not a separate rule.
- **Types are part of the signature.** Every module opens with
  `from __future__ import annotations`, and every def annotates its parameters
  and its return. Annotation-only imports (`ManagerBasedRlEnv`, mjlab's
  `*TermCfg`, our own action terms) go under `if TYPE_CHECKING:`, which is also
  what keeps the `mdp` ↔ `actions` direction from closing into a cycle.
- 2-space indent (ruff `indent-width = 2`), 88-column lines. `uv run ruff
  format` settles everything it can, and none of the above.

# Commit messages

Keep them concise: a subject line plus a 2-4 line body carrying the one number
or reason the diff does not show. Everything longer belongs in `docs/` under a
grep-able `##` heading.

**Never** put a session line, a session id, an agent URL, or a "Generated with
<agent>" line in a commit message — they outlive the session and stay in
`git log` forever. Your own agent's `Co-Authored-By:` trailer is wanted and
stays; use the model identity of the session you are in rather than one copied
from an earlier commit. This holds for messages carried through a history
rewrite too: strip the session line rather than preserve it.

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
  residual on torque). Three more modules extend that base rather than widening
  it: `walking_reference_action.py` feeds `ismpc_walking::set_ref_vel` through
  the generic extension hooks (`AbsoluteWalkingReferenceMixin` is the surviving
  drive mode, a command-manager target adding no action dimensions; the
  recovery-gated delta variant is retired, docs/walking-reference.md);
  `residual_feedback_action.py` puts the residual on the controller's own
  feedback rather than its output; and `residual_mpc_joint_torque_action.py`
  combines the torque action with the absolute walking reference.
- `residuals/` — the residual machinery the action terms compose, one concern
  per module: `safety` (feasibility projection against the `RobotModule`'s
  bounds), `recovery_authority` (the calibrated detector and its gate),
  `mpc_math` and `printer`. It depends on `bridge/`, never on `mdp/` or `tasks/`.
- `mc_mjlab/bridge/sensors.py` — the low-level MuJoCo sensor lookups
  (`wrench_sensor`) that both `mdp/sensors.py` and `residuals/recovery_authority.py`
  need. It imports neither actions nor MDP terms, which is the whole reason it
  is its own module rather than a helper on either caller.
- `mc_mjlab/bridge/sim_controller_bridge.py` — simulation-side reference-order scatter/gather,
  biased encoders, measured effort, local root coordinates, wxyz-to-xyzw
  conversion and named sensors. Use native layout offset methods throughout.
- `mc_mjlab/bridge/controller_datastore.py` — numeric output aliases and independently
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
  Public `alpha` maps to native `qd`. Python retains `bridge/shared_memory.py`.
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
  that as a `[WARN]` plus traceback rather than failing. Twelve ids register, and
  they are all of them (six residual-balance, two zero-residual, two locomanip --
  the play-only cart demo and its trainable residual -- one each for
  residual_mpc and residual_feedback): the ten archived ablations and their
  `MC_MJLAB_REGISTER_ARCHIVED_TASKS` switch are gone. Reproducing an archived
  experiment means checking out the revision before that cleanup; every
  *supported* checkpoint still loads.
- `utils/` — what belongs to no one layer. `controller_install.py` answers whether
  the sourced workspace ships a named controller, by its module *or* its
  configuration (a variant like `LogisticController_ismpc` has no `.so` of its
  own). `rl/controller_provenance.py` snapshots what it finds; `tests/conftest.py`
  skips a controller's test modules when it finds nothing.
  docs/coupling.md#controller_is_installed
- `rl/` — what every task shares: the zero-init actor, the squashed Gaussian,
  `RolloutAdaptivePPO`, and `runner.py`'s `McRtcResidualOnPolicyRunner`, which
  snapshots external base controller inputs into the run directory
  (`controller_provenance.py`) and every checkpoint and validates them on load.
  Its effective-training manifest records live resolved manager terms, callable
  defaults and source hashes: full resumes enforce the semantic training
  contract and immediately recompute curricula after restoring the global
  counter, while actor-only loads enforce the narrower observation/action
  interface. `_RENAMED_MODULES` maps the paths this refactor moved, so an older
  checkpoint is compared against today's spelling. Both task runners subclass it
  through five hooks; the balance one adds the achievement curriculum, the
  training budget and the watchdog, the MPC one an action-semantics gate.
  Position and torque registrations use distinct full task ids as experiment
  names, so automatic resume cannot cross control modes.
- `scripts/` — the measurement and maintenance tools, and **not** part of the
  wheel: `[tool.scikit-build.wheel] packages` ships `src/mc_mjlab` only. The
  analysis machinery those scripts share lives in `scripts/evaluation/`, a
  package on pytest's `pythonpath` rather than in the library, because its only
  callers are the scripts and the tests: `rollout` (environment lifecycle,
  reset-without-history-update, pre-reset episode snapshots), `comparison` and
  `qualification` (which share that plumbing but keep their own sampling,
  statistics and episode records — they are different experimental designs, not
  one design twice), `qualification_strata`, `reward_audit`, `disturbances` and
  `scenarios`. A retired script goes with its docs section, which keeps its
  measurements under a `Retired:` heading; see docs/evaluation.md for what each
  surviving one measures.
- `mdp/` — the terms every task builds its managers from, split by
  responsibility: `sensors` (the `_ZmpSensors` plumbing and the action-term
  accessors the rest read through), then `observations`, `rewards`, `metrics`,
  `terminations`, `disturbances` (the push and impulse events) and `curricula`.
  `mdp/__init__.py` binds those seven submodules and nothing else: no star
  imports anywhere in this repo, so every call site reads
  `mdp.<submodule>.<term>` and says which file defines the term.
- `robots/<ROBOT>/<robot>_constants.py` — per-robot constants: spec loading
  (collisions disabled by default, geom groups 2=visual/3=collision/4=sites),
  actuator configs, stance initial state, PD-gains path. The three are
  parallel by construction; each is thin, delegating to the shared
  `robots/*.py` helpers below, and differing only in the
  robot-specific names (root body, foot bodies, deactivated joints).
- `robots/*.py` — the shared machinery those constants files call, one
  concern per module: `robot_module` (joint order, stance, base
  pose and torque limits read lazily from the mc_rtc `RobotModule`, so nothing
  is hand-transcribed), `collisions` (geom naming + the
  `CollisionCfg` presets), `actuators` (gains from the MJCF's
  armature), `sensors` (the RL-only sole velocimeters
  and root angular-momentum sensor), `mc_mujoco_assets` (first-use symlinks),
  and `registry` (`MainRobot` → `RobotSpec`, plus `prepare_cfg_for_mc_rtc`).
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
  (`robots/collisions`) before disabling them by group, and ships
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
- `import mjlab` imports this repo's tasks (its `mjlab.tasks` entry point) and
  they build their cfgs at import, so the *first* module to pull mjlab in must
  not be one of ours: `python -c "from mc_mjlab import mdp"` re-enters a
  half-built `mc_mjlab.actions` and mjlab swallows it as a `[WARN]`, leaving the
  ids unregistered until something imports `mc_mjlab.tasks` again. Any script
  that imports mjlab first — isort puts it first — is unaffected.
- A sourced ROS workspace puts `launch_testing`'s pytest plugin on the path, and
  its `pytest_pycollect_makemodule` imports every `test_*.py` in the directory
  during collection -- even the ones the command line did not name. So
  `pytest.skip(allow_module_level=True)` is unusable here: the `Skipped` escapes
  that hook and ends the session with `found no collectors for <the file you
  asked for>`, reporting one skip and running nothing. Skip a whole file with
  `pytestmark = pytest.mark.skipif(...)`, which is evaluated after the import.
  docs/coupling.md#controller_is_installed
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
  handle. The `PLANNED_ZMP` column `mdp.metrics.zmp_error` reads is deliberately
  the control-centroid plan, not ismpc's reachable `zmp_target`: it is the
  QP-commanded ZMP, and the two differ by delay compensation before the
  stabilizer builds its CoM-acceleration target. Never return zero there.
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
- Every run is seed 42 unless told otherwise; `--agent.seed -1` is the lever that
  draws one instead, and is how a promising checkpoint gets re-screened off its
  training seed. The drawn seed enters the training contract, so such a run
  cannot be resumed without passing the seed its log printed.
  docs/evaluation.md#agentseed
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
