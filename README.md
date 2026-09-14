# mc_mjlab

[mc_rtc](https://github.com/mc-rtc/mc-rtc-superbuild) controller integration for
[mjlab](https://github.com/mujocolab/mjlab).

## Layout

```
src/mc_mjlab/
  actions/mc_rtc_residual_joint_position_actions.py  # McRtcResidualJointPositionAction(Cfg)
  actions/mc_rtc_residual_joint_torque_actions.py    # McRtcResidualJointTorqueAction(Cfg)
  actions/mc_rtc_residual_action.py        # residual action base (interpolation, async dispatch)
  robots/                     # constants (assets are dynamically symlinked from mc_rtc install path)
  tasks/__init__.py           # imports every task sub-package (mjlab.tasks entry point)
  tasks/mdp.py                # the tasks' MDP terms: rewards, observations, events, metrics
  tasks/residual_balance/     # the RL task: __init__ registers the ids, env cfg + PPO cfg alongside
  tasks/residual_mpc/         # paper-style task: residual on the MPC's own inputs
  tasks/residual_feedback/    # residual on the controller's feedback rather than its output
  tasks/zero_residual/        # the demo task: mc_rtc alone, RL residual left at zero
  recovery_authority.py       # recovery detector and authority gating
  residual_mpc.py             # MPC-side residual plumbing
  residual_safety.py          # residual clipping and feasibility guards
  controller_io.py            # simulation joints, root and sensors in native layout
  controller_datastore.py     # numeric aliases, gated commands and baselines
src/mc_rtc_interface/
  cpp/                        # native manager, worker, host and controller instance
  hpp/io_layout.hpp           # authoritative shared-memory offsets
  hpp/ipc_socket.hpp          # worker protocol and memory descriptions
  bindings/module.cpp        # Python ControllersManager bindings
src/utils/                    # task ids, config, PD gains and shared memory
docs/                         # why the numbers are what they are (see docs/README.md)
etc/
  mc_rtc.yaml                 # mc_rtc controller config
scripts/
  compare_to_baseline.py      # score a checkpoint against the zero-residual controller
  probe_residual_authority.py # can the residual move the centre of pressure at all?
  validate_dcm_objective.py   # does the DCM reward score standing and walking alike?
  demos/run_test_mc_rtc.sh    # launcher: play --agent zero on the zero-residual task
```

Training and playing use mjlab's own `train`/`play` scripts — see
[Training and playing](#training-and-playing).

## Setup

### Native controller interface

`src/mc_rtc_interface` builds itself — there is no manual cmake step. The
build backend is scikit-build-core, so `uv sync` configures and builds it into
`build/`, and `editable.rebuild` means any later `uv run` rebuilds and
reinstalls it on import when a source file has changed. The actions own a
native `ControllersManager` and two shared-memory blocks;
`controller_timeout_ms` defaults to 60000. `console_output` selects no rows,
environment zero (`single`), or every row (`all`).

The extension and its worker load out of `build/install/platlib`, which the
editable install points at — they are not copied into `.venv`. That persistent
`build/` is also what makes the native tests runnable directly:

```sh
cd build && ctest            # native tests plus the deterministic contracts
```

Configuring by hand is only for building against a prefix other than this
project's: `cmake -S src/mc_rtc_interface -B <dir>` then
`cmake --install <dir> --prefix <prefix>`, which otherwise defaults to the
selected interpreter's environment.

### Controller datastore

The action reads numbers out of mc_rtc through its datastore, under public
aliases the task configures. Which side provides a callback is decided by one
rule: **a name beginning `mc_mjlab::` comes from this repo's own plugin, and
every other name must already exist on the controller.**

`src/mc_rtc_interface/{hpp,cpp}/instance_datastore_plugin.*` is that plugin. It
is compiled into the native interface — there is nothing to install and no
controller to patch. `ControllerInstance::finish_reset` registers each prefixed
name on the controller's datastore, binding it to the namespace function whose
name is the rest of the entry:

| Entry | Type | Value |
| --- | --- | --- |
| `mc_mjlab::planned_zmp` | `Vector3d` | control-centroid ZMP of the QP's own solution |
| `mc_mjlab::control_com` | `Vector3d` | CoM of the robot the QP integrates |
| `mc_mjlab::control_com_vel` | `Vector3d` | that robot's CoM velocity |
| `mc_mjlab::support_foot` | `double` | right=0, left=1 |

Registration repeats on **every** controller build, because
`MCGlobalController::reset()` destroys and rebuilds the controller together with
its datastore — a registration made once would not survive the first episode
reset. The getters read `MCController::robot()`, the robot the QP integrates,
not the canonical output robot, whose `comVelocity`/`comAcceleration` read
exactly zero.

`support_foot` is the one entry that still needs something from the controller:
it converts the `std::string` `ismpc_walking::support_foot_name`, which a
numeric callback cannot carry, using its `Left`/`Right` prefix.

Three things are hard errors at controller init rather than silent zeros:

- a `mc_mjlab::` name matching no plugin function,
- a `mc_mjlab::` name the controller *already* defines — two definitions of one
  quantity with no way to tell which is live,
- any configured non-prefixed callback the controller does not provide.

Point an alias at a different callback with `controller_vector_callbacks` /
`controller_scalar_callbacks` on the action cfg; the defaults are
`VECTOR_CALLBACKS` and `SCALAR_CALLBACKS` in `controller_datastore.py`. That is
also how a controller-side entry is reached — `walking_ref_vel` maps to
`ismpc_walking::get_ref_vel`, carries no prefix, and so must come from the
controller.

> [!CAUTION]
> `planned_zmp` must keep its control-centroid definition. Checkpoints were
> trained against it, and ISMPC's delay-compensated reachable `zmp_target` is a
> different quantity. Never substitute it, and never let the callback return
> zero.

Writing back through the datastore needs native per-callback usage flags;
commands fail loudly when the usage-offset methods or the configured callbacks
are absent. See [the numeric interface and unsupported callback
inventory](docs/coupling.md#DatastoreCommands), and
[instance_datastore_plugin](docs/coupling.md#instance_datastore_plugin) for the
live measurements behind each getter.

To check the adapter end to end, run `cd build && ctest` for the native tests
and deterministic contracts, then the live walking checks (`uv run python
scripts/verify_native_action_live.py --mode position`, then `--mode torque`). A
zero-residual demo exercises neither the adapter nor worker recovery, so it is
not a substitute. Checkpoint contract validation remains enforced either way.

### mjlab dependency

`pyproject.toml` declares a plain `mjlab` dependency and deliberately does not
choose where it comes from. Pick one before the first sync:

- **PyPI release**: nothing to add.

- **Local checkout**: develop against a local mjlab instead of a release. Add to
  `pyproject.toml`, but do not commit it:

  ```toml
  [tool.uv.sources]
  mjlab = { path = "<path/to/mjlab>", editable = true }
  ```

- **Git**: track upstream without a local checkout:

  ```toml
  [tool.uv.sources]
  mjlab = { git = "https://github.com/mujocolab/mjlab" }
  ```

then run:

```sh
uv sync
```

### mc_rtc dependency

Refer to the superbuild tutorial

> [!CAUTION] 
> The mc_rtc Python bindings and controller libraries come from the
> sourced workspace (`PYTHONPATH`/`LD_LIBRARY_PATH`); run from a shell that has
> it sourced. The workspace's bindings must be built for the same interpreter as
> this package's venv (`requires-python` pins it): a version mismatch fails at
> import, or worse, segfaults.

### ROS plugin: keep autoload disabled

mc_rtc autoloads its ROS plugin into every process that constructs an
`MCGlobalController` — here, every controller worker. The plugin's background
threads (an rclcpp node plus DDS discovery) corrupt the process heap: in a
controlled test, 15/15 short-lived controller processes crashed at teardown with
the plugin loaded (SIGSEGV, or `munmap_chunk(): invalid pointer` after resets)
and 0/15 without it, matching a week of kernel-log segfaults across python3 and
mc_mujoco. With dozens of workers this surfaced as workers dying or wedging
mid-training.

Autoload is disabled machine-side by removing the marker directory:

```sh
cd ~/workspace/install/lib/mc_plugins && mv autoload autoload.old
```

A workspace rebuild/reinstall can recreate it — if controller workers start
dying again, check this first. The plugins a controller itself requests (e.g.
`footsteps_planner_plugin` for LogisticController_ismpc) still load on demand;
only the unconditional autoload is affected.

## Running the demo

The demo runs the controller specified in `etc/mc_rtc.yaml` with the RL
residual left at zero, so the robot tracks raw mc_rtc output — a healthy run
holds a steady root height. It is a registered task (`Mc-Mjlab-Zero-Residual-*`)
driven by mjlab's `play --agent zero`; the script only resolves the id and
opens the viewer.

```sh
scripts/demos/run_test_mc_rtc.sh                    # viser viewer, 1 env
scripts/demos/run_test_mc_rtc.sh --num-envs 8       # extra args go to `play`
scripts/demos/run_test_mc_rtc.sh --viewer native    # native viewer instead
MC_MJLAB_CONTROL=torque scripts/demos/run_test_mc_rtc.sh   # torque control mode
```

> [!NOTE]
> `Mc-Mjlab-Zero-Residual-*` is a play task; `train` on it is not supported.
> It has no reward to optimise, and it is built for a zero action — a policy
> sampling every 2 ms with nothing to terminate a wrecked robot overruns the
> contact budget and faults in the physics. Train the balance task instead.

## Training and playing

Use existing `train` and `play` scripts:

A task id is `Mc-Mjlab-<task dir>-<Enabled>-<MainRobot>-<control suffix>`, the
controller and robot read from `etc/mc_rtc.yaml`, so editing that file changes
the ids — which is the point, since it also changes what a checkpoint is valid
against. It is title-cased with `_` turned into `-`, so it is not the yaml's
spelling: `LogisticController_ismpc` on `HRP5P` reads
`Logisticcontroller-Ismpc-Hrp5P`. Use `list-envs` rather than assembling one;
these are for the config as committed (`MainRobot: HRP5P`,
`Enabled: LogisticController_ismpc`):

```sh
uv run list-envs   # this repo's ids, plus mjlab's
uv run train Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position
uv run play  Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position \
  --checkpoint-file <path/to/model_*.pt>
```

The default mc_mjlab surface is ten tasks: zero-residual position/torque,
residual-balance position/torque, residual-balance position-ankle, the ankle
achievement curriculum, the two matched-impulse ankle variants, and the
residual-mpc and residual-feedback joint-torque tasks. Ten completed ablations
are hidden so `import mjlab` does not build them. To play an old velocity, sagittal, hardware, frozen/gradual, robust,
history, or GRU checkpoint under its original id:

```sh
MC_MJLAB_REGISTER_ARCHIVED_TASKS=1 uv run list-envs
MC_MJLAB_REGISTER_ARCHIVED_TASKS=1 uv run play <archived-task-id> \
  --checkpoint-file <path/to/model_*.pt>
```

Every residual-balance training run publishes a PID-bound heartbeat under
`<run>/watchdog/`. For an unattended run, attach the cooperative monitor from a
second tmux window:

```sh
run_dir=/absolute/path/to/the/run
trainer_pid=$(jq -r .pid "$run_dir/watchdog/runtime.json")
uv run python scripts/watch_training.py \
  --pid "$trainer_pid" --run-dir "$run_dir" --max-level stop
```

It warns on stale progress, low GPU memory, controller-worker failures, and
optional qualification regressions. A stop is honored only at an iteration
boundary and only after a recoverable checkpoint has been acknowledged. See
[docs/training-watchdog.md](docs/training-watchdog.md) for thresholds and
artifacts.

### Achievement-gated curriculum

The `Position-Ankle-Curriculum-Achievement` task advances only from held-out
qualification reports. Find its full controller/robot-specific id with
`uv run list-envs`, then train it normally. The run publishes its current stage
under `<run>/curriculum/qualification_request.json`.

Evaluate one of that run's checkpoints against the requested stage:

```sh
run_dir=/absolute/path/to/the/run
checkpoint="$run_dir/model_100.pt"
stage=$(jq -r .stage "$run_dir/curriculum/qualification_request.json")
uv run python scripts/qualify_checkpoints.py "$checkpoint" \
  --achievement-stage "$stage" --out-dir "$run_dir/curriculum"
```

Achievement mode defaults to seeds 42 and 43, all four qualification scenarios,
and ankle authority. A second distinct passing report advances the stage; three
valid failing reports roll it back. Use a later checkpoint or different seeds
when replacing the report. The exact stage-passing checkpoint is retained under
`<run>/curriculum/`, and the complete hysteresis state is embedded in subsequent
training checkpoints. See [docs/difficulty.md](docs/difficulty.md) for the
mixtures and resume semantics.

> [!TIP] 
> To add a task, drop a package under `src/mc_mjlab/tasks/` whose
> `__init__.py` calls `register_mjlab_task`; the walk picks it up with no
> wiring. It has to be a directory — a bare module beside `tasks/__init__.py` is
> never imported and would silently never register.

### Did it beat the controller?

Not something the training curves can say: `Episode_Reward/*` is an episode sum,
and those correlate with episode length at r = +0.98.

```sh
# Score a checkpoint against the zero-residual controller, deterministically.
uv run python scripts/compare_to_baseline.py --checkpoint <path/to/model_*.pt>
# Inspect every raw reward, its live weight, rate, activity, and distribution.
uv run python scripts/audit_rewards.py --checkpoint <path/to/model_*.pt>
# Can a constant residual move the centre of pressure at all?
uv run python scripts/probe_residual_authority.py --level 1.0
uv run python scripts/probe_walking_reference.py
uv run python scripts/inspect_controller_datastore.py
uv run python scripts/probe_step_duration.py
# Does the DCM objective still prefer standing to walking? (no checkpoint needed)
uv run python scripts/validate_dcm_objective.py
```

`compare_to_baseline.py` steps both arms in one run and takes a fixed number of
episodes per env: counting everything that finished inside a time budget
oversamples short episodes and inflates the failure rate. `--num-workers`
defaults low on both, to leave room for a training job.

### Environment variables

| Variable | Effect |
| --- | --- |
| `MC_MJLAB_CONTROL` | `position` (default) or `torque`, for the demo |
| `MC_MJLAB_PRINT_RESIDUAL` | Steps between `[residual]` printouts during `play`; `0` silences |
| `MC_MJLAB_REGISTER_ARCHIVED_TASKS` | `1` also registers the ten archived ablation ids |
| `MC_MJLAB_PUSH_DEBUG` | Positive float: scale pushes, cap warm-up at 1 s, print `[push]` lines |

### External paths

These symlink outside this repo (into the mc_rtc workspace) and must exist for
the demo to actually step controllers:

- `src/mc_mjlab/robots/<ROBOT>/` — the MJCF, meshes, and PD gains are symlinked
  on first use from `$HOME/workspace/install/share/mc_mujoco/<ROBOT>` if they
  are found
