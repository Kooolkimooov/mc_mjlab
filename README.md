# mc_mjlab

[mc_rtc](https://github.com/mc-rtc/mc-rtc-superbuild) controller integration for
[mjlab](https://github.com/mujocolab/mjlab).

## Layout

```
src/mc_mjlab/
  actions/mc_rtc_residual_joint_position_actions.py  # McRtcResidualJointPositionAction(Cfg)
  actions/mc_rtc_residual_joint_torque_actions.py    # McRtcResidualJointTorqueAction(Cfg)
  actions/mc_rtc_residual_action.py        # residual action base (interpolation, async dispatch)
  actions/mc_rtc_controller_pool.py        # worker processes, pipes, shared-memory blocks
  actions/mc_rtc_controller_io_binding.py  # sim <-> mc_rtc I/O wiring (IoLayout, input assembly)
  actions/mc_rtc_controller_host.py        # worker-side per-env controller host
  robots/                     # constants (assets are dynamically symlinked from mc_rtc install path)
  tasks/__init__.py           # imports every task sub-package (mjlab.tasks entry point)
  tasks/residual_balance/     # the RL task: __init__ registers the ids, env cfg + PPO cfg alongside
  tasks/zero_residual/        # the demo task: mc_rtc alone, RL residual left at zero
etc/
  mc_rtc.yaml                 # mc_rtc controller config
scripts/demos/
  run_test_mc_rtc.sh          # launcher: play --agent zero on the zero-residual task
```

Training and playing use mjlab's own `train`/`play` scripts — see
[Training and playing](#training-and-playing).

## Setup

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
scripts/demos/run_test_mc_rtc.sh                    # viser viewer, 2 envs
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

A task id is `Mc-Mjlab-<task dir>-<Enabled>-<MainRobot>-<Position|Torque>`, the
controller and robot read from `etc/mc_rtc.yaml`, so editing that file changes
the ids — which is the point, since it also changes what a checkpoint is valid
against. `list-envs` prints the ids for your config; these are for the config
as committed (`MainRobot: JVRC1`, `Enabled: Posture`):

```sh
uv run list-envs   # this repo's ids, plus mjlab's
uv run train Mc-Mjlab-Residual-Balance-Posture-Jvrc1-Position
uv run play  Mc-Mjlab-Residual-Balance-Posture-Jvrc1-Position \
  --checkpoint-file <path/to/model_*.pt>
```

> [!TIP] 
> To add a task, drop a package under `src/mc_mjlab/tasks/` whose
> `__init__.py` calls `register_mjlab_task`; the walk picks it up with no
> wiring. It has to be a directory — a bare module beside `tasks/__init__.py` is
> never imported and would silently never register.

### External paths

These symlink outside this repo (into the mc_rtc workspace) and must exist for
the demo to actually step controllers:

- `src/mc_mjlab/robots/<ROBOT>/` — the MJCF, meshes, and PD gains are symlinked
  on first use from `$HOME/workspace/install/share/mc_mujoco/<ROBOT>` if they
  are found
