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
etc/
  mc_rtc.yaml                 # mc_rtc controller config
scripts/demos/
  test_mc_rtc.py              # benchmark / viewer demo
  run_test_mc_rtc.sh          # launcher (uv run + viser viewer by default)
```

## Setup

### mjlab dependency

`pyproject.toml` declares a plain `mjlab` dependency and deliberately does not
choose where it comes from. Pick one before the first sync:

- **PyPI release**: nothing to add.

- **Local checkout**: develop against a local mjlab instead of a release.
  Add to `pyproject.toml`, but do not commit it:

  ```toml
  [tool.uv.sources]
  mjlab = { path = "../mjlab", editable = true }
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

### ROS plugin: keep autoload disabled

mc_rtc autoloads its ROS plugin into every process that constructs an
`MCGlobalController` — here, every controller worker. The plugin's background
threads (an rclcpp node plus DDS discovery) corrupt the process heap: in a
controlled test, 15/15 short-lived controller processes crashed at teardown
with the plugin loaded (SIGSEGV, or `munmap_chunk(): invalid pointer` after
resets) and 0/15 without it, matching a week of kernel-log segfaults across
python3 and mc_mujoco. With dozens of workers this surfaced as workers dying
or wedging mid-training.

Autoload is disabled machine-side by removing the marker directory:

```sh
cd ~/workspace/install/lib/mc_plugins && mv autoload autoload.old
```

A workspace rebuild/reinstall can recreate it — if controller workers start
dying again, check this first. The plugins a controller itself requests (e.g.
`footsteps_planner_plugin` for LogisticController_ismpc) still load on demand;
only the unconditional autoload is affected.

## Running the demo

The mc_rtc Python bindings and controller libraries come from the sourced
workspace (`PYTHONPATH`/`LD_LIBRARY_PATH`); run from a shell that has it
sourced. The workspace's bindings must be built for the same interpreter as this
package's venv (`requires-python` pins it): a version mismatch fails at import,
or worse, segfaults.

```sh
scripts/demos/run_test_mc_rtc.sh                # viser viewer (2 envs, cpu)
scripts/demos/run_test_mc_rtc.sh --viewer none  # throughput benchmark (420 envs, cuda)
```

This loads the config from `etc/mc_rtc.yaml`.

### External paths

These point outside this repo (into the ROS workspace) and must exist for the
demo to actually step controllers:

- `src/mc_mjlab/robots/<ROBOT>/` — the MJCF, meshes, and PD gains are symlinked
  on first use from `$HOME/workspace/install/share/mc_mujoco/<ROBOT>` if they
  are found
