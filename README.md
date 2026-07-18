# mc_mjlab

[mc_rtc](https://github.com/mc-rtc/mc-rtc-superbuild) controller integration for
[mjlab](https://github.com/mujocolab/mjlab).

## Layout

```
src/mc_mjlab/
  actions/mc_rtc_joint_position_actions.py  # McRtcResidualJointPositionAction(Cfg)
  actions/mc_rtc_host.py      # worker-process controller host (shared-memory I/O)
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
