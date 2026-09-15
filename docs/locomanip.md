# Locomanip cart demo

## controller_objects

**Current:** `controller_objects: {obj: cart}` maps the passive MuJoCo cart to
mc_rtc's `obj` robot. Each dispatch carries environment-local position, an xyzw
quaternion, world linear velocity, and world angular velocity. Ordinary steps
update only `realRobot("obj")`; initialization and reset seed both measured and
reference robots before Locomanip initializes its trajectory.

**Re-measure if:** the cart model, environment origins, reset ordering, object
transport, or Locomanip initialization changes.

**History:** Added for the zero-residual HRP5P cart milestone. Empty mappings
leave the earlier native input offsets unchanged.

## LocomanipController dependency patch

**Current:** Apply `patches/locomanip-unattended.patch` to the LocomanipController
source, rebuild it, and install the rebuilt libraries into the sourced workspace.
The patch adds `ManipManager.enableRos`, numeric object/phase/completion datastore
callbacks, and an opt-in `DemoFSM` reload. The repository profile uses these
features without changing the installed controller YAML.

```sh
cd ~/workspace/src/catkin_locomanip_ws/src/LocomanipController
git apply /home/martin/git/mc_mjlab/patches/locomanip-unattended.patch
cmake -S . -B /tmp/locomanip-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$HOME/workspace/install" \
  -DCMAKE_PREFIX_PATH="$HOME/workspace/install" \
  -DBUILD_TESTING=OFF -DENABLE_MUJOCO=OFF -DENABLE_CNOID=OFF
cmake --build /tmp/locomanip-build -j4
install -m 755 /tmp/locomanip-build/src/libLocomanipController.so \
  "$HOME/workspace/install/lib/libLocomanipController.so"
install -m 755 /tmp/locomanip-build/src/LocomanipController.so \
  "$HOME/workspace/install/lib/mc_controller/LocomanipController.so"
install -m 755 /tmp/locomanip-build/src/states/*State.so \
  "$HOME/workspace/install/lib/mc_controller/locomanip_controller/states/"
```

These targeted installs leave the machine-wide `LocomanipController.yaml`
untouched. A broad `cmake --install` would regenerate and overwrite it.

**Re-measure if:** Locomanip gains native non-ROS object input, its FSM is built
after overwrite processing, or equivalent datastore callbacks land upstream.

**History:** The patch retains ROS behavior by default. The demo sets
`enableRos: false`, preventing node creation, subscription setup, and spinning.

## HRP5P Locomanip cart acceptance

**Current:** Run `uv run python scripts/verify_locomanip.py`. It disables automatic
reset, records a JSONL transition trace, and requires five complete cycles plus
the physical pose, displacement, contact, release, upright, health, and two-second
hold gates. Run `--cycles 1 --isolation` separately for the two-environment shared
worker and mid-sequence reset gate.

The five-cycle acceptance run completed every cycle in 52.194 simulated seconds.
Cart position error ranged from 0.1331 to 0.1355 m and yaw error from 0.6013 to
0.6811 degrees. Base displacement ranged from 1.0819 to 1.0821 m, minimum root
height from 0.7151 to 0.7152 m, and measured bilateral cart contact from 0.080 to
0.106 seconds. Both hands returned to Free, every cycle held for 2.0 seconds,
and no controller or worker failed. Machine-readable results and the transition
trace are in `logs/locomanip/five-cycle.json` and its adjacent JSONL file.

The two-environment isolation run reset env 0 at 10 simulated seconds. Env 1
continued and completed, then env 0 completed its restarted cycle. Both took
52.194 seconds in their own episode clocks. Their cart errors were 0.1382 m and
0.1374 m, and neither row reported a controller or worker failure.

**Re-measure if:** controller configuration, hand frames, collision geometry,
friction, PD gains, sensor routing, or async timing changes.

**History:** HRP5P needs `LeftHandCloseContact` and `RightHandCloseContact` in
place of Locomanip's JVRC1 gripper frames. The hand transforms are rotated 180
degrees about their local X axes for those planar surfaces. Sparse Jacobians are
required because HRP5P plus the free cart has 65 velocities, above MuJoCo Warp's
dense limit of 60.
