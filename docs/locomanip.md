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

## CART_INIT_X

**Current:** `0.90` m, the cart body's initial x in front of the robot's stance.

**The number that matters is not this one.** `Cart.xml` puts the grasped handle
bar at cart-local `x = -0.35, z = 1.0`, matching the controller's
`ManipManager.objToHandTranss` of `[-0.35, +/-0.3, 1.0]`. So the hands reach to
`CART_INIT_X - 0.35`, while the cart box's rear face — the part that looks like
"the cart" — starts at `CART_INIT_X` itself and runs 0.85 m forward from there
(box half-size 0.425 centred at local `x = 0.425`).

At the original `0.75` the handle sat **0.40 m** in front of the robot, which
read as the cart crowding it. `0.90` puts the handle at **0.55 m** and the box
rear face at 0.90 m.

**This is a free choice, not a value the controller pins.** The environment
feeds the measured cart pose to mc_rtc through `controller_objects = {obj:
cart}`, so the controller grasps wherever the cart actually is; it does not
assume an initial pose. Only reachability bounds it — the hands must still make
`[-0.35, +/-0.3, 1.0]` in the object frame from the initial stance.

**Re-measure if:** the cart asset changes, `objToHandTranss` changes, or the
robot's initial stance moves.

**History:**
- 2026-09-16 — raised `0.75` -> `0.90` on the report that the cart was too close
  to the robot. Geometry above read from `Cart.xml` and
  `LocomanipController.yaml`; the handle offset is what set the spacing.

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

## locomanip_residual_env_cfg

**Current:** The trainable task, registered as
`Mc-Mjlab-Locomanip-Locomanipcontroller-Hrp5P-Position-Residual`, beside the
play-only demo id. It is deliberately a skeleton: one task reward, the four
regularizers every residual task carries, the five terminations, four metrics,
and no curriculum. Each manager comes from its own `_`-prefixed builder in the
module, so a new term is one entry in one dict.

What the wiring fixes, and why each choice is not free:

| Setting | Value | Why it cannot simply be changed |
| --- | --- | --- |
| Action name | `mc_rtc_residual` | `rl/controller_provenance.py` reads this key by name; the demo's `robot_joints` would `KeyError` before the first step |
| `decimation` | 20 | 50 Hz policy over the 500 Hz controller; must stay divisible by `frameskip` |
| `frameskip` | 2 | 1 kHz physics into the config's 2 ms controller period |
| `jacobian` | `sparse` | HRP5P plus the free cart is 65 velocities, over MuJoCo Warp's dense limit of 60 |
| `required_controller` | `LocomanipController` | a wrong `Enabled` otherwise surfaces as a missing datastore callback |
| `episode_length_s` | 60.0 | the installed DemoFSM completes its cycle in 52.194 s |
| `num_envs` | 32 | a starting point, not a measurement: one Locomanip controller per env, built serially, plus a cart |

**Deliberately absent**, in the order they are likely to matter: per-episode goal
randomization (the FSM pushes the same 1 m waypoint every episode, so the task is
one trajectory until a datastore setter feeds the waypoint), payload
randomization beyond the `cart_mass_scale` hook, contact and hand-wrench reward
terms, a curriculum, and an actor/critic split — both groups are the same terms.

**Re-measure if:** the controller config, the FSM, the robot or the cart asset
changes.

**History:** added as the barebones trainable task, alongside the zero-residual
demo that stays play-only.

## RESIDUAL_SCALE

**Current:** `0.01` rad, applied to every actuator as a float and clipped to the
same magnitude, matching residual balance's uniform position authority.

**Not measured for this task.** It is inherited, not derived: no probe has been
run against Locomanip's own authority. `docs/residual-authority.md` explains how
the balance task replaced its uniform scale with per-joint hardware limits, which
is the same move available here.

**Re-measure if:** the residual joint set narrows, or the control mode changes to
torque.

**History:** taken from `residual_balance`'s uniform position scale.

## OBJECT_TRACKING_STD

**Current:** `0.10` m, the Gaussian kernel width of `object_position_tracking`.

**Not measured for this task.** The kernel has to be placed against the baseline's
own error distribution, and that distribution has not been taken on HRP5P: the
JVRC1 sweep in the knowledge base puts the zero-residual cart error between
0.02 m and 2.1 m depending on the cart mass, which is three orders of magnitude of
choice. Place it before reading anything into the reward curve.

**Re-measure if:** the cart, the waypoint, or the robot changes.

**History:** chosen as a round number so the task trains at all.

## cart_mass_range_kg

**Current:** `(1.0, 1000.0)` kg on a 10 kg cart asset, drawn per episode by a
`reset`-mode `dr.pseudo_inertia` event on the cart body. It is the range the
mc_mujoco sweep characterised: 50 log-spaced masses from 1 kg to 1000 kg.

`pseudo_inertia`'s `alpha` is a *log* scale -- mass and inertia both scale by
`e^(2a)` -- so `mass_alpha_range` converts kilograms to it, and sampling `alpha`
uniformly makes the mass log-uniform, the spacing the sweep used.
`dr.body_mass` is the wrong term here: it leaves inertia behind.

**This is the whole experiment.** mc_rtc keeps modelling the 10 kg `LMC/Cart`
URDF whatever MuJoCo simulates, and the object feedback carries pose, never mass,
so a heavy cart is exactly the feedforward mismatch the base controller cannot
observe. On JVRC1 in mc_mujoco that mismatch dropped the robot from 105 kg
upward, while the same controller given the true mass pushed 868 kg without
falling.

**Expect most sampled episodes to fail** until something closes that gap: two
thirds of a log-uniform draw over 1-1000 kg sits above 100 kg. Narrow the range
for a first training run rather than reading the reward curve as a policy
verdict.

**Re-measure if:** the cart asset's mass or body name changes, or the robot does
-- the 105 kg threshold is a JVRC1 measurement, not an HRP5P one.

**History:**
- 2026-09-16 -- wired as an opt-in `cart_mass_scale` hook, then given the swept
  range as its default and moved to per-episode resampling.

## ZMP_TRACKING_STD

**Current:** `0.05` m, the Gaussian kernel width of the `zmp_tracking` reward,
which scores the distance between the measured centre of pressure and the ZMP the
controller planned, and scores **zero** on any step whose feet carry less than
`min_normal_force` (20 N) -- an unloaded foot has no centre of pressure, and
without the mask a robot in flight would collect the maximum.

Read the companion metrics as a pair, `zmp_error / zmp_grounded`: `zmp_error`
alone averages over every step, so it *falls* when the robot spends more time off
the ground.

**Not measured for this task.** Standing, the baseline already scores 0.997 on
this kernel, so at this width the term says almost nothing until the push loads
the feet; the width has to be placed against the error distribution during the
push, which has not been taken on HRP5P.

**Re-measure if:** the robot, the cart or the foot force sensors change.

**History:** chosen as a round number, half of `residual_balance`'s `DCM_STD`.
