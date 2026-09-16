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

## RESIDUAL_FALLBACK_SCALE

**Current:** per joint, not uniform: `residuals/authority.hardware_residual_scales`
gives each residual joint `0.2 * tau_limit / kp` rad -- a residual worth 20% of
that joint's torque capacity at its own PD stiffness -- with no cap, and the clip
is the same value. 53 actuators are partitioned; the residual joints span
0.0075 rad (hip and knee pitch) to 0.0552 rad (wrist yaw).
`RESIDUAL_FALLBACK_SCALE` only fills the partition for joints that take no
residual.

**Why not the uniform 0.01 rad it started with:** the same number is 27% of a
knee's torque capacity and 4% of a wrist's, and the wrists and shoulders are what
push the cart. Per joint, on HRP5P:

| joint | tau limit (Nm) | kp | 0.2 tau/kp (rad) |
| --- | --- | --- | --- |
| RKP (knee pitch) | 1419.4 | 36000 | 0.0079 |
| RCP (hip pitch) | 602.2 | 16000 | 0.0075 |
| RAR (ankle roll) | 253.4 | 2244 | 0.0226 |
| WP (waist pitch) | 946.3 | 8000 | 0.0237 |
| RSY (shoulder yaw) | 250.9 | 1429 | 0.0351 |
| RWRY (wrist yaw) | 382.5 | 1386 | 0.0552 |

`residual_balance` uses the same rule but caps it at its own uniform 0.01, which
here would throw away two to five times the authority on exactly the joints that
do the work.

**Measured, not assumed.** `scripts/probe_locomanip_authority.py` holds one
environment at zero residual beside constant full-clip arms on the same cart
mass, and reports how far the cart travelled in 16 s (about 5.5 s of it holding):

| cart | zero residual | +1.0 | -1.0 |
| --- | --- | --- | --- |
| 10 kg | +0.002 m | +0.174 m | -0.022 m |
| 300 kg | -0.016 m | +0.189 m | -0.055 m |

Nobody fell (minimum base height 0.703-0.728 m), and the authority survives a
30x payload, which is the regime that matters. The asymmetry is the controller's
own push direction: a positive residual adds to it.

**Re-measure if:** the PD gains file, the robot, the residual joint set or
`TORQUE_FRACTION` changes.

**History:**
- 2026-09-16 -- calibrated from a uniform 0.01 rad inherited from
  `residual_balance`, whose ankle-authority screen has no bearing on pushing.

## OBJECT_TRACKING_STD

**Current:** `0.04` m, the Gaussian kernel width of `object_position_tracking`.

**Placed against the baseline's own error, not guessed.** Zero-residual episodes
on HRP5P put the cart 0.02-0.06 m from its commanded position at the nominal
10 kg (measured 2026-09-16, two environments: 0.144 and 0.154 m at completion
after a full 1 m push, 0.02-0.06 m through the push itself). At 0.04 m a typical
0.03 m error scores `exp(-(0.03/0.04)^2) = 0.57`, so the policy starts mid-range
and can move the term in both directions.

**Why not the 0.10 m it started at:** the same 0.03 m error scores 0.91 there.
The baseline begins at the ceiling, the only gradient left is in the regime where
the cart is already lost, and the curve reads as "solved" from iteration one.
`ZMP_TRACKING_STD` needed no such correction: at 0.05 m the measured 0.036-0.077 m
ZMP error already lands at 0.10-0.60.

**It is gated on the grasp.** Both object kernels are multiplied by
`accessors.holding`, which is 1 only while both hands are in
`ManipPhaseLabel::Hold`. Before the grasp and after the release the object cannot
be moved at all, so an ungated kernel pays a constant 1.0 for roughly a fifth of
the episode -- the reach, and the hold after the FSM completes. `zmp_tracking` is
*not* gated this way: it keeps its own unloaded-feet mask, because the approach
walk is exactly where a ZMP term still says something.

**Re-measure if:** the cart, the waypoint, the payload range or the robot
changes.

**History:**
- 2026-09-16 -- 0.10 m, a round number chosen so the task trained at all,
  narrowed to 0.04 m once the baseline's own error distribution was measured.

## cart_mass_range_kg

**Current:** `(1.0, 1000.0)` kg on the 10 kg cart asset -- the range the mc_mujoco
sweep characterised -- drawn per episode by a `reset`-mode `dr.pseudo_inertia`
event on the cart body. Its `alpha` is a *log* scale (mass and inertia both scale
by `e^(2a)`), so `mdp.events.mass_alpha_range` converts kilograms to it and a
uniform draw in `alpha` is log-uniform in mass, the spacing the sweep used.
`dr.body_mass` is the wrong term: it leaves inertia behind.

**It needs mujoco-warp >= 3.11.** On 3.10.0.2 -- the version the PyPI mjlab 1.6.0
resolved -- any event declaring `RecomputeLevel.set_const_0` or above empties
`data.sensordata` from five environments up: 105 of 280 values at five, 0 of 448
at eight. The controller then sees no force or IMU feedback, its stabilizer has
nothing, and the robot falls after about 3.8 s, which reads as a policy failure.
Measured 2026-09-16 with a no-op event that declares only the level, so nothing
but the recompute differed:

| recompute level | recomputes | sensors, mjwarp 3.10.0.2 | sensors, mjwarp 3.11.0 |
| --- | --- | --- | --- |
| `none` | -- | 384 of 448 | 384 of 448 |
| `set_const_fixed` | `body_subtreemass` | 384 of 448 | 384 of 448 |
| `set_const_0` | `dof_invweight0`, `body_invweight0`, `tendon_*0` | **0 of 448** | 384 of 448 |
| `set_const` | both of the above | **0 of 448** | 384 of 448 |

It was never this task's scene: on 3.10.0.2 a `dr.body_mass` term added to
*residual balance*, one entity and no cart, broke identically at 8 environments.
Nor was it the contact budget (`nconmax` 100 to 2000, `njmax` 1500 to 40000
change nothing), CUDA graph capture, the event's mode, or the `cart_floor` pair.

**Re-measure if:** the mujoco-warp pin moves below 3.11, or `pseudo_inertia`
changes what `alpha` scales.

**History:**
- 2026-09-16 -- added and defaulted to the swept range; turned off when a smoke
  run showed `zmp_grounded` pinned at 0 and episodes ending at 3.7 s; carried a
  `set_const_fixed` term that rescaled the cart's own reference weights by hand
  until mujoco-warp 3.11 made the workaround unnecessary, and it was dropped.

## ZMP_TRACKING_STD

**Current:** `0.05` m, the Gaussian kernel width of the `zmp_tracking` reward,
which scores the distance between the measured centre of pressure and the ZMP the
controller planned, and scores **zero** on any step whose feet carry less than
`min_normal_force` (20 N) -- an unloaded foot has no centre of pressure, and
without the mask a robot in flight would collect the maximum.

Read the companion metrics as a pair, `zmp_error / zmp_grounded`: `zmp_error`
alone averages over every step, so it *falls* when the robot spends more time off
the ground.

**Not measured for this task**, but it does discriminate: on one zero-residual
HRP5P episode with a 112 kg cart, the kernel read 0.9995-0.9998 through the
approach and dropped to 0.745 once the push loaded the feet. The width still has
to be placed against the error distribution *during the push*, over episodes and
masses, which has not been taken.

**Re-measure if:** the robot, the cart or the foot force sensors change.

**History:** chosen as a round number, half of `residual_balance`'s `DCM_STD`.

## OBJECT_HISTORY

**Current:** `20` frames, 0.4 s at the 50 Hz policy rate, on the object's
velocity and tracking error and on all four force sensors -- the terms that carry
evidence about the payload. The manipulation phase, the completion flag and the
proprioception terms stay single-frame.

**Why any history at all:** the task randomizes the cart's mass and never tells
the controller or the policy what it is, and one frame of force and velocity
cannot separate a heavy cart from a stuck one. 0.4 s matches
`residual_balance`'s `CONTROLLER_HISTORY`, and covers both the hand-wrench
filter's own 0.1 s constant and a footstep.

**Cost:** the actor grows from 175 to 536 dimensions, the critic to 657, since
`flatten_history_dim` defaults to true and each term flattens term-major.

**Re-measure if:** the policy rate changes, or an ablation shows a shorter window
identifies the payload as well.

**History:** added with the payload randomization; without it the mass is not
identifiable from the observation at all.

## task_success

**Current:** the metric the comparison scores, and the one
`tasks/locomanip/evaluation.py` declares as its verdict. It is the FSM's
`Locomanip::complete` **and** the cart within `SUCCESS_POSITION_TOLERANCE_M`
(0.15 m) of its commanded position, **and** within `SUCCESS_YAW_TOLERANCE_RAD`
(10 degrees) of its commanded yaw, **and** no controller or worker failure, and
no fall termination on the step. Those tolerances are
`verify_locomanip.py`'s own acceptance gates, so the training verdict and the
acceptance run agree on what "done" means.

**Why `task_complete` alone is not success.** `Locomanip::complete` reports that
the executor reached `LMC::DemoHold`, and `ConfigManipState` leaves the push
phase when its waypoint queue empties -- on a schedule, not on arrival. A cart
that never moved completes exactly like one that arrived.

Measured 2026-09-16, two zero-residual environments at the nominal 10 kg cart:
both reported `complete = 1` at t = 47.6 s with the cart 0.154 m and 0.144 m from
its commanded position, landing either side of the tolerance, so
`task_success` read 0 and 1. Against a 400 kg cart the robot fell at t = 18.5 s
with the cart 0.29-0.33 m out, where completion would never have fired at all.

`task_complete` stays in the metrics beside it: the gap between the two is how
often the schedule outran the cart.

**Re-measure if:** the FSM's waypoint durations change, or the acceptance
script's tolerances move.

**History:** added when a review pointed out that the evaluation spec was scoring
the controller's own schedule rather than where the cart ended up.

## OBJECT_POSE_NOISE

**Current:** `OBJECT_POSE_NOISE_M` 0.02 m and `OBJECT_YAW_NOISE_RAD` 2 degrees,
uniform, on the actor's `object_pose`, `object_velocity`,
`object_position_error` and `object_yaw_error`. The proprioception terms carry
`residual_balance`'s own levels, which were measured on this robot
(docs/observations.md): base linear velocity 0.02, angular 0.03, projected
gravity 0.05, joint position 0.01 with `biased=True` behind an `encoder_bias`
startup event, joint velocity 0.05. The force sensors carry none -- they are
already real measurements. The actor group sets `enable_corruption=True`; without
it every `noise=` on the cfg is inert.

**Why the object terms need any at all:** they are simulator truth. Nothing on
the robot measures where the cart is; on hardware that pose comes from vision or
motion capture, with centimetre error and latency, and a policy trained on a
perfect 50 Hz pose with 0.4 s of history will lean on precision it will not have.

**Not measured.** No tracker has been characterised for this cart; 2 cm and 2
degrees is a deliberately loose stand-in. Replace it with the real sensor's error
before claiming anything about transfer.

**The critic keeps the clean copy.** Its terms are the actor's with `noise=None`
and the encoder bias dropped, so the value function sees the true state while the
policy pays for the noise.

**Re-measure if:** an actual pose source is chosen, or the robot's own noise
levels are re-measured.

**History:** added when the actor turned out to be training on clean simulator
state -- no noise specs and `enable_corruption` false on both groups.
