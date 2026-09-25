# Locomanip cart demo

## per-robot-controller-config

**Current:** the robot-specific half of Locomanip's configuration is the
*controller's* to hold, not this repo's. mc_rtc loads it for free:
`MCController::robot_config()` reads
`<the directory the controller library loads from>/<controller name>/<robot
MODULE name>.yaml` and `~/.config/mc_rtc/controllers/<controller
name>/<module>.yaml` into `config("robots")(<module>)`, and
`BaselineWalkingController`'s constructor then does `config().load(rconfig)`,
promoting it to the root before `OverwriteConfigList` is applied. So the
precedence is: base YAML, then the per-robot file, then this repo's
`MjlabCartDemo` overwrite.

| Robot | Where its Locomanip configuration lives |
| --- | --- |
| HRP5P | `LocomanipController/etc/robots/hrp5_p.yaml`, installed by `install_controller_robot_configuration` |
| JVRC1 | the `robots: jvrc1:` section inside `LocomanipController.yaml` itself |

The controller owns both, which is how `mc_logistic_controller` does it -- its
`src/controller/etc/robots/{hrp5_p,rhps1}.yaml` install into each
`LogisticController_<variant>/` the same way.

**`mc_hrp5_p` installs to the same path and only one file is ever read.** Its
`etc/controllers/LocomanipController/hrp5_p.yaml` lands on exactly
`lib/mc_controller/LocomanipController/hrp5_p.yaml`, so whichever package
installs last wins. LocomanipController's copy is therefore the *complete*
configuration rather than a supplement -- it carries `mc_hrp5_p`'s posture
weights, CoM active joints, base frame, foot timings and `refComZ: 0.95`
verbatim, so either install leaves HRP5P configured the same. Reinstalling
`mc_hrp5_p` drops only the three Locomanip keys; re-run the targeted install
below to put them back.

**The filename is the robot MODULE name, not `MainRobot`.** `hrp5_p` and
`jvrc1`, never `HRP5P`/`JVRC1` -- and a file under the wrong name is simply not
read, with nothing logged. BWC's own check is the only tripwire: it
`error_and_throw`s when `robots/<module>` is empty, so a robot with neither a
per-robot file nor a section in the base YAML fails at construction rather than
running mis-configured.

**Both files are provenance inputs**, and
`rl/controller_provenance.py` globs `<lib>/<controller>/*.yaml` beside
`<lib>/etc/<controller>.yaml` to catch them. A checkpoint from before
2026-09-18 did not record the per-robot file, so its signature cannot see a
change to it.

What stays in this repo is only what mc_rtc cannot know: which profile to run
(`etc/mc_rtc_{hrp5,jvrc1}_locomanip_patched.yaml`, differing in `MainRobot`
alone), and `tasks/locomanip/profiles.py`'s two scene values -- where the cart
stands and which MuJoCo bodies are the hands.

**Re-measure if:** BWC stops promoting `robots/<module>` to the root, or a
robot package starts shipping a Locomanip section that conflicts with
`MjlabCartDemo`.

**History:** 2026-09-18 -- moved HRP5P's `HandTaskList`, `objToHandTranss` and
`preReachTranss` out of this repo's profile and into the controller. Three
wrong turns on the way, each worth not repeating: per-robot *branches* of this
repo; a patch adding `ControllerParameters::overwrite_config` to
LocomanipController, written and reverted once BWC's existing promotion was
found; and parking the keys in `mc_hrp5_p`, which works but puts a controller's
configuration in a robot package.

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

## LocomanipController dependency

**Current:** the demo needs features upstream Locomanip does not have:
`ManipManager.enableRos`, numeric object/phase/completion datastore callbacks,
an opt-in `DemoFSM` reload, the `HandWrenchAdaptation` block, and the
initial-gripper retry that HRP5P needs to grasp at all. They live on the `ros2`
branch of the LocomanipController checkout, which also installs
`etc/robots/hrp5_p.yaml` (#per-robot-controller-config). Build that checkout and
install the rebuilt libraries into the sourced workspace. The repository profile
uses these features without changing the installed controller YAML.

```sh
cd ~/workspace/src/catkin_locomanip_ws/src/LocomanipController
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
install -D -m 644 etc/robots/hrp5_p.yaml \
  "$HOME/workspace/install/lib/mc_controller/LocomanipController/hrp5_p.yaml"
```

These targeted installs leave the machine-wide `LocomanipController.yaml`
untouched. A broad `cmake --install` would regenerate and overwrite it.

**Re-measure if:** Locomanip gains native non-ROS object input, its FSM is built
after overwrite processing, or equivalent datastore callbacks land upstream.

**History:** carried as `patches/locomanip-unattended.patch` until 2026-09-18,
when the same changes were committed to the checkout and the patch deleted. ROS
behavior is retained by default: the demo sets `enableRos: false`, preventing
node creation, subscription setup, and spinning.

## cart_init_x

**Current:** per robot, on `profiles.LocomanipProfile` -- `0.90` m for HRP5P and
`0.75` m for JVRC1, the cart body's initial x in front of the robot's stance.
`0.75` is Locomanip's own `robots.obj.init_pos`, which JVRC1's arms were sized
against; HRP5P has the longer reach.

**The number that matters is not this one.** `Cart.xml` puts the grasped handle
bar at cart-local `x = -0.35, z = 1.0`, matching the controller's
`ManipManager.objToHandTranss` of `[-0.35, +/-0.3, 1.0]`. So the hands reach to
`cart_init_x - 0.35`, while the cart box's rear face — the part that looks like
"the cart" — starts at `cart_init_x` itself and runs 0.85 m forward from there
(box half-size 0.425 centred at local `x = 0.425`).

At `0.75` the handle sits **0.40 m** in front of the robot and the box rear face
at 0.75 m. On HRP5P that read as the cart crowding it, so `0.90` puts the handle
at **0.55 m** instead. JVRC1 is the smaller robot and keeps the original
spacing, unmeasured until its acceptance run.

**This is a free choice, not a value the controller pins.** The environment
feeds the measured cart pose to mc_rtc through `controller_objects = {obj:
cart}`, so the controller grasps wherever the cart actually is; it does not
assume an initial pose. Only reachability bounds it — the hands must still make
`[-0.35, +/-0.3, 1.0]` in the object frame from the initial stance.

**Re-measure if:** the cart asset changes, `objToHandTranss` changes, or the
robot's initial stance moves.

**History:**
- 2026-09-18 — became per-robot rather than one constant, when JVRC1 registered
  alongside HRP5P.
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

**Current:** `(1.0, 1000.0)` kg -- the range the mc_mujoco sweep characterised --
drawn per episode by a `reset`-mode `dr.pseudo_inertia` event on the cart body.
Its `alpha` is a *log* scale (mass and inertia both scale by `e^(2a)`), so
`mdp.events.mass_alpha_range` converts kilograms to it and a uniform draw in
`alpha` is log-uniform in mass, the spacing the sweep used. `dr.body_mass` is the
wrong term: it leaves inertia behind.

**The range is absolute kilograms, not multiples of the asset.** `alpha` scales
whatever mass the compiled model has, so the conversion needs that mass and
`cart.cart_nominal_mass_kg()` reads it from the asset rather than assuming the
nominal 10 kg. That asset is `Cart.xml` in the LocomanipController *source tree*,
which the mc_mujoco sweep harness (`testing_harness.py`) rewrites in place to
sweep mass -- so it is not reliably 10 kg between runs, and a hardcoded nominal
silently multiplies every draw by the leftover.

What this does **not** fix is the controller's own belief: mc_rtc reads
`Cart.urdf` beside it, and the harness rewrites that too. A leftover there means
the base controller knows the payload, which is the opposite of the nominal
feedforward this task measures against. Check both are at 10 kg
(`git -C <LocomanipController> status`) before a run.

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
- 2026-09-17 -- read the nominal from the asset instead of a `10.0` constant,
  after a sweep run left `Cart.xml` and `Cart.urdf` at 350 kg: every draw would
  have been 35x its intended mass.
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

## locomanip reward weights

**Current:** `object_tracking` 1.0, `zmp_tracking` **0.5**, `termination_penalty`
-200.0, `upright` -2.0, `residual_magnitude` and `residual_rate` **-0.3**.
The two bold values changed on 2026-09-25 and are **hypotheses, not
measurements** -- they are the two levers run
[2026-09-24_17-38-11](#run-2026-09-24_17-38-11_hrp5p-feedback) left open.

**Why `zmp_tracking` came down.** Its per-episode share measured against the
zero-residual arm was 31.2 of a 43.5 total, against 14.4 for `object_tracking` --
and the policy moved it by +3.3% per step at p = 0.12, i.e. not at all. The term
the policy does move was carrying under a third of the signal. Equal kernel
widths do not make equal contributions: `zmp_tracking` is satisfied on almost
every grounded step while `object_tracking` is gated on both hands holding.

**Why the request penalties went up.** Over the same run `feedback_force_rms`
grew 3 N -> 13 N between iterations 600 and 1800 with `task_success` flat, so the
policy bought effort that returned nothing, and -0.1 priced that at -3.6 per
episode against a ~40 total. The weight is dense and bounded -- the request is
clamped into [-1, 1] before it is charged
(reward-shaping.md#requested_action_l2) -- so raising it cannot create the
outlier-sample failure that -2000 created for `termination_penalty`.

**Three neighbouring levers were left alone deliberately**, each already closed
by a measurement in this repo:

| lever | why not |
| --- | --- |
| `termination_penalty` past -200 | -2000 was tried: `Loss/value` never converged and the LR pinned to its floor for 20899 iterations (reward-shaping.md#termination_penalty) |
| `std_range` ceiling | 0.30 was tried and replaced by 0.15 precisely because std climbed 0.2 -> 0.52 unchecked (ppo.md#std_range); the pin at 0.15 is the ceiling working, not a ceiling that is too low |
| `entropy_coef` | already cut tenfold to 5e-5 on the argument that exploration noise is itself a disturbance on a residual task (ppo.md#entropy_coef) |

A sparse completion bonus was considered and rejected for a fourth reason: at
`gamma = 0.997` the credit horizon is 6.7 s, so a payment at the end of a 52 s
cycle is invisible from most of the episode.

**Re-measure if:** anything above is trained. Both changed weights are single
untested numbers, and the point of changing exactly two is that the next run's
delta is attributable.

**History:** 2026-09-25 -- halved `zmp_tracking` and tripled the two request
penalties after the first full feedback run plateaued for 1200 iterations while
its force effort quadrupled.

## NUM_ENVS

**Current:** `256` environments over `NUM_WORKERS` 64 mc_rtc worker processes,
measured on this machine 2026-09-16 over a 30-iteration run:

| | 8 envs | 256 envs |
| --- | --- | --- |
| throughput | 240-330 steps/s | 850-1650, mean ~1250 |
| iteration time | ~1.6 s | 11-19 s |
| resident memory | -- | 36 of 59 GB, flat, freed at exit |
| worker failures | 0 | 0 |
| controller (QP) failures | 0 | 0, bar one iteration at 0.118 |

That is 85 MB per environment and about five times the throughput for
thirty-two times the environments: the controllers are the bottleneck, not the
GPU. 23 GB stayed free, which is *not* enough to run `compare_to_baseline`
beside a training run -- it wants its own controller pool.

At this rate 2000 iterations is about 7 hours (32.8M steps, ~42 episodes per
environment).

**The iteration time does not track the fall rate.** 0.40 of ended episodes
falling took 13.3 s, 1.00 took 14.1 s, 0.55 took 19.1 s, so the 11-19 s spread is
something else; it has not been chased down.

**Re-measure if:** the machine, the worker count, the cart asset or the
controller build changes.

**History:** raised from 32 to 256 before the first capacity run; the run
confirmed it fits.

## HandWrenchAdaptation

**Current:** a config block on the push state, added by the LocomanipController
checkout, over what used to be hardcoded in
`ConfigManipState`'s push phase. On HRP5P, in mc_mujoco:

```yaml
HandWrenchAdaptation:
  enable: true
  alpha: 0.06                 # tau = dt / alpha = 33 ms at HRP5P's 2 ms period
  forceProjection: [1, 1, 1]  # the whole measured force, couple dropped
  momentProjection: [0, 0, 0]
```

The blend is `target <- alpha * measured + (1 - alpha) * target`, so its fixed
point is the measured wrench and `alpha` is a low-pass coefficient. The projection
sets the ceiling and `alpha` tunes within it: on a projection that already caps at
350 kg, every `alpha` from 0.003 to 0.05 caps there too, which is why an earlier
revision of this section claimed `alpha` moves no envelope. On `[1,1,1]` it moves
100 kg, and it is a band rather than a slope -- too slow lags the load, too fast
passes the measurement noise into the reference:

| alpha | tau | 600 kg | 700 kg |
| --- | --- | --- | --- |
| 0.001 | 2.0 s | fails | -- |
| 0.003 | 0.67 s | fails | -- |
| 0.005 to 0.04 | 0.4 to 0.05 s | 0.01 and 0.03 complete | fails |
| 0.05 | 40 ms | completes | completes |
| **0.06** | **33 ms** | -- | **completes, and 725 kg** |
| 0.07 | 29 ms | -- | completes |
| 0.085 | 24 ms | -- | fails |
| 0.1 | 20 ms | fails | fails |

The passing values form one plateau, 0.05 to 0.07, with 0.04 and 0.085 outside
it; 0.06 sits in the middle and is the only value that also takes 725 kg. Values
an order of magnitude slower complete 600 kg but nothing above it. A cell repeats
its verdict, so these are the boundaries and not sampling noise.

Both bound the result and neither substitutes for the other: at 500 kg with
`alpha` 0.01 or 0.03, push-only and the shipped projection still fail.

`alpha` against the reference's slew rate, one 350 kg run each, 2026-09-17:

| alpha | tau | slew rms | travel | final lag |
| --- | --- | --- | --- | --- |
| 0.003 | 0.67 s | 33.5 N/s | 0.992 m | 0.008 m |
| 0.008 | 0.25 s | 86.2 N/s | 0.994 m | 0.006 m |
| 0.02 | 0.10 s | 207 N/s | -- | -- |
| 0.05 | 0.04 s | 486 N/s | -- | -- |

JVRC1 at 281 kg runs 36.9 N/s, which 0.003 matches at no cost in travel or
tracking. Note `alpha` is per control period, and HRP5P's is 2 ms against JVRC1's
5 ms, so the shipped 0.02 filters 2.5x less on HRP5P than on the robot it was
tuned on.

**The projection is in the hand surface frame, which is robot-specific.** The two
robots' right-hand frames are rolled 180 degrees apart, measured at the push:

| local axis | JVRC1 | HRP5P |
| --- | --- | --- |
| x | +y world (lateral) | +y world (lateral) |
| y | +z world (up) | -z world (down) |
| z | +x world (forward) | -x world (backward) |

So `z` is the push axis on both and `y` the vertical on both, which is why the
shipped `(1,0,1)` keeps the push and drops the vertical on either robot. Keeping
*only* the vertical (`[0,1,0]`) completes 400, 500 and 600 kg where every
push-keeping setting fails at 400 -- but that reference points straight down in
the viewer: it commands a pull on the handle rather than estimating the push, so
it buys the envelope by leaning on the cart. It is recorded here as a measurement,
not a recommendation.

**Envelope, mc_mujoco, HRP5P, zero `preHandWrenches`, one run per cell:**

| setting | completes | fails |
| --- | --- | --- |
| shipped `(1,0,1)`, alpha 0.02 | 350 kg | 400 kg |
| `[0,0,1]` push only, alpha 0.003 / 0.008 / 0.02 / 0.05 | 350 kg | 400 kg |
| `[1,0,1]` push and lateral, alpha 0.003 | -- | 400 kg |
| `[1,1,1]`, alpha 0.003 | 500 kg | 600 kg |
| `[1,1,1]`, alpha 0.01 or 0.03 | 600 kg | 700 kg |
| **`[1,1,1]`, alpha 0.06** | **725 kg** | 750 kg |
| `[1,1,0]` and `[0,1,1]`, alpha 0.003 | 500 kg | -- |
| `[0,1,0]` vertical only, alpha 0.008 | 600 kg | -- |
| `[1,1,0]` in **world** (horizontal plane), alpha 0.003 | -- | 400 kg |

The pattern across projections is the hand-frame vertical term: every setting
that keeps it reaches 500 kg or more, every setting that drops it -- push only,
the shipped projection, the world horizontal plane -- caps at 350 to 400 whatever
`alpha` does.

Whatever is projected away is a load the stabiliser is not told about, and on
HRP5P the discarded terms are large: 163 N of net lateral force and, at 500 kg,
a reference of `(+42, -25, -140) N` in world -- the push is real but the lean
dominates, because that is what a humanoid pushing half a tonne actually does.
Dropping the vertical costs 150 kg of envelope; keeping only the vertical buys
100 more but commands a pure downward pull rather than any estimate of the push.

**Sharing one estimate between the hands changes nothing.** `shared: true`
averages the measured force in world -- the hand frames are mirrored, so
averaging in either frame would cancel the push -- filters once and splits it
evenly. Envelope, travel and the hands' own disagreement are unmoved: at 350 kg
travel is 0.995 m against 0.992, and the fore-aft disagreement between the hands
is 35.5 N against 33.8 N. The asymmetry is in the contact, not in the references
drifting apart.

**`preHandWrenches` must stay zero.** With the blend running, the push reference
*is* the blend's output; a feed-forward on top of it is what the original
`testing_harness.py` writes per mass (`10 * mass * friction / 2` per hand), and
mixing the two drops JVRC1 at 300 kg (travel -1.177 m, base z 0.087) where the
zero-feed-forward run does +1.054 m at z 0.720.

**The envelope has no hole at the light end.** At `alpha` 0.06 the cart completes
from 10 kg to 725 kg, and the object's final tracking error grows smoothly with
the load rather than jumping at a boundary:

| cart | 10 | 100 | 300 | 500 | 700 | 725 |
| --- | --- | --- | --- | --- | --- | --- |
| travel [m] | 1.109 | 1.047 | 0.982 | 0.911 | 0.829 | 0.814 |
| final lag [m] | -0.109 | -0.047 | +0.018 | +0.089 | +0.171 | +0.186 |

Base height holds at 0.712 to 0.719 across all of them, so the failure at 750 kg
is not the end of a slow collapse.

**Telling the controller the true mass buys nothing; measuring the force is worth
3x.** A feed-forward `preHandWrenches` is only an initial condition once the blend
runs -- the blend overwrites the target wrench every cycle, so it decays with
`tau` -- which means an oracle can only be read with `enable: false`:

| setting | pushes | fails |
| --- | --- | --- |
| blend off, no feed-forward | 300 kg | 350 kg |
| blend off, true mass as `preHandWrenches` on the push axis | 300 kg | 350 kg |
| blend off, true mass on the harness's lateral axis | 300 kg | 350 kg |
| blend at `alpha` 0.06, told nothing | 725 kg | 750 kg |

The cart model's mass is equally irrelevant: pinned at 10 kg or set to the true
mass, the blend stops at the same 725 kg. So the headroom is not an estimation
problem, and nothing that only sharpens the load estimate will move it.

**A `QP failed to run()` is not by itself a load failure.** With the blend off at
300 kg the robot pushes the cart 0.997 m of the 1.000 m reference at a steady
0.723 m base height, then topples in `Free` *after* releasing the handle -- the
harness scores that as `qp_failed` like a collapse under load. Score a cell by
what the object did before the hands let go: cut the log at the last `Hold`
sample, compare travel against the reference there, and check the base height over
that window. Runs that fail under load look nothing like it -- at 400 kg the same
setting drags the cart 0.253 m backwards with the base at 0.437 m. The blend's
runs do not fall after release at any mass it completes.

**The feed-forward's axes are not mirrored the way the harness assumes.** Measured
on HRP5P at the push, local `z` points along world `-x` for *both* hands while `x`
points at world `+y` on the right and `-y` on the left:

| | local x | local y | local z |
| --- | --- | --- | --- |
| Right | +y world | down | -x world |
| Left | -y world | up | -x world |

So a push needs the same sign on both hands' `z`, not opposite signs, and the
original `testing_harness.py`, which writes `force[0]` as `-value` on the left and
`+value` on the right, produces `2 * value` of net *lateral* force rather than a
squeeze or a push. That is the 163 N of net lateral measured at the hands.

**Re-measure if:** the hand frames, the hand impedance gains, the control period
or the robot changes.

**History:** the envelope was 275 kg for every setting until the grippers were
fixed (LocomanipController `7417204`): mc_rtc's gripper safety latched on the
startup transient, abandoned the opening 8 degrees from the stance, and left the
thumb closing on nothing, so the hands pushed with an empty fist. That fix alone
took the shipped setting from 275 kg to 350. An earlier revision of this section
blamed the vertical feedback for destabilising HRP5P and reported a fall at
100 kg; those runs carried a stale +-30 N `preHandWrenches` and a broken grasp,
and none of their numbers survive.

## ResidualFeedbackJointPositionAction

**Current:** the action is the shared residual-feedback one, in its
feedback-only, position-controlled form -- see
[docs/residual-feedback.md](residual-feedback.md#feedback_modalities) for the
modality itself. Locomanip configures it with `feedback_modalities=("wrench",)`,
`wrench_force_only`, its two hands in `wrench_sensor_names`, and a gate of
`Locomanip::leftPhase`/`Locomanip::rightPhase` at `Hold`. The action knows none
of those words; the measurements below are locomanip's, which is why they
live here.

The `position-feedback` registrations for HRP5P and JVRC1 therefore expose
exactly six policy outputs: left `Fx, Fy, Fz`, then right `Fx, Fy, Fz`.
`force_sensor_names` resolves those hands by name, independently of their native
layout order. Each action is clamped to `[-1, 1]` and multiplied by
`force_scale` (initially 50 N). The coordinates and sign are those of MuJoCo's
sensor input: the existing native conversion negates both force and moment to
produce mc_rtc's reaction wrench. These are virtual measurements, not forces
applied to the cart. The original sensor readings remain available to the actor,
critic, rewards and physical metrics.

The action composes `ResidualFeedbackActionBase` over the position action with an
empty joint residual, so the feedback block is its whole action. Its bridge correction affects only
the hands' force triples; measured moments, feet, IMU, encoders, cart state, real
PD gains and joint-target interpolation retain their existing path. No walking
reference or MPC torque blend is introduced. The controller's existing
hand-force adaptation remains enabled.

After each collection, the next dispatch enables the correction only when the
phase outputs are fresh, both hands report `Hold`, and neither controller nor
worker failure has latched. Resets clear that row's correction and action
histories; fresh phase outputs are required to enable it again. Dispatch and
collection retain the one-control-period delay. The action observation is the
six **normalized dispatched corrections**, while the magnitude and rate costs
use all six bounded requests and the phase gate. Rate cost excludes onset from
an unauthorized preceding step. `feedback_active`, `feedback_force_rms` (N), and
`feedback_saturation` report the last dispatch; play prints the six corrections
in N using the usual residual-print interval.

The environment reuses the position-residual task's objective, noise/history,
payload distribution, terminations and PPO configuration. Experiment names are
the distinct full task IDs. The action configuration records sensor order and
force scale in the checkpoint interface, so changing either rejects a load.
Existing position-residual and walking-feedback checkpoints keep their original
interfaces.

What was checked on 2026-09-18, all under
`logs/locomanip/feedback-validation/`. `scripts/smoke_locomanip_feedback.py`
took each robot to Hold at zero feedback, ran two PPO updates and reloaded the
checkpoint into a fresh runner: six action dimensions, 507/628 observations on
HRP5P and 489/610 on JVRC1, finite gradients and losses, bit-exact reload.
`scripts/verify_locomanip.py --task position` and `--task feedback` ran the same
cycle at a fixed 10 kg cart with no encoder bias: HRP5P's two runs differ by
0.6 mm of cart travel and 4 mm laterally, and both miss the demo's 0.15 m
position gate by the same hair (0.1506 m against 0.1518 m). JVRC1 never finishes
the cycle in either task: it drops to 0.577 m and terminates at 19.04 s under the
position residual and 20.08 s under feedback, failing the same three checks. The
demo separates the two causes -- `--task demo` on the same day held JVRC1 upright
at 0.694 m through a complete cycle, so the early fall belongs to the
position-residual environment rather than to JVRC1 or to the feedback path, and
wants its own investigation before either task is trained on that robot. That
demo run misses the position gate too, by 0.779 m: the 0.15 m tolerance was
calibrated on HRP5P, which passed the same run at 0.1432 m.
Authority is in [AuthorityTrace](#authoritytrace).

**Re-measure if:** the hand sensors, controller impedance gains, adaptation,
phase transitions, force scale, robot assets or dispatch timing change.

**History:** 2026-09-18 -- added feedback-only force policies for both robot
profiles. The initial 50 N bound is experimental, not a demonstrated optimal
training scale; full policy training is separate from the implementation checks.
At 50 N every channel clears the zero-control floor in every measured regime,
and the vertical one both rescues a payload the controller loses and, at the
other sign, drops the robot -- so a policy has room to help and to fall. 5 N does
not clear the floor; see [AuthorityTrace](#authoritytrace) for the numbers.

## AuthorityTrace

**Current:** `scripts/probe_locomanip_authority.py --task feedback` excites each
selected hand-force component in both directions at normalized amplitudes 0.1,
0.5 and 1.0 (5, 25 and 50 N).
`--channel` selects one component; its default covers all six. Three zero rows
measure the numerical floor. All rows use the same fixed payload and no encoder
bias; actor noise never feeds these constant actions. Controller-target RMS
differences are measured only while both the comparison row and the first zero
row hold the cart in their first episode. A response above the largest zero-row
RMS plus 1e-6 rad is labelled measurable authority, which is not a claim of
better control.

The JSON retains each row's first episode, including termination, physical cart
error/travel, minimum height, contact forces, holding duration and dispatched
residual RMS. Failed rows are recorded before reset and then recycled with zero
actions so their controllers cannot wedge while other conditions finish.
`--task position` retains the all-joint constant-action screen, whose single
zero arm is also the response reference, so its floor is zero by construction
and `measurable_authority` only carries information under `--task feedback`.
Use `--robot` and `--mass` for the two profiles and the 10/300 kg comparison
regimes.

Measured 2026-09-18 at `force_scale` 50 N, 16 s per condition, seed 42, encoder
bias off, three zero controls per set and **three repeats of every set**, under
`logs/locomanip/feedback-validation/repeats/`. The cart is fixed at the named
mass: `--mass` collapses `alpha_range` to a point and every other
`pseudo_inertia` range is zero, so payload and inertia are identical in every row
of every repeat.

**The simulation does not reproduce bitwise.** Two runs of identical code, seed
and configuration diverge: the zero controls of one set span 0.004 m of cart
travel at HRP5P/10 kg, and the floor itself moves by up to a factor of two
between repeats. So the figures below are medians over three repeats with their
range, and "above floor" counts only the rows that cleared it in *all three*.

| robot | cart | zero floor | above floor | excited rows that fell | zero rows that fell |
| --- | --- | ---: | ---: | ---: | ---: |
| HRP5P | 10 kg | 0.0053-0.0078 rad | 29/36 | 3/36 | 0/9 |
| HRP5P | 300 kg | 0.0335-0.0611 rad | 26/36 | 2/36 | 0/9 |
| JVRC1 | 10 kg | 0.0055-0.0065 rad | 34/36 | 20/36 | 0/9 |
| JVRC1 | 300 kg | 0.0045-0.0267 rad | 32/36 | 31/36 | 9/9 |

**Amplitude decides, not channel.** All 24 channel-direction pairs clear the
floor at 50 N in all three repeats of all four regimes, and all but one
(`left_fy` at -25 N, HRP5P/300 kg) at 25 N. What is not resolvable is 5 N: 22 of
the 48 rows at that amplitude sit inside the floor in at least one repeat. No
channel is inert -- 5 N is simply below the controller's own spread.

**Vertical is the channel that moves the cart**, and its sign decides which way.
Left hand, median over three repeats with the range, and how many of the three
fell:

| regime | +50 N | -50 N |
| --- | --- | --- |
| HRP5P 10 kg | -0.851 m [-0.863, -0.844], 3/3 fell | +0.517 m [+0.464, +0.544], 0/3 |
| HRP5P 300 kg | -0.325 m [-0.326, -0.308], 3/3 fell | +0.258 m [+0.242, +0.296], 2/3 fell |
| JVRC1 10 kg | -0.466 m [-0.468, -0.466], 3/3 fell | +1.040 m [+1.033, +1.059], 3/3 fell |
| JVRC1 300 kg | -0.208 m [-0.208, -0.207], 3/3 fell | +0.596 m [+0.595, +0.604], 0/3 |

The right hand tracks it to within the repeat spread. `Fx` and `Fy` move the
controller's targets as much and the cart far less. Travel does not compare
across robots: the hands hold for 5.4 s of the 16 s window on HRP5P against
10.7 s on JVRC1.

JVRC1 at 300 kg is the regime that matters. **All nine zero controls fall inside
the window**, across all three repeats -- the unbiased controller loses that
payload -- while every -25 N and -50 N vertical row survives all three, at
0.70-0.72 m base height against 0.574 m for zero, with up to +0.617 m of cart
travel. That is a constant bias rather than a policy, but it is the strongest
evidence so far that these six channels reach the failure the task is about.

**Re-measure if:** the compared action, payload, zero-control variability or
observation/control conventions change.

**History:** 2026-09-18 -- added channel-isolated force conditions, repeated-zero
controls and terminal-safe records to the existing position authority probe. The
first write-up of these numbers quoted one run per regime and claimed 36/36
channels above the floor; three repeats put it at 26-34/36 and showed the
simulation is not reproducible run to run, so single-run counts overstated what
the probe can resolve.

## Run 2026-09-24_17-38-11_hrp5p-feedback

**The first full training run of the hand-force feedback task, and a negative
result.** 2000 iterations, 6 h 36 m, 256 envs over 64 workers, seed 42, HRP5P.
`model_1800.pt` is the only checkpoint worth keeping.

| iterations | success | complete | length | force RMS | fell_over | collapsed | ctrl-fail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-600 | 0 -> 0.6 | 0 -> 0.85 | 1558 -> 2750 | 0 -> 6 N | 0.031 | 0.099 | 0.037 |
| 600-1800 | 0.61-0.64 | 0.80-0.85 | ~2700 | 6 -> 13 N | 0.093 | 0.06-0.09 | 0.031 |
| 1804-1999 | 0.16 | 0.20 | 1488 | 11 N | 0.177 | **0.615** | 0.083 |

It learned the task in 600 iterations, plateaued for 1200, then collapsed at 1804
and never recovered over the remaining 196.

**The policy saturated its own action space.** `Loss/entropy` fell -3.2 (iteration
300-900) to -37 (1649) while `Policy/mean_std` stayed pinned at exactly 0.15, the
ceiling of `std_range`, from iteration ~300 onward. At fixed sigma a 6-D squashed
Gaussian can only reach that entropy through the tanh log-determinant, so the
pre-squash mean was driven deep into saturation. `feedback_saturation` (first
nonzero at iteration 843, rising to 0.05) and force RMS 3 N -> 13 N agree.

**The effort bought nothing.** Between iterations 600 and 1800 the RMS roughly
doubled with `task_success` and `task_complete` flat.

**What it was not.** Not the learning rate: the adaptive schedule held 2.6e-5 to
5.9e-5 from iteration ~1700 with no anomaly at 1804 (1803: 5.85e-5, 1805:
3.90e-5). Not cart mass or a curriculum: the per-200-iteration mean stayed
141-156 kg and the task has none. Not infrastructure:
`controller_worker_failed` was 0.0000 for all 2000 iterations. Not exploration
collapse: sigma was at its *maximum*.

**The critic went first.** `Loss/value` roughly doubled, 0.05-0.06 to 0.10-0.22,
starting around iteration 1797 -- before the behavioural collapse at 1804. That is
the documented precondition for this configuration's horizon: `gamma = 0.997`
leans on the critic, and is only affordable while the critic converges
(ppo.md#gamma). The proximate trigger is not identifiable beyond that, because
this task logged none of the `Diagnostics/` scalars that would have shown it --
see the wiring change below.

**Baseline comparison of `model_1800`** (144 vs 160 paired episodes,
`logs/comparisons/2026-09-24_17-38-11_model_1800.log`): it is worse than the
zero-residual controller.

| | baseline | policy | delta | p |
| --- | --- | --- | --- | --- |
| survival to cap | 88.2% | 65.6% | -22.6 pp | 3.8e-06 |
| `task_complete` | 0.882 | 0.656 | -0.226 | 1.1e-06 |
| reward/step total | 0.01522 | 0.01272 | -16.4% | 1.2e-07 |
| `object_position_error` | 0.1102 | 0.0629 | **-43%** | 2.2e-05 |
| `object_yaw_error` | 0.0575 | 0.0231 | **-60%** | 1.7e-05 |
| `task_success` | 54.2% | 57.5% | +3.3 pp | 0.56 |

**The mechanism has real authority; the trade is bad.** Cart tracking improves
decisively and survival pays for it. Stratified by mass, the policy helps
everywhere except above 300 kg (0.353 -> 0.182), on 17 and 33 episodes -- too few
to establish.

`object_position_tracking` is Hold-gated and the `object_position_error` metric is
not, so the reward moving one way while the metric moves the other is an
averaging difference, not a contradiction.

**What changed as a result:**

- every residual runner now logs the eleven `Diagnostics/` scalars, not just
  residual_balance (ppo.md#training-diagnostics). `explained_variance`,
  `action_saturation` and `policy_mean_saturation` are the three that would have
  named this failure while it was happening.
- locomanip moved to `RolloutAdaptivePPO` (ppo.md#RolloutAdaptivePPO), so the rate
  follows one full-rollout KL per iteration instead of four per-minibatch ones,
  and `schedule_kl` is logged.
- `EPISODE_LENGTH_S` 60 -> 70, and two reward weights moved
  (#locomanip-reward-weights).

**Re-measure if:** anything in that list is trained. None of it is validated; the
run that produced this section is the only evidence either way.
