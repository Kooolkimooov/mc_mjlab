# Locomanip cart demo

## per-robot-controller-config

**Current:** Locomanip's configuration is the *controller's* to hold, not this
repo's -- both the robot-specific half and the cart demo itself. mc_rtc loads
the robot-specific half for free: `MCController::robot_config()` reads
`<the directory the controller library loads from>/<controller name>/<robot
MODULE name>.yaml` and `~/.config/mc_rtc/controllers/<controller
name>/<module>.yaml`. LocomanipController constructs with
`ControllerParameters{}.load_robot_config_into({}).overwrite_config(true)`, as
`LogisticController` does, so `MCController`'s constructor merges that file at
the configuration **root** -- before `fsm::Controller` loads `states`, so a
`states:` entry in it merges into the state of the same name. The walking
configuration still sits under `robots: <module>:`, which
`BaselineWalkingController`'s constructor promotes to the root with
`config().load(rconfig)` before `OverwriteConfigList` is applied. So the
precedence is: base YAML, then the per-robot file, then its `robots/<module>`
promotion, then any overwrite.

| Robot | Walking configuration | State overrides |
| --- | --- | --- |
| HRP5P | `robots: hrp5_p:` in `LocomanipController/etc/robots/hrp5_p.yaml`, installed by `install_controller_robot_configuration` | `HandWrenchAdaptation` on `LMC::PushCart_`, same file |
| JVRC1 | `robots: jvrc1:` in `LocomanipController/etc/robots/jvrc1.yaml` | `HandWrenchAdaptation` on `LMC::PushCart_`, same file |

Each per-robot file also sets the cart's start, `robots: obj: init_pos` (#cart_init_x);
the base `LocomanipController.yaml` declares only the cart's module, so it carries
no robot-specific section at all.

The controller owns both, which is how `mc_logistic_controller` does it -- its
`src/controller/etc/robots/{hrp5_p,rhps1}.yaml` install into each
`LogisticController_<variant>/` the same way.

**`mc_hrp5_p` installs to the same path and only one file is ever read.** Its
`etc/controllers/LocomanipController/hrp5_p.yaml` lands on exactly
`lib/mc_controller/LocomanipController/hrp5_p.yaml`, so whichever package
installs last wins. LocomanipController's copy is therefore the *complete*
configuration rather than a supplement -- it carries `mc_hrp5_p`'s posture
weights, CoM active joints, base frame, foot timings and `refComZ: 0.95`
verbatim, under `robots: hrp5_p:`. `mc_hrp5_p`'s file is flat, so with the
root merge above, reinstalling `mc_hrp5_p` leaves `robots/hrp5_p` empty and
construction throws (next paragraph) rather than running without the Locomanip
keys; re-run the targeted install below to put them back.

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

**The project YAML cannot reach the FSM, which is why the demo moved too.**
`MCGlobalController::GlobalConfiguration::load_controllers_configs` seeds the
controller's configuration with the *project* `mc_rtc.yaml` and then loads the
installed `<lib>/etc/<controller>.yaml` **over** it, so a `states:` block in this
repo's profile is overwritten rather than applied. And `fsm::Controller` builds
the factory and executor in its own constructor, before BWC's constructor runs
`OverwriteConfigList` -- so an overwrite cannot reach the root FSM either. The
only project-side lever was LocomanipController's opt-in `DemoFSM` reload, which
is a second, complete FSM; keeping the demo there meant keeping a copy of the
controller's own states in this repo. The controller-side per-robot file *can*
reach states, through the root merge above.

What stays in this repo is only what mc_rtc cannot know: which profile to run
(`etc/mc_rtc_{hrp5,jvrc1}_locomanip_patched.yaml`, differing in `MainRobot`
alone), and `tasks/locomanip/profiles.py`'s two scene values -- where the cart
stands and which MuJoCo bodies are the hands.

`DemoFSM` survives as the probe's lever rather than the demo's:
`scripts/probe_hand_wrench_adaptation.py` reads the installed FSM back and
re-emits it as a `DemoFSM` with one state's `HandWrenchAdaptation` changed
(#HandWrenchAdaptation), which is the only way a script can vary a state config
without writing into the workspace.

**Re-measure if:** LocomanipController stops constructing with
`load_robot_config_into({})` and `overwrite_config(true)`, BWC stops promoting
`robots/<module>` to the root, `load_config` stops letting the installed file win
over the project one, a robot package starts shipping a Locomanip section that
conflicts, or the FSM is built after overwrite processing.

**History:** 2026-10-12 -- per-robot files merge at the root, so they can set
state configs. JVRC1's `DemoFSM` copy of the push state
(`LMC::PushCartJvrc1_` and its own transition map) became a three-key
`states: LMC::PushCart_:` override, and HRP5P's walking keys moved under
`robots: hrp5_p:` to keep BWC's non-empty check satisfied. Constructing both
robots through the Python bindings reads back HRP5P's `footstepDuration` 0.9,
`horizonDt` 0.002, `refComZ` 0.95 and `LeftHandCloseContact`, and JVRC1's
`HandWrenchAdaptation` on `LMC::PushCart_` with no `DemoFSM`. This is the
`overwrite_config` patch reverted on 2026-09-18 (below), re-adopted for what
BWC's promotion cannot do: reach `states`. Both per-robot files changed bytes,
so checkpoints recorded before this fail the provenance check;
`base_controller_config/` in each run directory holds the files they need.
One `verify_locomanip.py --cycles 1` demo per robot, 10 kg, seed 42: HRP5P
passes every gate (cart 0.1445 m, 1.204 m forward, z min 0.7226 m, 33.0 s
bilateral contact, 2.0 s hold); JVRC1 falls in Hold at 19.12 s (z min 0.5485 m)
against 20.08 s (0.5489 m) on the pre-change install, the knee-saturation fall of
#jvrc1-knee-effort-limits either way.

Later the same day the base `LocomanipController.yaml` lost its last
robot-specific keys: JVRC1's `robots: jvrc1:` section moved into `jvrc1.yaml`,
the cart's `init_pos` became per robot (#cart_init_x), and each robot's file
ships its own `HandWrenchAdaptation` -- HRP5P `alpha: 0.06` (#HandWrenchAdaptation),
JVRC1 `alpha: 0.008` (#jvrc1-hand-force-blend-in-mc_mjlab). JVRC1 at 0.008, one
demo cycle as above: upright (z min 0.6914 m), completes and holds 2.0 s with
33.0 s bilateral contact, and fails only `cart_position` -- 0.9007 m error after
walking 1.9553 m, the overshoot every stable alpha in that section shows. HRP5P,
one cycle each on the installed files: the compiled default passes every gate
(cart 0.1435 m, 1.2024 m forward, z min 0.7219 m); 0.06 completes, holds 2.0 s
with 33.0 s bilateral contact and z min 0.7078 m, and fails only `cart_position`
at 0.1588 m against the 0.15 m tolerance -- within the 7.8 mm run-to-run spread
#task_success records, so one run does not separate the two blends at 10 kg.

2026-09-28 -- moved the rest of the demo out: the `MjlabCartDemo`
overwrite is gone, and `ManipManager.enableRos: false`, `CompletionState:
LMC::Pause_`, the hour-long hold and HRP5P's `PreviewControlExtZmp` now sit in
the controller's own `etc/LocomanipController.in.yaml` and
`etc/robots/hrp5_p.yaml`. The overwrite's `StepByStep: false` was dropped as a
no-op: `Executor.cpp` fires an `Auto` transition regardless of `step_by_step_`,
and all three transitions are `Auto`.

Verified by `verify_locomanip.py --cycles 1` on both robots, each cycle completing
and holding its 2 s: JVRC1 reproduces its recorded 0.779 m position miss at
0.7801 m, and HRP5P's cart error is 0.14663 m before the move against 0.14917,
0.15088 and 0.15695 m after. The 7.8 mm spread across three identical post-move
runs is wider than the 4.3 mm between the configurations, so the gap is the
gate's own noise (#task_success) rather than the move.

2026-09-18 -- moved HRP5P's `HandTaskList`, `objToHandTranss` and
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
`CompletionState`, an opt-in `DemoFSM` reload, the `HandWrenchAdaptation` block,
and the initial-gripper retry that HRP5P needs to grasp at all. They live on the
`mc_mjlab-setup` branch of the LocomanipController checkout, which also carries
the cart demo's own configuration (#per-robot-controller-config): its FSM, the
push waypoint and the two robots' YAML. Build that checkout and install the
rebuilt libraries *and* both YAML files into the sourced workspace -- the
configuration is no longer something this repo's profile supplies.

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
install -D -m 644 etc/robots/jvrc1.yaml \
  "$HOME/workspace/install/lib/mc_controller/LocomanipController/jvrc1.yaml"
install -m 644 /tmp/locomanip-build/etc/LocomanipController.yaml \
  "$HOME/workspace/install/lib/mc_controller/etc/LocomanipController.yaml"
```

The last line is the demo's configuration, and `configure_file` writes it at the
*configure* step -- a YAML-only change needs no `cmake --build`. It overwrites the
machine-wide `LocomanipController.yaml` on purpose, which is what moving the demo
into the controller bought; a broad `cmake --install` now lands the same content.

**Beware `testing_harness_hrp5p.py`**, the checkout's own mass sweep: it rewrites
`etc/LocomanipController.in.yaml` with `yaml.safe_dump` to set the arm's blend, so
every comment in that file is stripped the first time a sweep runs.

**Re-measure if:** Locomanip gains native non-ROS object input, its FSM is built
after overwrite processing, or equivalent datastore callbacks land upstream.

**History:** carried as `patches/locomanip-unattended.patch` until 2026-09-18,
when the same changes were committed to the checkout and the patch deleted. ROS
behavior is retained by default: the demo sets `enableRos: false`, preventing
node creation, subscription setup, and spinning.

## cart_init_x

**Current:** per robot, on `profiles.LocomanipProfile` -- `0.90` m for HRP5P and
`0.75` m for JVRC1, the cart body's initial x in front of the robot's stance.
The controller carries the same values as `robots: obj: init_pos` in each
robot's `etc/robots/<module>.yaml`, which is where mc_mujoco spawns the cart, so
the two places must move together. `0.75` was Locomanip's original single
`init_pos`, which JVRC1's arms were sized against; HRP5P has the longer reach.

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
| `episode_length_s` | 60.0 | the installed FSM completes its cycle in 52.194 s |
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

## Retired object tracking width

**Retired:** `OBJECT_TRACKING_STD` was `0.04` m, the Gaussian kernel width of
`object_position_tracking`. Object tracking is no longer a reward.

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

**Current:** `(1.0, 700.0)` kg, drawn *flat in kilograms* per episode by a
`reset`-mode `dr.pseudo_inertia` event on the cart body. Its `alpha` is a *log*
scale (mass and inertia both scale by `e^(2a)`), so `mdp.events.mass_alpha_range`
converts kilograms to it and `mdp.events.FLAT_MASS` samples `e^(2a)` uniformly
rather than `alpha`. `dr.body_mass` is the wrong term: it leaves inertia behind.

**Why flat: the light carts the grasp shoves away (below) stay rare.** Over
1--700 kg a flat draw puts 1.3% of episodes under 10 kg and 7.0% under 50 kg; the
log-uniform draw the sweep's spacing implied put 35.1% and 59.7% there. Light-cart
grasping has not been revalidated with the current controller, so that 1.3% still
measures the grasp rather than the policy, and `safe_completion` can count them.

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

**Why not below 10 kg: Grasp shoves a light cart off the hands.** One
zero-action HRP5P position-task cycle per mass, seed 42, encoder bias removed,
episode cap lifted to 90 s (`scripts/verify_locomanip.py` trace), 2026-10-10:

| | 1.5 kg | 3 kg | 10 kg |
| --- | --- | --- | --- |
| Grasp -> Hold | 7.70 -> 11.70 s | 7.70 -> 10.18 s | 7.69 -> 10.63 s |
| cart x at Hold (starts at 0.90 m) | 1.82 m | 1.02 m | ~0.90 m |
| hand-cart contact during Hold | none | single hand, from ~26 s | both hands |
| Ungrasp -> Release | 44.70 -> 48.70 s | 43.20 -> 45.24 s | -> 46.16 s |
| `Locomanip::complete` | 50.20 s | 46.74 s | 47.66 s |

The controller completes on its own clocks -- a 30 s waypoint, and Grasp and
Ungrasp end on the gripper's `complete()` -- so a light cycle "completes" with
no push at all, and `task_success` passes when the shoved cart happens to stop
near the target (0.138 m at 1.5 kg). Under the 49.66 s cap the late completions
read as time-outs instead: `compare_to_baseline.py` on `model_1960` of run
2026-10-09_15-16-21 (64 envs, 30 min, randomized) timed out 95 baseline and 98
policy episodes, every one with a cart under 6.01 kg and both hands still out of
Free, and completed 0/49 and 2/58 below 2 kg. Neither arm is pushing there, so
that band measures the grasp, not the policy.

**Re-measure if:** the mujoco-warp pin moves below 3.11, `pseudo_inertia`
changes what `alpha` scales, or the grasp approach or gripper commands change.

**History:**
- 2026-10-19 -- range moved to `(1.0, 700.0)` and the draw made flat in kilograms;
  log-uniform over that range would have put 35% of episodes in the shoved band.
- 2026-10-10 -- lower bound raised from `1.0` to `10.0` after the traces above.
- 2026-09-17 -- read the nominal from the asset instead of a `10.0` constant,
  after a sweep run left `Cart.xml` and `Cart.urdf` at 350 kg: every draw would
  have been 35x its intended mass.
- 2026-09-16 -- added and defaulted to the swept range; turned off when a smoke
  run showed `zmp_grounded` pinned at 0 and episodes ending at 3.7 s; carried a
  `set_const_fixed` term that rescaled the cart's own reference weights by hand
  until mujoco-warp 3.11 made the workaround unnecessary, and it was dropped.

## JVRC1 hand-force blend in mc_mjlab

**Current:** the instability is caused by insufficient knee torque authority.
mc_mjlab caps both JVRC1 knees at the RobotModule's 100 Nm; mc_mujoco's PD
actuation is unclamped and its nominal cart gait needs up to 117.45 Nm. Removing
only the two knee caps in a diagnostic restores nominal stability. The recovered
blend therefore compares different actuator limits across the simulators. See
[JVRC1 knee effort limits](#jvrc1-knee-effort-limits) for the matched intervention,
loaded checks and remaining limits of the comparison. The earlier measurements
below retain the history of the investigation.

**The block, recovered from the controller's own history.** `git log -S
measuredWrench` on `ConfigManipState.cpp` reaches commit `e1f57c9` (2026-07-02
09:02), the revision current when `tests/test_logs_translate_force=auto` was
written (Jul 2 09:46). It hardcoded `double alpha = 0.1`, summed both hands
through a constant `diag(-1, -1, 1)` mirror, filtered once and halved, and forced
the couple to zero. That algebra reduces exactly to today's `sharedPush()`: with
`lh_target = M*prev` and `M*M = I` the expected sum is `2*prev`, so
`new = alpha*(measured_sum/2) + (1-alpha)*prev`. As configuration:

```yaml
HandWrenchAdaptation:
  enable: true
  shared: true
  alpha: 0.1
  forceProjection: [1, 1, 1]
  momentProjection: [0, 0, 0]
```

It lives in `LocomanipController/etc/robots/jvrc1.yaml` as a `states:
LMC::PushCart_:` override, which the per-robot file can carry since it merges at
the configuration root (#per-robot-controller-config). Until 2026-10-12 it was a
`DemoFSM` reload declaring `LMC::PushCartJvrc1_`.

**Shipped `alpha` is `0.008`, not the recovered `0.1`** (2026-10-12): the
original hardcoded `0.02`, tuned on JVRC1 at its 5 ms period, rescaled to
mc_mjlab's 2 ms so `dt / alpha` stays 0.25 s. `alpha` is per period, so under a
5 ms configuration the same value is 0.625 s.

**Under mc_mujoco it works**, coarse sweep, `--ref-mass 1.0` so the controller has
no payload knowledge, `LMC::Pause_` shortened from 3600 s to 3 s to keep each log
near 140 MB instead of 9.8 GB:

| mass | pos err | worst tau | z min | final state |
| --- | --- | --- | --- | --- |
| 1.0 kg | 0.003 | 1.14 | 0.720 | Pause |
| 100.0 kg | 0.018 | 1.51 | 0.718 | Pause |
| 215.4 kg | 0.006 | 1.63 | 0.706 | Pause |
| **464.2 kg** | **0.008** | **1.82** | **0.709** | Pause |
| 1000.0 kg | 1.315 | 18.60 | 0.094 | fail |

That matches the historical `translate_force=auto` arm (0.013 m at 494 kg, tau
1.75) and its failure signature (754 kg, tau 19.41).

**Under mc_mjlab with module effort limits it does not.** Same masses through
`verify_locomanip.py --task position --cart-mass`:

| mass | cart err | robot fwd | z min | contact | completed |
| --- | --- | --- | --- | --- | --- |
| 1.0 kg | 0.471 | +1.525 | 0.713 | 33.0 s | yes |
| 10.0 kg | 0.204 | +1.268 | 0.567 | 27.6 s | no |
| 100.0 kg | 0.254 | **-0.189** | 0.579 | 6.8 s | no |
| 215.4 kg | 0.273 | **-0.244** | 0.576 | 7.3 s | no |
| 464.2 kg | 0.290 | **-0.232** | 0.623 | 8.2 s | no |
| 1000.0 kg | 0.222 | -0.140 | 0.627 | 8.9 s | no |

Only 1 kg completes. From 100 kg the robot travels backwards and loses hand
contact after 7-9 s. Every loaded run sits at z 0.567-0.627 against JVRC1's
`0.7 * 0.828 = 0.579` collapse threshold, so the task's own height check ends the
episode; mc_mujoco holds 0.706-0.722 under the same load and has no such check.

**At 10 kg the config is the differentiator, and any single gain reduction fixes
it.** One gate cycle each, reverting one field of the block to its compiled
default while the rest stay:

| variant | cart err | fwd | z min | contact | upright |
| --- | --- | --- | --- | --- | --- |
| compiled defaults | 0.769 | 1.829 | 0.694 | 33.0 s | yes |
| full recovered block | 0.103 | 0.588 | 0.550 | 14.1 s | **no** |
| `shared: false` | 0.992 | 2.032 | 0.691 | 33.0 s | yes |
| `alpha: 0.02` | 0.858 | 1.891 | 0.689 | 33.0 s | yes |
| `forceProjection [1,0,1]` | 0.800 | 1.861 | 0.700 | 33.0 s | yes |
| `momentProjection [0,1,0]` | 0.099 | 0.942 | 0.549 | 23.7 s | **no** |

`shared`, `alpha` and `forceProjection` together set the loop gain; the moment
projection is irrelevant. The stability ceiling is between 0.07 and 0.10 -- every
alpha from 0.02 to 0.07 stays upright, 0.10 falls -- which mirrors the HRP5P
plateau in #HandWrenchAdaptation. But no alpha recovers the envelope: 0.07 and
0.02 both fail from 100 kg up, at the same z.

**The signature is an oscillation, not an inverted force.** Left hand `fy`
through Hold, medians over 0.5 s windows:

| t since Hold | mc_mjlab measured | mc_mujoco measured |
| --- | --- | --- |
| 0-2 s | 1.2, 3.5, 6.6 | 0.7, 3.3, 6.8 |
| 3 s | -8.4 | -8.6 |
| 4 s on | +3.1, -0.3, -6.4, +2.5, -2.6, +13.1 | -17.2, -19.1, -12.0, -19.1, -13.5, -19.0 |

Both agree for the first 3 s including the reversal. mc_mujoco then settles to a
steady -13 to -19 N; mc_mjlab's sign flips every window. An FFT over Hold +4-14 s
puts both at **1.00 Hz with a 2 Hz harmonic** -- the gait, since
`FootManager.footstepDuration: 0.9` -- at std **15.27 N** here against **5.68 N**
there. Same mode, three times the amplitude.

**Eliminated, each by measurement:**

- *Wrench sign or frame.* Signs and magnitudes agree at matched Hold windows. Both
  simulators copy MuJoCo sensordata verbatim (`mj_sim.cpp:758`,
  `sim_controller_bridge.py:294`); the negation to mc_rtc's convention happens
  natively for both.
- *A missing gain.* Foot sensors total 581.6 N here and 581.9 N there against
  JVRC1's 612.1 N -- ratio 0.950 and 0.951, the shortfall being mass below the
  ankle sites.
- *Hand sensor fidelity.* `|site sensor| / |contact sensor|` scatters 0.19-2.60
  during gait, which is wrist-sensor distal inertia, and mc_mujoco reads the same
  channel from the same MJCF.
- *Grasp orientation.* Hand position and surface z axis agree to three decimals;
  the quaternions differ only by an overall sign.
- *Control period.* mc_mjlab runs JVRC1 at 2 ms against the controller config's
  5 ms, and `alpha` is per period -- but forcing 5 ms (`Timestep 0.005`,
  `frameskip 5`, `decimation 5`) made it worse: fwd 0.036 m, still falling.
- *Dispatch lag.* Collecting each solve in the period that dispatched it, instead
  of one period later, changed nothing: z min 0.549 against 0.550.
- *Solver options.* mjlab matches the XML on iterations, tolerance, solver and
  cone; only `jacobian` differs, dense against sparse.
- *Object state feedback.* Both routes carry the pose;
  docs/coupling.md#object-state-feedback.

**The NaN lead is resolved (2026-09-29).** The 251 `FootManager_*`,
`CentroidalManager_*`, `ManipManager_*` and `*Task_*` channels previously
reported as NaN from sample 0 had finite values in the raw binary payload.
`finish_reset` calls the controller's reset after global initialization has
already done so; both calls register `perf_UpdateContacts`. mc_rtc's logger
overwrites the callback without emitting a removal event, leaving its reader
with one extra key and shifting subsequent values. This also happens on startup,
so skipping episode resets could not eliminate it. `mc_log_ui` uses the same
converter, and its agreement was not independent evidence of live NaNs.

The bridge now removes that entry explicitly before the second reset. A native
regression test fails before the fix and passes after it. Repairing the duplicate
event in a copy of the archived 25 s log restores **12,498 / 12,500** finite
`CentroidalManager_ZMP_measured_x` and `FootManager_refZmp_x` samples. A fresh
15 s JVRC1 demo gives **7,498 / 7,500** finite samples; the two missing samples
precede manager registration. See [controller logs](coupling.md#controller-logs)
for the cause and recovery details. No protected-member accessor is needed to
read those channels. Existing uncorrected exports and their `zmp_err_mean_m`
remain untrustworthy.

**The centroidal configuration gap is also eliminated.** JVRC1 inherits the
top-level `CentroidalManager` block; its per-robot section need not repeat it.
The recovered mc_mjlab log and the archived mc_mujoco comparison both report
`method: PreviewControlExtZmp`, `refComZ: 0.825`, `dcmGainP: 2`,
`comZGainP: 2000`, `comZGainD: 500` and `zmpVelGain: 0.02`, with ZMP and CoM-Z
feedback enabled and both `useActualStateForMpc` and
`useActualComForWrenchDist` false.

**What the recovered logs add.** One archived trace per simulator, measured over
Hold +4 to +14 s (5,000 mc_mjlab samples at 2 ms; 2,000 mc_mujoco samples at
5 ms). The mc_mjlab source is
`mjlab25-LocomanipController-2026-09-29-14-11-57.bin`; the reference is the saved
`mcm.flat` used in the earlier force comparison. These are a diagnostic pair,
not a new payload sweep:

| quantity | mc_mjlab | mc_mujoco |
| --- | --- | --- |
| measured minus control ZMP, x RMS | 0.04854 m | 0.01435 m |
| measured minus control ZMP, y RMS | 0.04391 m | 0.02043 m |
| actual CoM z, mean / std | 0.81700 / 0.00433 m | 0.82424 / 0.00127 m |
| external-ZMP x offset, mean / std | +0.00414 / 0.01738 m | -0.02977 / 0.00733 m |

**Resolved by the physical intervention below:** the earlier exclusions did not
test actuator torque saturation. Matching PD gains does not match actuation when
one simulator clips their output. The logger fix made this visible in `tauIn`;
it changed no targets or feedback itself.

**Re-measure if:** the coupling's control period changes, the blend's
implementation changes again, or the actuator effort bounds change.

## JVRC1 knee effort limits

**Current:** `get_robot_cfg` passes `get_effort_limits("JVRC1")` to each ideal
PD actuator. mjlab clamps the computed PD torque before writing MuJoCo's `ctrl`.
`R_KNEE` and `L_KNEE` each have a 100 Nm magnitude limit. mc_mujoco's motors have
neither control nor force limits, and its PD loop applies the full torque. The
`EFFORT_LIMIT = inf` helper default is only a fallback; the robot-specific config
supplies finite limits. Nominal position/velocity target clipping was inactive
in every diagnostic below, which did not make the torque clamp inactive.

One fresh 10 kg zero-action demo per intervention, seed 42, simulation step
1 ms, controller period/frameskip/decimation 2 ms/2/2, recovered blend unchanged.
No checkpoint or encoder randomization. The archived mc_mujoco nominal reference
uses 5 ms. Statistics use Hold +4 to +14 s: **5,000 samples per mc_mjlab run** and
**2,000 mc_mujoco samples**. Knee torque is the measured actuator torque `tauIn`,
not the kinematic controller's zero `tauOut`.

| quantity | module limits | unclamped PD | mc_mujoco reference |
| --- | --- | --- | --- |
| left hand fy mean / std | +2.25 / 17.27 N | -15.00 / 5.64 N | -13.97 / 5.68 N |
| x ZMP error RMS | 0.04796 m | 0.01401 m | 0.01435 m |
| y ZMP error RMS | 0.04472 m | 0.02024 m | 0.02043 m |
| true minus estimated base x, mean | 0.17408 m | 0.02046 m | 0.01616 m |
| base x error, Hold +9 to +10 s mean | 0.19120 m | 0.02277 m | 0.01825 m |
| right / left knee at 100 Nm | 42.46% / 40.72% | 0.02% / 0% | 0% / 0% |
| right / left knee above 100 Nm | 0% / 0% | 21.86% / 21.70% | 22.75% / 21.75% |
| both knees absolute torque p90 / p99 / max | 100 / 100 / 100 Nm | 107.96 / 114.67 / 116.11 Nm | 109.22 / 115.75 / 117.45 Nm |

![Matched hand force, base estimation and knee torque traces](figures/jvrc1-effort-limits.svg)

The traces separate when the gait starts, about three seconds into Hold. The
bounded run spends much more of each stance at its knee limit, the base estimate
falls behind simulation truth, and the hand/ZMP loop amplifies the gait. Removing
the caps brings all three signals close to the mc_mujoco reference. The
intervention establishes torque saturation as the cause of the nominal
instability; the base-estimation drift is part of that disturbed loop, rather
than independent evidence of a broken observer.

| fresh diagnostic | duration | minimum root z | robot forward | final cart position error | physical fall |
| --- | --- | --- | --- | --- | --- |
| 10 kg, module limits | 31.232 s | 0.54906 m | 1.45828 m | 0.46531 m | yes |
| 10 kg, only knees unclamped | 40.000 s | 0.72187 m | 1.17205 m | 0.12060 m | no |
| 10 kg, all PD unclamped | 43.396 s | 0.72298 m | 1.18399 m | 0.12169 m | no |
| 100 kg, module limits | 12.162 s | 0.54898 m | -0.33515 m | 0.37558 m | yes |
| 100 kg, all PD unclamped | 40.000 s | 0.72129 m | 1.09085 m | 0.03268 m | no |
| 464.2 kg, all PD unclamped | 40.000 s | 0.72153 m | 1.07020 m | 0.00384 m | no |

Loaded diagnostics change only cart mass, retaining the asset's diagonal inertia
`[1, 1, 1]`, as the mc_mujoco sweep does. The task's payload randomization also
scales inertia, so these are not replicas of its earlier sweep. The 40 s runs
cover the push and release onset, not the complete acceptance gate. The 43.396 s
nominal run reaches the configured completion state, then truncates on a worker
timeout during Exit; it is evidence of physical stability, not a healthy final
controller gate. The 0.12 m nominal endpoint error also remains larger than the
historical mc_mujoco result.

**Controls that did not recover the bounded gait:** one run each. CPU MuJoCo
3.11.0 on the same scene stays upright to 43.314 s but misses the cart target by
0.92875 m. The installed mc_mujoco engine, MuJoCo 3.3.6, substituted into the
same mjlab pipeline reaches 40 s with 0.92838 m error. Restoring mc_mujoco's
second coplanar cart floor falls at 18.194 s; at 5 ms it falls at 29.870 s.
Removing the extra policy-boundary `forward` call falls at 33.344 s. The compiled
robot masses, inertias, damping, armature and mesh vertices agree. These controls
show sensitivity in late collapse timing, while the earlier poor tracking
persists with the torque caps.

**Meaning for the baseline:** the recovered blend expects knee torque authority
above the current module limits even at 10 kg. Reproducing mc_mujoco requires
explicitly matching that authority; evaluating a bounded controller requires
keeping the limits and changing the controller's gait or load allocation. A gain
reduction that merely stops the fall does not establish a feasible loaded gait.
The diagnostic changes no shipped actuator or controller configuration.

**Re-measure if:** knee effort bounds, PD gains, footstep timing, cart contact
dynamics or the blend change. Compare saturation during the cart gait itself;
an unrelated settled zero-residual baseline cannot certify these caps as inert.

**History:** 2026-09-29 -- followed the recovered logger channels to stance-time
knee saturation, then independently removed all PD caps and only the knee caps.
The plot retains the archived mc_mujoco reference and the two new nominal traces.
Local diagnostic code, state snapshots and reports are saved under
`logs/locomanip/instability-2026-09-29/`.

## zmp_tracking

**Current:** `zmp_error` remains a diagnostic metric. The retired reward compared the planned ZMP
against the centre of pressure of **every** contact -- both feet and both hands
(`CONTACT_ZMP_PARAMS`) -- while the `min_normal_force` gate still reads the feet
alone (`ZMP_PARAMS`). Two `ZmpSensors` instances, resolved once by
`mdp.sensors.contact_zmp_sensors`; the caches are keyed by sensor set, so the
foot-only one is still shared with `gait_phase`, `foot_load_share` and
`zmp_grounded`.

**Why:** `planned_zmp` (docs/coupling.md#planned_zmp) inverts the centroidal
equation `m(c̈ - g) = ΣF` with `dL/dt = 0`, so it is the ground-plane
intersection of the resultant of *all* external contact forces. Aggregating only
the feet compares that against a different wrench the moment the hands carry
load, and the two disagree by `f·(p_hand_z - h) / F_z`. For a 50 N push with the
hands 0.2 m above the CoM and `F_z ≈ 520 N` that is **≈0.096 m** -- about twice
the former `ZMP_TRACKING_STD`, which drove the kernel to ~0.03 for a robot tracking its
plan perfectly. With `CART_MASS_RANGE_KG = (1.0, 1000.0)` the task guarantees
such pushes, so the term was paying the policy least exactly where the
manipulation was going best.

Aggregating the hands is the correction rather than building a second reference:
the ground-plane ZMP of the total contact wrench *is* the centroidal ZMP, so the
two sides of the comparison become the same quantity again, with no new
datastore callback and no assumption about what the controller planned for the
hands.

**Residual error:** the identity assumes `dL/dt = 0`, which `planned_zmp` also
assumes, and the MuJoCo site wrenches carry the distal links' own inertia -- the
same approximation the foot sensors already made, now applied at the wrists too.

**Re-measure if:** the hands grasp somewhere other than the cart handle, the
robot gains a third contact set, or `planned_zmp` starts reporting a
hand-corrected plan of its own.

**History:** 2026-09-28 -- found by review; before it, the measured side was
`GROUND_CONTACT_SENSORS` only. No locomanip training curve taken before this date
is comparable with one taken after, since the reward's value changed wherever the
hands were loaded.

## Retired ZMP tracking width

**Retired:** `ZMP_TRACKING_STD` was `0.05` m, the Gaussian kernel width of the `zmp_tracking` reward,
which scores the distance between the measured centre of pressure -- over every
contact, see `zmp_tracking` above -- and the ZMP the controller planned, and
scores **zero** on any step whose feet carry less than
`min_normal_force` (20 N) -- an unloaded foot has no centre of pressure, and
without the mask a robot in flight would collect the maximum.

Read the companion metrics as a pair, `zmp_error / zmp_grounded`: `zmp_error`
alone averages over every step, so it *falls* when the robot spends more time off
the ground.

**Not measured for this task**, but it does discriminate: on one zero-residual
HRP5P episode with a 112 kg cart, the kernel read 0.9995-0.9998 through the
approach and dropped to 0.745 once the push loaded the feet. That reading was
taken with the foot-only measured side, so the 0.745 confounds tracking error
with the hand-force bias `zmp_tracking` above removes; it has not been retaken. The width still has
to be placed against the error distribution *during the push*, over episodes and
masses, which has not been taken.

**Re-measure if:** the robot, the cart or the foot force sensors change.

**History:** chosen as a round number, half of `residual_balance`'s `DCM_STD`.

## FORCE_HISTORY

**Current:** `20` frames, 0.4 s at the 50 Hz policy rate, on all four force
sensors. The manipulation phase and completion flag stay single-frame. Balance
signals use `SAFETY_HISTORY = 5` frames.

**Why any history at all:** the task randomizes the cart's mass and never tells
the controller or the policy what it is, and one frame of force and velocity
cannot separate a heavy cart from a stuck one. 0.4 s matches
`residual_balance`'s `CONTROLLER_HISTORY`, and covers both the hand-wrench
filter's own 0.1 s constant and a footstep.

**Re-measure if:** the policy rate changes, or an ablation shows a shorter window
identifies the payload as well.

**History:** added with payload randomization, while the actor still saw object
state: that policy grew from 175 to 536 actor dimensions and 657 critic
dimensions with this history. Those counts do not describe the safety policy;
the force history stays for detecting changes in hand and foot loading.

## task_complete

**Current:** a termination, carrying `time_out=True`, so the episode ends the
step the controller raises `Locomanip::complete` and the value bootstraps rather
than paying the fall penalty. Nothing after the FSM's own verdict is the task.

**Why it is a termination and not just a metric.** Without it the FSM runs on
into the state after the cycle for the rest of the 70 s episode, and on JVRC1
that state wedges the worker. Seen twice: the 2026-09-29 nominal diagnostic
truncated on a worker timeout during Exit, and the first JVRC1 feedback training
run (2026-09-29, unclamped knees) stalled permanently the step after
`task_complete` first reached 1.0 -- iterations 0-30 at 0.0000, 31 at 0.7500, 32
at 1.0000, then 35 minutes of 60 s `controller_timeout_ms` expiries and worker
respawns without completing a single further iteration. HRP5P completes the same
cycle without wedging, which is why its 2000-iteration run never saw this.

**What it changes in the objective.** Episodes now end at completion rather than
at the cap (then 70 s), so a completed episode is shorter and its reward sum smaller.
Read `Episode_Reward/*` against episode length accordingly, and note that the
2026-09-24 HRP5P feedback run trained *with* the post-completion tail: its
episode sums are not comparable term by term with any run after this.

**Re-measure if:** the controller's post-cycle state changes, or the episode cap
moves.

**History:** 2026-09-29 -- added after the wedge above. Terminating on the
controller's verdict was already the intent of `EPISODE_LENGTH_S`, which only
ever bounded a cycle that would not end.

## task_success

**Current:** a cart-placement diagnostic retained in the comparison. The
comparison's safety verdict is now `safe_completion`. `task_success` is the FSM's
`Locomanip::complete` **and** the cart within `SUCCESS_POSITION_TOLERANCE_M`
(0.15 m) of its commanded position, **and** within `SUCCESS_YAW_TOLERANCE_RAD`
(10 degrees) of its commanded yaw, **and** no controller or worker failure, and
no fall termination on the step. Those tolerances are
`verify_locomanip.py`'s own acceptance gates, so the training verdict and the
acceptance run agree on what "done" means.

**Why `task_complete` alone is not success.** `Locomanip::complete` reports that
the executor reached `LMC::Pause_`, and `ConfigManipState` leaves the push
phase when its waypoint queue empties -- on a schedule, not on arrival. A cart
that never moved completes exactly like one that arrived.

Measured 2026-09-16, two zero-residual environments at the nominal 10 kg cart:
both reported `complete = 1` at t = 47.6 s with the cart 0.154 m and 0.144 m from
its commanded position, landing either side of the tolerance, so
`task_success` read 0 and 1. Against a 400 kg cart the robot fell at t = 18.5 s
with the cart 0.29-0.33 m out, where completion would never have fired at all.

`task_complete` stays in the metrics beside it: the gap between the two is how
often the schedule outran the cart.

**The 0.15 m tolerance is inside the run-to-run noise on HRP5P.** Three identical
`--task demo` cycles on 2026-09-28 gave 0.14917, 0.15088 and 0.15695 m, so the
same configuration passes or fails the position check by which run you look at.
Read a single cycle's `cart_position` as a coin toss at this tolerance; only a
distance well clear of 0.15 m says anything.

**Re-measure if:** the FSM's waypoint durations change, or the acceptance
script's tolerances move.

**History:** added when a review pointed out that the evaluation spec was scoring
the controller's own schedule rather than where the cart ended up.

## Retired object pose noise

**Retired:** `OBJECT_POSE_NOISE_M` was 0.02 m and `OBJECT_YAW_NOISE_RAD` 2 degrees,
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

**Retired:** `object_tracking` 1.0, `zmp_tracking` **0.5**, `termination_penalty`
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

## POSITION_REQUEST_WEIGHT

**Current:** the registered position-residual task charges `residual_magnitude`
and `residual_rate` at **-0.05** each. `REQUEST_WEIGHT = -0.3` stays the
builder default, so the hand-force feedback task (and the safety task built on
it) and the motion task, whose rewards were retuned against -0.3
(locomanip-motion.md#DCM_REWARD_WEIGHT), are unchanged. -0.05 is a hypothesis,
not a measurement; it is the only weight changed, so the next run's delta is
attributable to it.

**Why it came down: at -0.3 the penalty priced the policy's own exploration
noise, not its residual.** The request is the sampled tanh-Gaussian output over
35 residual joints. With a near-zero mean, each joint's expected square is about
`std^2`, and independent noise doubles that for the rate term. At the initial
std 0.10 that is `0.3 x 35 x 0.01 x (1 + 2) = 0.315` per second in
`Episode_Reward` units (episode sum / `max_episode_length_s`), against
`contact_dcm_stability` 0.36-0.38: noise alone cost about 85% of the objective.
The logged terms match that prediction, leaving almost nothing for the mean:

| iteration | std | `residual_magnitude` (noise-only prediction) | `residual_rate` (noise-only prediction) |
| --- | --- | --- | --- |
| 50 | 0.09 | -0.0945 (-0.085) | -0.1824 (-0.170) |
| 249 | 0.06 | -0.0501 (-0.038) | -0.0738 (-0.076) |

Run `2026-10-16_14-55-11` (HRP5P position residual, 256 envs, seed 42,
`REQUEST_WEIGHT = -0.3`) raised `Mean reward` from 3.10 to 10.0 between those
iterations with `Mean episode length` flat (2017.63 -> 2033.15). The two request
terms cover that rise: they fell by 0.153 per second, 7.9 over the 51.66 s
episode, while `termination_penalty` went from -0.0221 to -0.0465.
The training curves did not move. Means over 30-iteration windows, each
iteration closing only about 5-20 episodes:

| iterations | reward | `task_success` | `safe_completion` | `collapsed` | `controller_failed` | `contact_dcm_stability` metric |
| --- | --- | --- | --- | --- | --- | --- |
| 38-67 | 3.73 | 0.53 | 0.62 | 0.21 | 0.17 | 0.461 |
| 98-127 | 6.98 | 0.53 | 0.61 | 0.26 | 0.13 | 0.472 |
| 158-187 | 8.56 | 0.47 | 0.60 | 0.23 | 0.15 | 0.475 |
| 218-247 | 10.22 | 0.51 | 0.63 | 0.25 | 0.12 | 0.480 |

So the cheapest return was annealing std toward its 0.05 floor, and most of the
run's early gradient went there.

**The deterministic policy had beaten mc_rtc anyway.** `compare_to_baseline.py
--task ... --num-envs 16 --num-workers 6 --minutes 20` on `model_300.pt` of that
run, beside the training job, scored the first 9 episodes of each env, 72 per
arm:

| | baseline | policy | delta |
| --- | --- | --- | --- |
| `safe_completion` | 54.2% [42.7, 65.2] | 79.2% [68.4, 86.9] | +25.0%, p = 1.46e-03 |
| `collapsed` | 36.1% | 8.3% | -27.8% |
| `controller_failed` | 9.7% | 11.1% | +1.4% |
| `contact_dcm_stability` metric | 0.5170 | 0.4940 | -0.0230, p = 1.27e-04 |
| `zmp_error` | 0.1807 | 0.3113 | +0.1306, p = 0.150 |

Every cart drew above 300 kg (mean 474.8 vs 466.7), so this is one stratum. The
p-values treat episodes as independent, but each arm holds only 8 encoder-bias
draws. Per env, the policy completed 7/10 to 8/9 everywhere; the baseline ran
2/10 to 8/9, six envs at 6/10 or below. A Mann-Whitney test on those 8 vs 8
rates sits near p = 0.05.

**Replicated with `--nominal`, the effect is real but a third the size.**
Same checkpoint and command with `--nominal --minutes 75`, which drops
`encoder_bias` (this task's only startup draw), so episodes within an env are
independent. First 36 episodes per env, 288 per arm, cart mass balanced
(mean 468.4 vs 471.1 kg, KS p = 0.93):

| | baseline | policy | delta [95% Newcombe] | p |
| --- | --- | --- | --- | --- |
| `safe_completion` | 186/288 = 64.6% | 213/288 = 74.0% | +9.4% [+1.8, +16.8] | Fisher 0.019; per-env Welch 0.050; mass-stratified CMH 0.007 |
| `collapsed` | 62/288 = 21.5% | 34/288 = 11.8% | -9.7% [-15.8, -3.6] | Fisher 0.0024; CMH 0.0016 |
| `controller_failed` | 32/288 = 11.1% | 35/288 = 12.2% | +1.0% | 0.80 |

By cart mass, `safe_completion` reads 98/98 vs 98/98 at 300-400 kg, 68.8% vs
73.2% at 400-525 kg, and 24.7% vs 47.3% at 525-700 kg: the policy's whole
gain is in the heaviest band. It pays for it in tracking: the
`contact_dcm_stability` metric 0.5067 -> 0.4937 (p = 8.4e-06), `zmp_error`
0.2339 -> 0.3497 m (p = 0.0023), `object_position_error` 0.0694 -> 0.0985 m
(p = 0.0075), `task_success` unchanged (0.5556 vs 0.5764, p = 0.62). The 20-min
run's +25% came from 8 unmatched encoder-bias draws per arm, as
evaluation.md#episodes-are-not-independent-samples warns.

**`model_1040` gained nothing over `model_300`.** Same nominal command, 288 per
arm, mass balanced (473.0 vs 473.6 kg, KS p = 0.96): `safe_completion` 183/288 =
63.5% -> 207/288 = 71.9%, +8.3% [+0.7, +15.8] (Fisher 0.040, per-env Welch
0.022, CMH 0.011); `collapsed` 20.8% -> 11.8% (Fisher 0.0046);
`controller_failed` 15.3% vs 14.9%. At 525-700 kg 26.0% -> 41.4%, at 400-525 kg
66.7% -> 76.0%. Tracking costs match: `contact_dcm_stability` 0.5050 -> 0.4915
(p = 2.3e-06), `zmp_error` 0.2316 -> 0.3430 m (p = 0.022). Policy against policy,
74.0% at 300 vs 71.9% at 1040 is indistinguishable, so the run's real gain was
in place by iteration 300.

Collapse in the stochastic training rollouts (0.21-0.26) exceeded the
deterministic policy's 11.8%, so exploration noise may itself cost
training-time falls. -0.05 is tested against a run that already beats the
baseline, not one that failed to.

**Why -0.05.** It cuts the noise-only cost at std 0.10 to `0.05 x 1.05 = 0.053`
per second, about 14% of the objective. It still prices a real residual: a
0.3 normalized request on all 35 joints costs 0.16 per second, 43% of the
objective, and the clamp keeps the term bounded at any weight
(reward-shaping.md#requested_action_l2). It also raises the net living reward
from about 0.06 to 0.32 per second, so a fall forfeits more without touching
`termination_penalty`.

**Left alone:** `termination_penalty` (-2000 failed in reward-shaping.md#termination_penalty),
`contact_dcm_stability` 1.0 (the only positive term), and the guard terms
`upright`, `collapse_risk`, `rapid_tilt` and `foot_slip`. Those four together
read under 0.01 per second in that run, so they did not compete with the
objective.

**Re-measure if:** the residual joint set, `std_range` or `init_std`, the
request definition, or any other reward weight changes. Judge the next run by
the windowed task metrics above, not by `Mean reward`.

**History:** 2026-10-16 -- lowered from the shared -0.3 for the position task
only, from run `2026-10-16_14-55-11` at iteration 250, without stopping it.

## locomanip safety objective

**Current:** the trainable position and hand-feedback tasks reward
`contact_dcm_stability` at 1.0, retain `termination_penalty` -200 and `upright`
-2.0, and charge `collapse_risk` -2.0, `rapid_tilt` -0.2, `foot_slip` -1.0,
and the existing residual magnitude/rate costs -0.3 each (-0.05 on the
position task, see [POSITION_REQUEST_WEIGHT](#position_request_weight)). `zmp_tracking` is
retired as a reward but its `zmp_error` and `zmp_grounded` diagnostics remain.
These weights are initial hypotheses, not a measured optimum.

`contact_dcm_stability` uses the commanded CoM velocity and the equivalent
ground-plane ZMP of the *foot and hand* wrench together. Its Gaussian width,
`CONTACT_DCM_STD = 0.10` m, is an initial choice; it shares only the value with
`residual_balance`'s `DCM_STD`. The feet alone gate the reward at 20 N: unloaded
feet cannot earn balance credit from hand contact. This is an instantaneous
dynamic measure under the linear inverted-pendulum approximation; the moving
cart is not a fixed support, so the value is not a proof of future capturability.
The old foot-only DCM formula would omit the hand wrench during Hold, exactly
when the task is most loaded.

`collapse_risk` starts `COLLAPSE_WARNING_MARGIN = 0.08` m above each robot's
existing 70%-of-nominal height termination, and doubles its bounded penalty when
downward speed reaches `DOWNWARD_SPEED = 0.30` m/s. `rapid_tilt` charges only
roll/pitch angular speed above `FREE_TILT_RATE = 0.50` rad/s.
The loaded-foot `foot_slip` term uses the sole velocimeters already installed on
the robot assets. These guards leave normal swing-foot motion uncharged.

Actor observations now add root height and vertical speed, foot normal loads
and load share, sole velocities, controller joint-position error, and planned
CoM velocity. Base angular velocity, delivered action and the new balance
signals carry five frames of history. The critic also sees true CoM state and
the measured foot ZMP offset. Cart mass and true hand-cart force remain
critic-only because payload affects fall risk, even though cart pose and
trajectory tracking no longer enter the actor or reward.

Both trainable tasks and the play-only demo classify a low tilted robot as
`fell_over`; `collapsed` requires low height *and* orientation within the tilt
limit. The two labels cannot coincide.

`safe_completion` is the comparison's primary success metric: the controller
must complete its cycle without a fall or controller/worker fault on that step.
`robot_displacement` is reported beside it so a policy that stays upright by
halting forward motion can be rejected. `task_success` still reports the old
cart-placement gate as a separate diagnostic, and cart mass still stratifies
the comparison.

**Re-measure if:** any reward weight or threshold is tuned, the hand contacts or
controller references change, or training starts on another robot. Judge a
candidate by survival to controller completion and fall hazard by phase and
payload mass, with robot displacement to catch a policy that suppresses walking.

**Runtime check, 2026-10-09:** one step completed with finite rewards for the
position task and both feedback robot profiles. A zero-action HRP5P feedback
rollout at fixed 10 kg, seed 42, encoder bias disabled, ended at step 56
(1.12 s) as `fell_over`, before Hold. A matching control with the new safety
observations and rewards removed ended at the same step with the same label
(one run per condition, seed unspecified for that pair). This checks that the
new terms did not cause the early fall in that setup; it does not validate the
contact DCM reward under a loaded hand contact. Recheck the installed
controller and workspace before treating a longer rollout as a reward test.

**Runtime check, 2026-10-09, later:** `scripts/verify_locomanip.py --cycles 1`
in that setup (zero action, fixed 10 kg, seed 42, encoder bias removed) did
*not* reproduce the fall. The feedback task completed at 47.66 s with cart error
0.144 m and 0.25 degrees, 1.166 m forward and minimum root height 0.7227 m; the
position task at 47.62 s with 0.137 m, 1.52 degrees, 1.171 m and 0.7235 m.
Neither reported a controller failure. One run each, and what differed in the
1.12 s rollout was not recorded. Both fail only `held_two_seconds` (0.0 s),
because `task_complete` now truncates the episode on the completion step.

## EPISODE_LENGTH_S

**Current:** `51.66` s (2583 policy steps): the slowest completion of a
successful push measured on 2026-10-13 plus a 2 s margin. One-cycle
`scripts/verify_locomanip.py --isolation` runs (72 s cap), HRP5P feedback task,
zero action, seed 42, encoder bias removed, one run each, on LocomanipController
`7713446` (`reachDuration` 2.5 s): `Locomanip::complete` rose at 49.66 s at
10 kg and 49.26 s at 500 kg. Under the previous 49.66 s cap the 10 kg completion
landed on the cap's last step. Since `task_complete`
truncates every completed episode, the cap only ever ends an episode that has
not completed, which `safe_completion` then scores as a failure.

**The margin is a choice, not a measurement.** Completion time varies with
cart mass: over run [2026-09-24_17-38-11](#run-2026-09-24_17-38-11_hrp5p-feedback)
the healthy phase held a mean episode length of ~2750 of a 3000-step cap against
a then-52.2 s cycle, and `task_complete` sat at 0.85 with the missing share
indistinguishable from "ran out of time". Beyond 10 and 500 kg, the spread over
the mass range and the JVRC1 cycle, which shares this constant, have not been
measured against 51.66 s.

**It does not change the credit horizon.** `gamma = 0.997` is 6.7 s
(docs/ppo.md#gamma), far inside the cap, so the cap changes how much of a slow
cycle fits, not what the policy can see.

**Re-measure if:** the FSM's waypoint durations change, `task_complete` falls
while `safe_completion`'s misses end at the cap, or training starts on JVRC1 or
at cart masses far from 10 kg.

**History:** 2026-09-25 -- widened from `60.0` to `70.0` after the first full
feedback run could not distinguish a failed cycle from a slow one; the cycle then
took 52.2 s. 2026-10-09 -- cut to `49.66` once completion truncated the episode
and the installed cycle measured 47.66 s. 2026-10-13 -- raised to `51.66`
after LocomanipController `7713446` slowed HRP5P's reach and release from 1.0 s
to 2.5 s (`ManipManager.reachDuration`), which moved completion to 49.66 s.

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
`ConfigManipState`'s push phase. HRP5P ships the block below in
`etc/robots/hrp5_p.yaml` since 2026-10-12; before that it ran on the compiled
default (`alpha: 0.02`, `forceProjection: [1, 0, 1]`, `momentProjection:
[0, 1, 0]`), which is what every HRP5P mc_mjlab run and comparison up to and
including run 2026-10-09_15-16-21 trained and was scored against. Tuned on
HRP5P, in mc_mujoco:

```yaml
HandWrenchAdaptation:
  enable: true
  alpha: 0.06                 # tau = dt / alpha = 33 ms at HRP5P's 2 ms period
  forceProjection: [1, 1, 1]  # the whole measured force, couple dropped
  momentProjection: [0, 0, 0]
```

LocomanipController `7713446` (2026-10-12) retuned HRP5P for light carts to
`alpha: 0.03`, `forceProjection: [0, 0, 0.98]` -- local x dropped as noisy, local
z the push -- with `reachDuration: 2.5` and hand offsets at -0.33 m. In mc_mjlab,
feedback task, zero action, one `verify_locomanip.py --isolation` cycle each,
2026-10-13: 10 kg completes with cart error 0.0944 m (0.1588 m at alpha 0.06),
500 kg with 0.0448 m, z min 0.7252 / 0.7268 m. Push-only capped at 350 kg in the
mc_mujoco table below; that table predates this tune.

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

## controller_force_feedback

**Current:** the locomanip position-residual task, including its motion and
braking derivatives, sets `controller_force_feedback=False`. The bridge omits
all measured force-sensor routes, both force and moment triples, while retaining
every native sensor slot. Each input fill writes zero into these slots, including
initialization and episode resets, so mc_rtc cannot retain an earlier measurement.
IMU, encoders, root/object state and actuator effort still reach the controller.
Actuator effort is `qfrc_actuator`, not a contact force/torque sensor reading.

MuJoCo continues computing force measurements for privileged critic observations,
rewards and diagnostics. The actor has no direct wrench observations. Controller
outputs observed by the actor now derive from the masked controller inputs.
No measured force is replaced with nominal support forces or simulator contact
estimates. Any wrench-feedback action explicitly configured on top would be a
synthetic policy output, not a measured sensor baseline.

The generic action default stays `True`; the demo and the separate hand-force
feedback task retain measured feedback, as do the latter's safety derivatives.
The checkpoint contract includes this flag for both full resumes and actor-only
loads. A checkpoint without the field means `True`, even though generic added
fields are otherwise tolerated. Replaying an old experiment therefore requires
explicitly restoring measured feedback; training the no-sensor experiment starts
a new run. C4 governs checkpoint validity and C5 limits what training results
establish about deployment in [hard-constraints.md](hard-constraints.md).

**Re-measure if:** sensor routing, controller observers or force-dependent tasks
change. Masking the measurements establishes the information restriction; it
does not demonstrate that the existing controller can balance without them.

**History:** on 2026-10-20, the audit of
`2026-10-19_17-57-41_mass-range` found no direct actor force observations and
`HandWrenchAdaptation.enable: false` in the snapshotted HRP5P controller config.
Nevertheless, `SimControllerBridge` routed every named hand and foot sensor into
native `ControllerInstance::apply_input`, which called `setWrenches` each period.
The hand tasks were impedance tasks. Disabling hand-wrench adaptation had not
disabled their measurements or the controller's foot feedback. Results from that
run describe a sensor-enabled base controller, not a robot without force sensors.

Live HRP5P check on 2026-10-20: seed 42, one environment and one native worker,
10 kg cart, zero residual, encoder-bias event removed, automatic reset disabled.
Two explicit 12 s reset cycles per condition first verified input routing: the
sensor-disabled condition had exactly zero in all 24 native wrench columns on
every policy step while simulator forces reached 604.154 N. Both disabled cycles
survived those 12 s; minimum root heights were 0.768678844 and 0.768678904 m.

A separate full-cycle check used the same settings, n=1 episode per condition:

| Measured feedback | End time | Minimum root height | Planar displacement | Outcome |
| --- | --- | --- | --- | --- |
| Enabled | 49.64 s | 0.725322247 m | 1.210022569 m | Completion truncation |
| Disabled | 16.66 s | 0.543859005 m | 0.743203044 m | Physical termination |

Neither condition reported a controller or worker failure. The disabled run's
maximum absolute native wrench input was exactly 0 throughout; simulator force
components reached 1485.612 N. The no-sensor base controller fell during motion.
These are routing and baseline smoke checks, not policy evaluations or evidence
that residual training can recover a controller deprived of foot feedback.
