# Residual feedback learning

## SCOPE

**Current:** `Mc-Mjlab-Residual-Feedback-...-Joint-Torque` reproduces the
residual-feedback formulation of
[Ranjbar 2021](https://arxiv.org/abs/2106.04306) on the ResidualMPC task. The
policy adds a bounded offset to the joint positions mc_rtc reads, alongside the
existing torque residual, so the controller replans *with* the correction
instead of resisting it.

Everything except the action term is shared with `## residual-mpc.md#SCOPE`:
same rewards, observations, events, disturbance, command box and PPO settings.
The only difference is where the residual enters, which is the comparison the
paper makes.

**Re-measure if:** the ResidualMPC task's rewards or observations change — the
two tasks are only comparable while they share them.

**History:**
- 2026-09-01 — added after the ResidualMPC residual was measured spending more
  torque for no tracking gain, the signature Ranjbar describes.

## WHY_FEEDBACK

**Current:** a residual added to a controller's *output* fights that
controller's own feedback loop. Ranjbar states it directly: the residual "causes
a feedback distribution shift that the controller sees as external perturbation
which it tries to resist," so the two "compete".

**Our prior does exactly this.** mc_rtc receives live encoder, IMU and wrench
feedback, and the ISMPC stabilizer is an explicit force-feedback loop, so an
injected torque is observed and opposed. The measurements match: across every
ResidualMPC comparison the residual spent *more* torque
(`+5.2%`, p `0.010` on `ent001`) while gaining nothing on the objective, and
under domain randomization it was significantly worse than the bare prior
(`-4.2%`, p `0.009`). `docs/residual-mpc.md#POWERED_RESULT`

**The formulation.** Where residual *action* learning computes
`a = f(o) + a_rl`, residual *feedback* learning computes `a = f(o + o_rl)`, and
the hybrid — Ranjbar's best variant, and the default here — computes
`a = f(o + o_rl) + a_rl`. Their result is that feedback residuals handle
position uncertainty best, action residuals handle orientation uncertainty best,
and the hybrid handles both.

**Re-measure if:** the controller stops receiving live state, which is what makes
the competition possible in the first place.

**History:**
- 2026-09-01 — recorded with the ResidualMPC torque evidence that motivated it.

## RESULT

**Current:** residual feedback roughly halves survival on this prior. The
formulation transfers structurally but not behaviourally: what a compliant
impedance controller absorbs, a balance-critical QP stabilizer does not.

**Matched-iteration episode length**, 128 environments, identical env config
(measured: zero differences outside the action term):

| run | 50 | 100 | 150 | 200 | 245 |
| --- | ---: | ---: | ---: | ---: | ---: |
| feedback, hybrid | 754 | 945 | 1004 | 946 | 960 |
| feedback, `joint_position` only | 741 | 1032 | 991 | 984 | 909 |
| **ResidualMPC + curriculum** | 478 | 1224 | 2172 | 2296 | **2318** |
| ResidualMPC, no curriculum (`std015`) | 534 | 1318 | 2011 | 2207 | 2079 |

**The curriculum is not the cause.** It was added at the same time as the
feedback channel, so both were confounded until this control ran. ResidualMPC
with it tracks ResidualMPC without it, and `kick_scale` stayed at `1.0` because
survival never fell below the regress threshold — the term behaves as designed.

**Neither is the channel count or the root-pose modality.** Dropping the torque
residual and every modality but `joint_position` changed nothing: `909` against
`960` at iteration 245. The remaining cause is the joint-position feedback
residual itself.

**And it is worse than it looks.** Both feedback runs drove `kick_scale` to its
`0.4` floor within the first `~100` iterations, so across the window compared
above they survived half as long against a disturbance `2.5x` milder than the
`1.0` ResidualMPC was holding at iteration `245`.

ResidualMPC reaches the same floor eventually — `kick_scale` was `0.4` by
iteration `474` — so the difficulty gap is a property of the comparison window,
not of the whole run. It does not weaken the result: by then both arms sit at the
same easiest setting and ResidualMPC still survives about twice as long
(`1948` at `474`).

**The mechanism is the one `## feedback_scale` warns about.** A virtual joint
offset is a lie told to a stabilizer that closes a state and force feedback loop.
Ranjbar's prior is a Cartesian impedance controller on a 7-DoF arm, where a
false joint reading perturbs a spring-damper; here it perturbs what keeps the
robot upright. `## SCOPE` records that difference; this is it measured.

**What is not claimed.** No deterministic checkpoint comparison was run — these
are training curves, and this repository has been wrong reading those before
(`docs/residual-mpc.md#forward_speed`). The claim is about survival during
training under matched conditions, which is what the four runs share. A paired
`compare_to_baseline` at 64 environments would be needed to say anything about
tracking quality.

**Wrench feedback at a live scale survives best of anything tried.** Re-run at
`50 N` — the scale that actually reaches the stabilizer — against the earlier
`5 N` run that measured an inert channel:

| run at iteration ~249 | episode length | `kick_scale` |
| --- | ---: | ---: |
| `joint_position` feedback | ~950 | 0.4 |
| ResidualMPC | 2318 | 1.3 |
| wrench, `5 N` (inert) | 2340 | 1.3 |
| **wrench, `50 N` (live)** | **2419** | **2.0** |

The difficulty column carries as much weight as the length: the `50 N` run
survived *longer* while the curriculum held its kick at the `2.0` ceiling,
`1.5x` harder than what ResidualMPC was facing. This section previously
predicted wrench feedback would be worse than joint position; it is better than
either, and better than no feedback at all.

**So the harm is proprioceptive, not general.** Falsifying joint angles corrupts
the kinematic state every downstream computation is built on. Falsifying a foot
wrench perturbs a quantity the stabilizer already treats as noisy and contested,
and the policy can apparently use that channel productively.

**Not yet a claim about policy quality.** These are training curves under a
curriculum that moved differently for each arm, so they compare survival, not
tracking. A paired `compare_to_baseline` at 64 environments with `--nominal` is
what would settle whether the wrench policy beats its own prior.

**Superseded prediction:**
A 250-iteration run with `wrench` alone (`torque_channel=False`, no joint
channel) tracks ResidualMPC rather than the joint-position feedback runs:

| iteration | ~136 | 249 |
| --- | ---: | ---: |
| wrench only | 2059 | **2340** |
| ResidualMPC | 2172 (at 150) | 2318 (at 245) |
| `joint_position` feedback | 991 (at 150) | ~950 |

Its curriculum also raised `kick_scale` to `1.3-1.5`, where the joint-position
runs sat pinned at the `0.4` floor — the policy was surviving well enough to be
made harder. This section previously predicted the opposite.

If it holds, the harm is specific to **proprioceptive falsification**: a lie
about joint position corrupts the kinematic state everything downstream is built
on, while a force offset is a noisy measurement the stabilizer already expects to
react against.

**The competing explanation is that the channel is too weak to matter.** `5 N`
against a per-foot load near `260 N` is about `2%`, so "no harm" may mean "no
effect", in which case the run only shows that an inert channel is harmless. The
potency probe that would separate these is still outstanding — the first attempt
was invalid for the reason recorded under `## feedback_modalities`.

**`joint_velocity` remains untested behaviourally.**

**Re-measure if:** `feedback_scale` is swept — `0.02 rad` is twice the encoder
bias and remains unswept, so a far smaller offset might be tolerable even if this
one is not.

**History:**
- 2026-09-01 — three runs plus a control; the feedback channel halves survival
  and the curriculum is exonerated.

## WRENCH_RESULT

**THE RESULT BELOW DOES NOT REPLICATE.** Five seeds, each scored by the same
paired protocol on `model_999`, are all significantly *worse* than their own
prior:

| seed | prior | policy | delta | z |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.08053 | 0.07423 | `-7.83%` | `-10.25` |
| 2 | 0.08204 | 0.07495 | `-8.64%` | `-10.62` |
| 3 | 0.08436 | 0.08055 | `-4.52%` | `-4.99` |
| 4 | 0.08301 | 0.07696 | `-7.29%` | `-6.97` |
| 5 | 0.08515 | 0.06815 | `-19.97%` | `-18.99` |

Across seeds: **`-9.65%`, sd `5.97`, t `-3.61`** (df 4). None positive; the range
is `-4.5%` to `-20.0%`. The `+4.35%` recorded below came from a single seed and
falls outside that entire range.

**Scoring the near-peak checkpoint helps, and changes nothing.** `model_999` is
past every seed's peak (peaks fell at iterations `134-600`), so it penalises them
all. Iteration `150` has both the best mean and the best worst-seed across the
five, and was chosen on that basis rather than per seed, so there is no selection
freedom:

| seed | `model_999` | `model_150` |
| --- | ---: | ---: |
| 1 | `-7.83%` | `-1.82%` |
| 2 | `-8.64%` | `-11.77%` |
| 3 | `-4.52%` | `-8.38%` |
| 4 | `-7.29%` | `-4.27%` |
| 5 | `-19.97%` | **`-1.13%`** |
| **mean** | **`-9.65%`** (sd 5.97) | **`-5.48%`** (sd 4.52) |

The gain is `4` points and seed 5's outlier is mostly a late-training artifact
(`-19.97%` to `-1.13%`), but **all five remain negative** and seeds 2 and 3 stay
individually significant at `z < -9`. Wrench feedback does not beat the prior at
either checkpoint.

**Scoring these needs the training commit.** `be58edd` dropped HRP5P's finger
joints from the observation mid-session, taking it from `124` to `88` wide, which
retires every checkpoint trained before it — the actor's first layer and
`obs_normalizer` are both sized by the concatenated width. The `model_150` scores
above were produced by checking out `fd1973e`, the commit each run records in its
own `git/` directory.

**Wrench feedback does not beat the ISMPC.** Nor does any other residual
configuration tested here — torque residual (`-2.64%`,
`docs/residual-mpc.md#POWERED_RESULT`), joint-position feedback (halves
survival), or this. The seed-to-seed spread is `~6` percentage points, which is
larger than every effect any single run has claimed all session.

**The `+4.35%` came from the resume, not the architecture.** Seed 42 retrained
continuously to the same `1249` iterations, scored on the same `model_1100` with
the same `30`-minute protocol — the interruption is the only variable removed:

| run | prior | policy | delta | p |
| --- | ---: | ---: | ---: | ---: |
| seed 42, resumed | 0.08123 | 0.08476 | **`+4.35%`** | `<1e-5` |
| seed 42, **clean** | 0.08098 | 0.08127 | **`+0.37%`** | `0.66` |

Both arms' priors agree to `0.3%`, so this is a policy difference, not a baseline
artifact. The resumed run reset `kick_scale` to `1.0` and re-climbed to `2.0`,
and it kept improving to iteration `1129`; the continuous run peaked at `129` and
was `4.7%` lower at `1100`. An interrupted curriculum appears to train better
than a monotone one — a finding about `## survival_kick_curriculum`, not about
wrench feedback.

**And the resume explanation fails too.** Running the identical resume pipeline
on seed 1 — train `250`, resume to `1249`, score `model_1100`, same `30`-minute
protocol — gives `-4.66%` (p `<1e-5`), worse than that seed's own clean
`model_150`. Resuming helped seed 42 and hurt seed 1, so it is not a mechanism.

| run | delta | p |
| --- | ---: | ---: |
| seed 42 resumed | **`+4.35%`** | `<1e-5` |
| seed 42 clean | `+0.37%` | `0.66` |
| seed 1 resumed | **`-4.66%`** | `<1e-5` |
| seed 1 clean, `model_150` | `-1.82%` | `0.06` |
| seed 1 clean, `model_999` | `-7.83%` | `<1e-5` |

**One positive result in eight measurements** of this configuration, spanning
`-11.8%` to `+4.35%`. The `+4.35%` is best read as seed-42 noise that happened to
land on a resumed run, not as evidence about resuming or about wrench feedback.

**Five randomly drawn seeds settle it.** Seeds `51272, 66938, 18536, 25104, 9038`
(drawn with `secrets.randbelow`, recorded before use), each through the identical
resume pipeline:

| seed | prior | policy | delta | z |
| --- | ---: | ---: | ---: | ---: |
| 18536 | 0.08150 | 0.07150 | `-12.27%` | `-16.09` |
| 25104 | 0.08059 | 0.07771 | `-3.57%` | `-4.51` |
| 51272 | 0.08017 | 0.07164 | `-10.63%` | `-13.51` |
| 66938 | 0.08168 | 0.07883 | `-3.50%` | `-4.38` |
| 9038 | 0.08153 | 0.07383 | `-9.45%` | `-13.05` |

**Mean `-7.88%`, sd `4.10`, t `-4.30`, `0/5` positive.** Adding seeds 42 and 1:
seven resumed runs, mean `-5.68%`, range `-12.3%` to `+4.35%`, **`1/7` positive**.

Every prior arm scores `0.080-0.082` while the policies span `0.0715-0.0848`, so
the variance lives in the policy, not the measurement. Seed 42's `+4.35%` is the
tail of a distribution centred near `-6%`.

I twice explained that single number with a mechanism — first the architecture,
then the resume — and both died on the next measurement. The spread was simply
wider than the effect, which is what a null distribution measured up front would
have shown before either explanation was written down.

**The lesson is the protocol, not the architecture.** A single paired comparison
returns `p < 1e-5` on ~1,000 episodes and still says almost nothing, because the
variance that matters is between *seeds*, not between episodes within a seed.
Nothing in this repository should be called a win on one seed again.

**Superseded single-seed result:**

**Current:** wrench feedback is the first residual in this repository measured to
beat the controller it modifies. `rfl-wrench50-1k` `model_1100`, nominal model,
`5.5 s` skipped, 64 environments, clustered by environment:

| term | prior | policy | delta | p |
| --- | ---: | ---: | ---: | ---: |
| `linear_tracking` | 0.06049 | 0.06307 | `+4.3%` | `1.9e-05` |
| `torque_l2` | -0.04798 | -0.04713 | `+1.8%` (less torque) | `1.7e-09` |
| `termination` | -0.00056 | -0.00039 | `-29.7%` | `2.2e-03` |
| **TOTAL** per-episode | 0.08155 | 0.08520 | **`+4.5%`** | `7.0e-07` |
| **TOTAL clustered** | | | **`+4.35%`** | **`<1e-5`** |

It tracks better, spends *less* torque, and falls `30%` less often — the inverse
of every ResidualMPC result, where the residual bought nothing with more torque.

**Against the same protocol on the same day:**

| policy | clustered delta | p |
| --- | ---: | ---: |
| ResidualMPC residual | `-2.64%` | `0.003` |
| wrench feedback | **`+4.35%`** | **`<1e-5`** |

**Why this channel and not the other.** Falsifying joint angles corrupts the
kinematic state every downstream computation rests on, and halves survival
(`## RESULT`). Falsifying a foot wrench perturbs a quantity the stabilizer
already treats as contested and noisy, so the correction enters a loop built to
absorb disagreement. Ranjbar's formulation does transfer to a balance-critical
prior — through the right feedback space.

**It needed the right scale and the right columns.** At `5 N` the channel was
inert and its run was indistinguishable from ResidualMPC; `50 N` is where it
starts to move the solve (`## feedback_modalities`). The `250`-iteration run also
understated it: reward per step peaked at iteration `1129`, past where the
ResidualMPC control peaked (`864`).

**What would make this a claim.** One seed, one checkpoint. A second seed is the
obvious next step, and `feedback_scale` for the joint channel remains unswept, so
"proprioceptive feedback is harmful" is established only at `0.02 rad`.

**Re-measure if:** the wrench scale, the disturbance, or the prior changes.

**History:**
- 2026-09-02 — first measured win for a residual in this repository.

## feedback_modalities

**Current:** `("joint_position",)` by default. Each entry adds a block to the end
of the action vector, in the order listed, so the layout is
`[torque residual | walking reference | joint_position | root_pose]`.

| modality | width | offsets | scale |
| --- | ---: | --- | --- |
| `joint_position` | one per residual joint | encoder columns, `in_np[:, 0:T]` | `feedback_scale`, rad |
| `joint_velocity` | one per residual joint | velocity columns, `in_np[:, T:2T]` | `joint_velocity_scale`, rad/s |
| `root_pose` | 6 | root block, `ro:ro+3` and `ro+3:ro+7` | `root_translation_scale` m, `root_rotation_scale` rad |
| `wrench` | `6 x` force sensors | wrench block, `wrench_off` | `wrench_force_scale` N, `wrench_torque_scale` Nm |

`wrench` resolves its width from the live layout — on HRP5P that is `24`, from
`RightFootForceSensor`, `LeftFootForceSensor` and both hands — and refuses to
build on a model with no force sensors. It is the analogue of the paper's
end-effector wrench channel, and for a walking stabilizer it is the loop the
controller actually closes on. `joint_velocity` is the modality Ranjbar names as
optional beside joint position.

**Verified isolated.** Saturating each block in turn, every other modality stays
at exactly zero: `joint_position` `0.02000`, `joint_velocity` `0.05000`,
`root_pose` `0.00500`, `wrench` `5.00000`, no leakage in any direction, robot
still walking at `0.257`. `action_dim` is `66` with all four enabled.

**Why a second space.** Ranjbar's argument is that the residual should enter a
space *relevant to the task*, and the paper evaluates an end-effector-pose
variant beside the joint-position one. For a biped the analogue is the root pose:
the ISMPC stabilizer keys off base state, so a correction there has more leverage
than one distributed across leg encoders.

**Rotation is composed, not added.** The root block carries a `wxyz` quaternion;
adding to it yields something that is not a rotation.
`_compose_small_rotation` builds a unit quaternion from the residual's rotation
vector and right-multiplies it in the body frame.

**Applied after both branches.** `_fill_root_columns` writes the root block in
either the named-routing or the fallback path, and the IMU columns are
overwritten afterwards, so the offset is applied last rather than inside a branch.

**All three channels work once they reach the right columns.** `root_pose` was
offsetting the root block, which mc_rtc ignores for orientation; rotating the
**IMU** instead — measured gravity by `-delta`, so the observer infers `+delta` of
tilt — makes it live. `wrench` needed a scale above the stabilizer's own noise.
Fresh environment per condition, each against its own zero control:

| condition | `qp_objective` | `vx` |
| --- | ---: | ---: |
| control | -6,149 | +0.1058 |
| IMU pitch, `0.02 rad` | **-6,774** (`+10%`) | **+0.1245** (`+18%`) |
| IMU pitch, `0.10 rad` | **-5,550** (`-10%`) | **+0.0955** (`-10%`) |
| foot `Fz`, `25 N` | -6,148 | +0.1055 |
| foot `Fz`, **`50 N`** | **-6,663** (`+8%`) | **+0.1200** (`+13%`) |

`planned_step_dx` barely moves in any of these: a false tilt or foot load changes
how the stabilizer *executes* a step, not the footstep plan, which the velocity
command sets. Reading only `plan_dx` would have called these inert too.

**Defaults are now these measurements**, not guesses: `root_rotation_scale`
`0.02`, `wrench_force_scale` `50.0`, `wrench_torque_scale` `20.0`. The earlier
`0.005` and `5.0` were set from the confounded probe below and were both beneath
the threshold where anything happens.

**The effect is not monotone.** `0.02 rad` of false tilt speeds the robot up
`18%`, `0.10 rad` slows it `10%`. Do not extrapolate from one point.

**Superseded: only `joint_position` is a working channel.** Measured with a fresh
environment per condition and an explicit zero control, `root_pose` and `wrench`
move nothing:

| condition | `planned_step_dx` | `qp_objective` |
| --- | ---: | ---: |
| control | +0.1077 | -6,149 |
| root pitch, `0.005 rad` | +0.1070 | -6,109 |
| root pitch, `0.02 rad` | +0.1082 | -6,145 |
| root pitch, **`0.10 rad`** | **+0.1082** | -6,149 |
| foot `Fz`, `5 N` | +0.1075 | -6,159 |

`plan_dx` is identical to four decimals even at `0.10 rad` — `5.7` degrees of
false tilt. **mc_rtc does not take base orientation from the root block**: its
observer derives it from the IMU columns (`_gyro_adr` / `_accel_adr`), so
offsetting `ro+3:ro+7` writes something nothing reads. Fixing the modality means
offsetting the IMU instead; as written it is a knob that drives nothing, the
failure mode `docs/residual-mpc.md#mean_speed` records.

`wrench` at `5 N` against a per-foot load near `260 N` is likewise below the
noise the stabilizer already tolerates.

**This explains two results that otherwise look meaningful.** The hybrid run
matched `joint_position`-only (`1004` against `991` at iteration 150) because
`root_pose` added nothing, and the wrench-only run matched ResidualMPC (`2340`
against `2318`) because that channel added nothing either. Neither is evidence
about feedback residuals; both are evidence that an inert channel is harmless.

**THE PROBE BELOW IS INVALID — it has no time control.** Its four conditions ran
sequentially on one environment without a reset, so each row is a *later* window
of the same continuous episode. Running the identical probe against the unrelated
`wrench` channel reproduced the same numbers to four significant figures
(`+0.2438`, `+0.2152`, `-29,215` against `-29,316`), which is only possible if
the perturbation is not what moves them. What the table shows is the ISMPC
ramping up over its first minute.

The `root_rotation_scale` and `root_translation_scale` values chosen from it
(`0.005 rad`, `0.0025 m`) are therefore **unjustified** rather than
wrong — they may still be sensible, but nothing here establishes them. A sound
probe builds a fresh environment per condition, as `scripts`-style sweeps do
elsewhere; the numbers are kept below only to record what was mismeasured.

**Superseded probe:**

| condition | `planned_step_dx` | `qp_objective` | `vx` |
| --- | ---: | ---: | ---: |
| zero | +0.1435 | -5,407 | +0.1290 |
| root pitch `+0.02 rad` | **+0.2438** | -29,215 | +0.2283 |
| root pitch `-0.02 rad` | +0.2152 | -91,875 | +0.1969 |
| root x `+0.01 m` | +0.1888 | **-171,929** | +0.1686 |

These differences are the trajectory's own evolution across four consecutive
windows, not the channel's authority. The probe was written to catch a knob that
drives nothing (`docs/residual-mpc.md#mean_speed`) and instead demonstrated a
subtler failure: a measurement that moves for a reason other than the one under
test.

**Re-measure if:** a modality is added — the widths feed `action_dim`, and every
recorded checkpoint is tied to that width.

**History:**
- 2026-09-01 — `root_pose` added beside `joint_position` and both enabled by
  default; scales set from the probe above.

## survival_kick_curriculum

**Current:** `initial_velocity_kick.scale` moves on the smoothed fraction of
episodes that end in `time_out`. Above `0.7` survival the kick grows by `0.05`,
below `0.6` it shrinks, clamped to `[0.4, 2.0]`. Wired into **both**
`residual_mpc_env_cfg` and `residual_feedback_env_cfg`, and omitted when
`pushes=False`, since there is no kick to scale.

**It mirrors the paper's table.** Ranjbar raises task uncertainty when the
success rate exceeds `0.7` and lowers it below `0.6`, by a fixed increment. Our
uncertainty is the reset kick rather than hole pose, and survival stands in for
insertion success.

**Structure borrowed, not reinvented.** The smoothing and the advance/regress
deadband follow `episode_length_impulse_curriculum` (`mdp/curricula.py`), which could
not be reused directly: it requires a `stratified_finite_impulse_curriculum` push
term, and both tasks now use `initial_velocity_kick`. Like that term, it reads
the termination buffers during `curriculum_manager.compute`, which runs before
`_reset_idx` clears them.

**It invalidates the recorded ResidualMPC comparisons.** Every figure under
`docs/residual-mpc.md#POWERED_RESULT` and `#tuning_plateau` was trained at fixed
difficulty. Re-baseline before comparing anything against them, at `--num-envs 64`
with the clustered test — never at 16
(`docs/evaluation.md#episodes-are-not-independent-samples`).

**It regulates to its setpoint.** Over the full `mpc-curriculum-control` run:

| quantity | min | max | mean | final |
| --- | ---: | ---: | ---: | ---: |
| `kick_scale` | 0.400 | 2.000 | 1.093 | 1.875 |
| smoothed survival | 0.020 | 0.868 | **0.633** | 0.799 |

Mean survival lands inside the `[0.6, 0.7]` deadband, which is what the term is
for. `kick_scale` exercised both clamps and reversed at each — `1.0` to the `0.4`
floor while the policy struggled, up to the `2.0` ceiling once it did not, then
back. Wide swings are the adjuster hunting around its setpoint with a fixed
increment, not instability.

**It costs nothing on ResidualMPC.** Smoothed reward per step peaks at `0.07431`
against `std015`'s `0.07475` without a curriculum, and ends higher (`0.07034`
against `0.06957`). Episode length tracks the no-curriculum run throughout
(`## RESULT`). This run is the re-baseline that adding the curriculum required.

**Re-measure if:** the episode length or termination set changes — both move what
"survived" means.

**History:**
- 2026-09-01 — added to both tasks so the residual-feedback comparison stays
  like-for-like; validated over a full run, and neutral on ResidualMPC.

## feedback_scale

**Current:** `0.02` rad (about `1.15` degrees) at a saturated action, applied per
residual joint to the encoder vector in `SimControllerBridge._fill_joint_columns`.

**It is a lie told to a balance controller, so it is bounded.** The offset makes
mc_rtc believe a joint is somewhere it is not; too large a value and the
stabilizer solves for a state the robot is not in. `0.02` rad is double the
encoder-bias randomization the controller already tolerates
(`bias_range` `+-0.01`), so it is inside the range the controller is known to
absorb while still being large enough to steer a solve.

**Gated with the torque residual.** The offset is multiplied by the same
`last_gate` the torque residual uses, so a suppressed residual cannot keep
misreporting joint positions.

**Hardware caution:** this channel falsifies proprioception. In simulation that
is a research knob; on a real robot a false state estimate reaching a stabilizer
is a fall, and the scale is not a safety bound in any certified sense.

**Swept, and there is no usable window.** All twelve joint channels held at `+1`,
fresh environment per scale, `18 s` scored after a `12 s` settle:

| scale (rad) | `qp` | `vx` | `z_min` | falls | **MPC stability errors** |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | -11,265 | 0.2340 | 0.7508 | 4 | **0** |
| 0.002 | -11,265 | 0.2338 | 0.7508 | 4 | **0** |
| 0.005 | -11,300 | 0.2346 | 0.7509 | 4 | 3 |
| 0.01 | -11,313 | 0.2347 | 0.7507 | 4 | 3 |
| **0.02** (default) | -11,644 | 0.2389 | 0.7505 | 4 | 4 |
| 0.05 | -10,452 | 0.2333 | **0.5600** | 5 | **18** |

`0.002` is inert to four decimals on every measure. From `0.005` the ISMPC begins
emitting `MPC result is too far from stability condition`, and by `0.05` it
collapses eighteen times and minimum root height falls to `0.56 m` — the robot is
going down. There is no scale at which this channel does something without
breaking the solver.

**Read the error count, not `qp` or `vx`.** Those barely move from `0.002` to
`0.02` while the solver is already failing, because `run()` keeps returning true
after that error (`CLAUDE.md`, mc_rtc gotchas). A collapsed MPC leaves almost no
trace in the aggregate metrics; height and the error count are the honest signals.

**Caveat on the worst case.** This holds all twelve joints saturated in one
direction. A trained policy applies a smaller, varied pattern, so the practical
threshold may sit higher — confirming that needs a training run at `0.005`, which
is exactly the gap that made the `5 N` wrench result misleading.

**Re-measure if:** the encoder bias randomization changes, or a robot other than
HRP5P is used — the tolerable offset is a property of the stabilizer's gains.

**History:**
- 2026-09-01 — set at twice the encoder-bias range; not yet swept.

## torque_channel

**Current:** `True`, giving Ranjbar's hybrid. Setting it `False` zeroes the
torque residual and leaves the feedback channel alone, which is the paper's
feedback-only variant.

**The action layout is `[residual, walking_reference, feedback]`**, with the
feedback block appended last so the base class keeps owning the columns it
already slices. `process_actions` removes the trailing block before delegating.

**Re-measure if:** another action channel is appended — it must go before the
feedback block or after it consistently, since both ends are now sliced.

**History:**
- 2026-09-01 — hybrid chosen as the default because it is the paper's best
  variant on combined uncertainty.
