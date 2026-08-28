# Residual authority

How much the policy is allowed to add on top of mc_rtc, in the control channel's
own unit (rad for position, Nm for torque — one number cannot serve both).

`scale` maps the policy's ~unit output into that authority and `clip` makes it a
hard bound. A residual able to outvote the controller is how the policy learns to
freeze the gait instead of stabilizing it: a swing trajectory is ~0.5 rad, and
rejecting a push needs far less.

Set in `tasks/residual_balance/residual_balance_env_cfg.py`; applied in
`actions/mc_rtc_residual_action.py`.

## residual_scale

**Current:** `0.01` rad for position control, `10.0` Nm for torque. The position
value is the measured reference after both larger-authority experiments failed.

The bound is set in position units, so what it really buys is torque. The
actuators and torque action now enforce RobotModule hardware bounds, and the
position action projects requested targets into the module's position limits.
The `0.01` rad bound was 22-27% of every leg joint's hardware
limit. The rejected `0.20` rad bound could request roughly 4.4-5.4x that limit
before the torque-margin penalty responded.

**Re-measure if:** PD gains, torque limits or the residual joint set changes.
The projection rate is now reported explicitly; a non-zero settled rate means
the configured authority asks for action the hardware cannot deliver.

**History:**
- That warning was then ignored. A run at `0.1` (20899 iterations, 2026-07-31)
  put a saturated residual at **220-270% of the hardware limit**, and — because
  `scale` multiplies the *exploration* noise too — left the policy's own dither at
  115-140% of it once `mean_std` had grown to 0.52. Both tracking terms read
  ~0.008 of a possible 1.0 for the whole run, against 0.68/0.80 for the
  zero-residual baseline, from the first iteration onward and with a near-zero
  mean residual: **the environment was broken before the policy did anything.**
- 2026-08-03 — back to 0.01.
- 2026-08-15 — raised to `0.03` on the authority argument below, bundled with
  `gamma = 0.997`. **Refuted at 1150 iterations and reverted the same day.** More
  authority did not buy tracking; it cost it. Against the zero-residual baseline
  (n = 112/arm, deterministic policy) the per-step deficit *widened* to
  **zmp_tracking -16.7%, com_velocity_tracking -25.2%, recovery_tracking -16.5%**,
  against -9.9% / -8.1% / -16.6% at scale `0.01`. Not a state-distribution
  artifact: it holds in every episode-length band, including survivors-only where
  both arms ran identical 4500-step episodes (-15.1%, p = 6e-8). Nor is it
  exploration dither — `compare_to_baseline.py` runs
  `runner.get_inference_policy()`, the distribution mean.
  The horizon half of that bundle *did* work and was kept; see
  [reward-shaping.md](reward-shaping.md#residual-harm-at-gamma099).
- 2026-08-20 — raised to `0.20` for a 20x authority sweep. **Refuted and reverted
  on 2026-08-21.** Across models 500, 960 and 999, deterministic per-step tracking
  lost 13--35% against each checkpoint's own zero-residual arm. The deficit held
  in every episode-length band, including survivors, while the final policy's
  hazard was 1.27x baseline. More authority was actively harmful.

## residual_scales

**Current:** `{".*": residual_scale}` — uniform, which is exactly the previous
scalar behaviour written in the form that lets the ankles (which is what actually
moves the centre of pressure) be raised without also loosening the hips, once the
authority probe says which joints are worth it.

Per-joint authority is expressed once and used for both the scale and the clip so
the two cannot disagree: `processed = raw * scale` is clipped afterwards, so a
clip left at the old scalar would silently cap any joint given a larger scale.

**The footgun:** the patterns must partition **every actuator**, and only half of
that is enforced. mjlab's `resolve_matching_names_values` raises if a joint
matches two keys and if a key matches nothing, so a specific entry cannot sit
alongside a `".*"` catch-all — write it as
`{"[RL]A[PR]": 0.03, "(?![RL]A[PR]$).*": 0.005}`.

Note the pattern: matching is `re.fullmatch`, and HRP5P's residual joints are
`RCY RCR RCP RKP RAP RAR` and their `L` twins — so the ankles are `[RL]A[PR]`.
An `[LR]_ANKLE_.*` spelling, which this file carried until 2026-08-18, matches
**nothing** on this robot and raises. Joint names are robot-specific; read them off
`residual_actuator_names` rather than assuming.

What it does **not** raise
on is a joint matching no key at all: `BaseAction.__init__` seeds `scale` to ones
and `clip` to +/-inf, so a residual joint left out silently gets scale 1.0 — 100x
the intended authority — with no clip. That is the 220-270% failure above,
reachable by omission.

Note the resolution is against every actuator matched by `actuator_names`, not
just `residual_joints`; the residual subset is sliced out afterwards in
`_setup_residual`.

The action term exposes the bounded normalized request, scaled physical request,
executed physical residual, authority gate, and feasibility mask separately.
Rewards price the bounded request only while authority is nonzero, so a partial
gate or later projection cannot hide a large active request. Metrics report the
requested and executed quantities separately.

## ankle_pitch

**Current:** `(LAP, RAP)` for HRP5P — the `ankle` set with ankle roll removed.
It exists because direction-resolved qualification showed roll doing active harm.

**The measurement.** `model_180` of the matched-impulse seed, paired against the
controller over 16 environments and two seeds, split by push direction in the
`finite_impulse` scenario:

| stratum | episodes | gate duty | grounded DCM error | foot slip |
| --- | ---: | ---: | ---: | ---: |
| recovery/backward | 69 | `-7.6%` | **`-8.9%`** | `+27.8%` |
| recovery/forward | 54 | `+6.3%` | **`-3.7%`** | `-16.7%` |
| recovery/right | 72 | `+10.2%` | `+5.6%` | **`+442%`** |
| recovery/left | 62 | `+25.1%` | `+11.7%` | `+20.8%` |
| sustained/none | 104 | `+31.8%` | `+38.1%` | `-8.4%` |

Both sagittal recoveries improve and both lateral ones degrade, and the gate duty
moves the same way: the policy claims *more* authority on exactly the directions
it makes worse. Foot slip rising 442% on right-side recoveries is the mechanism —
it is scrubbing the stance foot sideways.

`ankle` is `leg[-2:]`, which for HRP5P is `(LAP, LAR)` and `(RAP, RAR)`: ankle
pitch and ankle **roll**. Roll is the lateral actuator, sagittal recoveries are
pitch, and the split falls exactly on that line.

**But roll magnitude does not predict the harm, and the first draft of this
section implied it did.** Executed physical residual RMS per joint, x1000 rad:

| stratum | LAP | LAR | RAP | RAR | roll total | DCM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| recovery/backward | 0.584 | 0.615 | 0.878 | 0.658 | 1.273 | `-8.9%` |
| recovery/left | 0.664 | 0.376 | 0.660 | 0.859 | 1.235 | `+11.7%` |
| recovery/forward | 0.798 | 0.346 | 0.896 | 0.538 | 0.884 | `-3.7%` |
| recovery/right | 0.736 | 0.321 | 0.895 | 0.575 | 0.896 | `+5.6%` |

Backward and left use nearly identical roll (`1.273` against `1.235`) for
opposite outcomes. Roll is `40.8%` of all executed residual. So the defect is not
that roll is over-used; it is that **the policy's lateral control law is wrong,
and roll is the only actuator that can produce a lateral centre-of-pressure
shift.** Sagittal pushes are recovered by pitch and are largely indifferent to
what roll does.

**What that predicts for `ankle_pitch`, stated before the run.** Removing roll
removes the ability to help laterally as well as the ability to harm, so the
lateral strata should move to roughly baseline — `0%`, not negative — while the
sagittal ones keep their gains. The recovery-DCM average over four directions
would then be near `(-8.9 - 3.7 + 0 + 0) / 4 = -3.2%`, which is **still below the
`5%` promotion gate**. The ablation is worth running because it should also close
the `sustained/none` feedback loop and the `+442%` foot slip, but on its own it is
not predicted to clear that gate at the qualifier's default magnitudes.

**Why this also explains the failed hazard gate.** In `sustained/none` — ordinary
walking between pushes — the detector fires `31.8%` more often and grounded DCM
error degrades `38.1%`. That is a loop: lateral residual disturbs the gait, the
disturbance raises the detector score, the higher score grants more authority,
which disturbs more. One root cause reaches every gate the checkpoint failed,
which is why the next arm changes authority rather than the band mixture.

**This is a point-estimate pattern, not a significance claim.** Those strata are
summarized per direction, and each interval is narrow only in the sense that the
sign is consistent across four independent strata and roughly 250 episodes. Treat
the direction of the effect as the finding and the magnitude as provisional.

**The seed ran 856 iterations and the training curves are flat.** Smoothed over
60 iterations, `zmp_error` sat in `0.0394-0.0417` for the whole run and
`recovery_dcm_error` in `0.0137-0.0156`. Both best windows land late (iterations
753 and 758), so the run never clearly peaked, but a 6% band over 856 iterations
is wandering rather than learning. `policy_mean_rms` climbed monotonically
(`0.070` at 100, `0.136` at 300, `0.201` at 856) while the hazard union drifted
from its best `0.4153` to `0.4860`, so the policy kept getting more assertive
without the metrics following. It was stopped at 856 rather than 1500 on that
basis.

Against the four-joint `ankle` arm on the identical mixture (`impulse_speed`
`0.142` against `0.143`), the two-joint policy fell less (`0.475` against
`0.542`), tracked recovery slightly better (`0.01492` against `0.01583`) and used
*less* residual (`0.00513` against `0.006623`). Those arms stopped at different
iteration counts, so this is suggestive, not controlled.

**It also exposed a run-killing numerical defect**, unrelated to authority: the
first attempt died at iteration 262 when one saturated action overflowed the PPO
ratio and advantage masking turned the resulting `inf` into `NaN`. See
[ppo.md](ppo.md#LOG_PROB_FLOOR); the fix let the second attempt pass 13 KL spikes
above `0.05`, peaking at `0.151`, without dying.

**Paired qualification: the nominal gate flipped, the recovery gate did not.**
`model_750` was scored with all four scenarios valid over 32 clusters:

| gate | four-joint `ankle` | `ankle_pitch` |
| --- | --- | --- |
| `nominal` foot slip | `0.000047 -> 0.000048`, failed twice | `0.000047 -> 0.000046`, **passes** |
| `nominal` ZMP | not reported clean | `0.03212 -> 0.03212` |
| recovery DCM | `+0.64%` | `+1.40%`, CI `[-0.0043, +0.0077]` |
| hazard ratio | `0.989` | `1.016` |

Dropping roll did what the direction-resolved read predicted for the gait: foot
slip now *improves* and ZMP is unchanged to five decimals, so the lateral
actuator was the source of the nominal damage. Nothing else improved. The
prediction recorded above — lateral strata moving to about baseline for a
four-direction mean near `-3.2%` — was wrong: recovery DCM went the wrong way at
`+1.40%`, so the sagittal gains did not survive the ablation either.

**Re-measure if:** the authority set, detector calibration, residual scale, or
push direction distribution changes.

**History:**
- 2026-08-28 — trained seed 42 for 856 iterations at `dcm_stability 0.5`; the
  curves are flat and the run exposed the `LOG_PROB_FLOOR` defect.
- 2026-08-27 — added after direction-resolved qualification separated sagittal
  improvement from lateral degradation in the same checkpoint.

## residual_joints

**Current:** the legs only. Balancing is what this task rewards, and the upper
body does not hold the robot up.

This is the task's opinion, so it lives in the env cfg — a locomotion or
manipulation task would want the arms. The robot's own `get_residual_joints`
stays the place for exclusions that hold whatever the task is: a joint mc_rtc
models as fixed can carry no residual anywhere, and is dropped there. Filtering
that set rather than taking the legs directly keeps the robot's carve-outs and
refJointOrder ordering.

## recovery_detector_path

**Current:** `etc/recovery_detector.json`. Residual authority is zero until a
deployable sensor detector opens it. Its monotonic magnitude score is the maximum
normalized command-relative DCM error, base angular speed, tilt, and total
foot-load deviation. A calibrated rise in base angular speed starts a smooth,
bounded recovery burst; another burst requires `0.5 s` back inside the calibrated
nominal-score envelope. The push schedule, time since push, and critic-only
observations are not action inputs.

`scripts/calibrate_recovery_detector.py` uses separate nominal and disturbed
environment cohorts, splits both into train and held-out environments, and uses a
fixed-magnitude `0.4 m/s` planar kick with uniform direction. It also asserts that
the independently implemented detector DCM agrees with `mdp._ZmpSensors` within
`0.1 mm` at every recorded step.

The accepted 2026-08-24 calibration used 16 environments x 1500 steps:

| split | nominal duty | recovery recall, first 2 s | authority >5% after 2 s |
| --- | ---: | ---: | ---: |
| train | 0.364% (n=4496) | 98.993% (n=993) | 0.000% |
| held out | 0.000% (n=4620) | 96.825% (n=882) | 0.000% |

The filter attacks in `0.1 s`, decays with a `0.5 s` time constant, and hard-cuts
each burst at `2.0 s`. `scripts/verify_live_recovery_detector.py` drove nonzero
actions through the simulator and measured maximum authority `1.000`, mean
authority `7.338%` under fixed-energy kicks, and exactly zero executed residual
while inactive.

**Re-measure if:** the gait, command speed, sensor model, control period, robot,
or disturbance-energy distribution changes.

## Removed coherence gate

**Removed 2026-08-24.** This whole-vector coherence gate was a no-op at
`gate_strength = 0.0` and is superseded by `recovery_detector_path`. The history
below records why directional gating was considered and the measurements made
before removal.

`scale` and `clip` bound **how much** the residual may do. This bounds **where and
when**, which the literature says matters far more. Jayasinghe et al. ablate a
residual recovery controller on a Go1 at 1.15x mass and report TTR-50 (lower
better): full system 168, **no directional alignment 3367**, no transient
filtering 1127, no dual-timescale 186, no gain modulation 174. Directional
alignment is twenty times the next-largest term. Their conclusion: "mechanisms
regulating where and when residual authority is applied are more critical than
those governing adaptation rate... Even a simple linear residual remains effective
when bounded and aligned, whereas unconstrained correction destabilizes recovery."

Our own runs say the same thing from the other direction:
`Episode_Reward/residual_magnitude` grows monotonically in **every** run
(-0.0126 -> -0.0603 in `scale01`; -0.0087 -> -0.0728 in `dcm-obs`). The *reward* is
gated on the post-push window by `recovery_dcm`; the *action* has never been gated
at all, so the residual acts on every step including the ~90% of nominal walking
where it can gain nothing and can still lose something.

**The removed shape** was:

```
cos    = <residual, alpha> / (|residual| |alpha|)
active = tanh(|alpha| / GATE_ALPHA_REF)
gate   = 1 - GATE_STRENGTH * relu(-cos) * active
```

Three properties it is built for, each of which the naive version gets wrong:

- **It can only remove authority**, never add — `gate <= 1` by construction. This
  is what makes gating on a *measured* quantity safe here when the same trick on
  the reward would be perverse: it withholds capability rather than granting
  payment, so there is no state the policy can steer into to be paid more. Compare
  `RECOVERY_TRACKING_WEIGHT` in [reward-shaping.md](reward-shaping.md), which is
  gated on the push *schedule* for exactly that reason.
- **`active` covers the degenerate case, which is most steps.** At every joint
  reversal and throughout double support `alpha -> 0`, the cosine is meaningless
  noise, and an ungated version would fire at random precisely when the controller
  is asking for nothing. Verified: at `|alpha|` of 1e-6 the gate reads 0.999997.
- **Aligned residuals are untouched.** `relu(-cos)` is 0 for `cos >= 0`, so this
  only ever attenuates opposition, never ordinary help.

It gates against the **interpolated** alpha, not `controller_reference("alpha")` —
that accessor returns the un-interpolated next target, and gating against a
different alpha than the one being tracked injects a substep-frequency artefact.

**Measured before committing to it**, because a gate with nothing to attenuate is
a wasted run. Under `dcm-obs` `model_1850`, 24 envs x 2500 steps, 60000 env-steps:

| | share `cos < 0` | mean `cos` |
| --- | --- | --- |
| all steps | **58.9%** | -0.047 |
| `\|alpha\| >= 0.5` | 58.5% | -0.038 |
| `\|alpha\| >= 1.0` | **64.7%** | -0.075 |

So the residual opposes the plan on most steps, and *more often* the faster the
gait is moving — which is the worst time for it. But the mean cosine is only
-0.047: it is largely **orthogonal** to the plan with a systematic opposing tilt,
not fighting it head-on. The gate therefore has real work to do without being
destructive; at `GATE_STRENGTH = 1.0` it withholds ~10% of authority on average.

Expect the realised effect to **shrink over training**: this was measured on a
policy trained without the gate, and once opposition costs authority the policy
should learn to align. A gate whose measured attenuation stays flat across a run
is one the policy is ignoring.

## Removed coherence alpha reference

**Removed 2026-08-24 with `gate_strength`.** Its historical value was `0.5`
rad/s, measured from the norm of the controller's joint-velocity reference over
the 12 residual joints.

Measured over the same 60000 env-steps: mean 0.848, median 0.845, p25 0.514,
p75 1.222, p90 1.390. There is no idle mode to speak of — even the 25th percentile
is 0.51 — so `active` saturates over almost all of normal walking and only relaxes
for genuinely still joints:

| `\|alpha\|` | 0.2 | 0.514 (p25) | 0.845 (median) |
| --- | --- | --- | --- |
| `active` at ref 0.5 | 0.380 | 0.773 | **0.934** |
| `active` at ref 1.0 | 0.197 | 0.473 | 0.688 |

`0.5` is chosen so the gate is fully effective during ordinary gait (0.93 at the
median) while still standing down where the cosine stops meaning anything. `1.0`
would blunt it across the whole operating range, which defeats the point.

## Does the residual have enough authority?

`scripts/probe_residual_authority.py` answers this directly: it drives a
**constant** residual instead of a policy and measures `mdp.zmp_error`. A constant
offset is the bluntest possible input — if a full-scale one does not move the
centre of pressure, nothing a policy does will either, and the reward is not the
binding constraint.

Authority is adequate if a full-scale constant residual shifts the error by
>= 0.007 m (20% of the ~0.036 m operating error).

**Measured 2026-08-15, and it falls well short.** 32 envs x 10 min each, ~0.3-0.5 M
grounded samples per run, settled operating error 0.0364 m:

| run | `zmp_error` mean | shift vs level 0 | % of the 7 mm threshold |
| --- | --- | --- | --- |
| `--level 0` | 0.04920 +/- 0.00019 | — | — |
| `--level 1.0 --pattern alternating` | 0.05125 +/- 0.00037 | **+0.00205 m** | 29% |
| `--level 1.0 --pattern all` | 0.05016 +/- 0.00017 | **+0.00096 m** | 14% |

At `residual_scale = 0.01` a saturated residual moves the centre of pressure by
about **2 mm against a 36 mm operating error — 5.6% authority against a 20%
criterion**. The shift is real (~5 sigma) but small, and a coordinated bias is
*weaker* than an alternating one, so the pattern is not what limits it.

**Caveat on the instrument.** A *constant* residual is a static posture offset,
and the stabilizer is a feedback loop that actively absorbs one, so this measures
steady-state authority against an opposing controller — close to a worst case. A
residual acting transiently in the 200 ms after a push may have more leverage
than this shows. What argues against reading it that way is the trained policy,
which had exactly that dynamic freedom and used it to make tracking 10% *worse*
(see [reward-shaping.md](reward-shaping.md#residual-harm-at-gamma099)).

**What this implied for the scale — and why it was wrong.** Authority should scale
roughly linearly, and torque does: 0.01 rad is 22-27% of the hardware limit and
5.6% authority, so ~0.03 rad should be ~66-81% of the limit and ~17% authority —
the first scale approaching the criterion while staying inside the hardware. That
argument was acted on 2026-08-15 and **the training result contradicted it**: at
0.03 the deficit widened to -16.7% rather than closing (History above). So the
probe measures what it says — steady-state authority — but authority is not the
binding constraint, and the criterion should not be read as a target to reach by
raising the scale. Do not raise it again on this reasoning alone.

The surgical variant is per-joint: the ankles are what actually move the centre
of pressure, and `residual_scales` already supports
`{"[RL]A[PR]": 0.03, "(?![RL]A[PR]$).*": 0.005}` — raising ankle authority
without loosening the hips. Which joints deserve it is not measured; the probe
drives every residual joint uniformly and offers no joint-subset pattern.

The question arose because the reward proved blind to the policy — see
`RECOVERY_TRACKING_WEIGHT` in [reward-shaping.md](reward-shaping.md).

## AUTHORITY_SETS

**Current:** `uniform` and `ankle` are registered by default. The completed
`sagittal` and `hardware` screens are available only through
`MC_MJLAB_REGISTER_ARCHIVED_TASKS=1`. Ankle authority selects the last two joints
of each leg. Sagittal authority selects hip pitch, knee pitch, and ankle pitch.
Hardware and uniform retain all twelve leg joints. Selection is derived from each
mc_rtc robot module rather than HRP5P names typed into the task.

For every non-uniform position set, per-joint authority is
`min(0.01 rad, 0.20 * effort_limit / kp)`. Torque uses
`min(10 Nm, 0.20 * effort_limit)`. The configuration emits an exact scale and
clip entry for every actuator so an unmatched non-residual actuator cannot fall
back to mjlab's unit scale.

**Re-measure if:** robot module joint groups, PD gains, effort limits, or control
mode changes.

**History:**
- 2026-08-27 — retained ankle as the active policy task and moved sagittal and
  hardware registrations behind the archive compatibility switch.
- 2026-08-24 — HRP5P hardware-normalized position scales are `0.009293` rad
  (hip yaw), `0.009318` (hip roll), `0.007527` (hip pitch), `0.007886` (knee
  pitch), `0.007602` (ankle pitch), and `0.010000` (ankle roll), symmetrically
  left/right.

## probe_selective_authority

**Current:** `scripts/probe_selective_authority.py` runs paired zero and pulse
environments for individual joints and anatomy groups, measuring the two-second
signed and absolute DCM response, centre-of-pressure shift, and peak effort
change relative to the hardware limit. It disables disturbances and the recovery
detector so the requested pulse is the only residual input.

**Re-measure if:** controller, gait, pulse duration, residual scale, gains, or
robot changes.

**History:**
- 2026-08-24 — full HRP5P position probe, 30 environments, 10 s settle plus 2 s
  positive full-scale pulse. Individual absolute DCM changes ranged
  `0.01352-0.02166 m`; CoP shifts `0.02615-0.04974 m`; effort changes
  `0.074-0.179` of hardware limit. Group results were ankle
  `0.01864/0.02689/0.052`, sagittal `0.01732/0.02640/0.060`, and all
  `0.01756/0.04880/0.157` for absolute DCM change / CoP shift / effort change.
  The groups therefore have measurable leverage, but only the fixed-budget
  training and qualification screen may promote one.
