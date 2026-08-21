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

**Current:** `0.20` rad for position control, `10.0` Nm for torque. The position
value was raised at the user's request on 2026-08-20 and is **unmeasured**; it is a
future-run default only, not a change to the running archived continuation.

The bound is set in position units, but the actuators are unlimited on purpose
(mc_mujoco parity, see `pd_actuator_configuration`), so what it really buys is
torque. The previous `0.01` rad bound was 22-27% of every leg joint's hardware
limit; `0.20` rad can therefore request roughly 4.4-5.4x that limit before the
torque-margin penalty responds. Do not call it safe until the authority probe and a
short training smoke run have been repeated.

**Re-measure before using this in a training decision.** Nothing in the sim clamps,
so a residual that outgrows the hardware is invisible here and divergent on the robot.

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
- 2026-08-20 — raised to `0.20` at the user's request. This is a 20x authority
  sweep, not an evidence-backed tuning decision; re-measure before interpreting a
  training result as a benefit of the larger bound.

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

## residual_joints

**Current:** the legs only. Balancing is what this task rewards, and the upper
body does not hold the robot up.

This is the task's opinion, so it lives in the env cfg — a locomotion or
manipulation task would want the arms. The robot's own `get_residual_joints`
stays the place for exclusions that hold whatever the task is: a joint mc_rtc
models as fixed can carry no residual anywhere, and is dropped there. Filtering
that set rather than taking the legs directly keeps the robot's carve-outs and
refJointOrder ordering.

## gate_strength

**Current:** `0.0` in the action term (a no-op), opted into by the task. How much
authority to withhold from a residual pointing *against* the controller's own
commanded joint velocity.

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

**The shape**, in `McRtcResidualActionBase._coherence_gate`:

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

## gate_alpha_ref

**Current:** `0.5` rad/s, measured. The norm of the controller's joint-velocity
reference **over the 12 residual joints** — not the per-joint figure of ~0.4 rad/s
in `CLAUDE.md`, which is a different quantity.

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
