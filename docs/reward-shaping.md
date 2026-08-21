# Reward shaping

The reward must pay for stability the base controller cannot buy itself, not for
agreeing with it and not for merely surviving. The dense terms do most of the
work: at `gamma=0.997` the horizon is ~333 steps (6.7 s), long enough that the
termination penalty now reaches the residual's own delayed consequences, but still
short against a 90 s episode.

**Reorganised 2026-08-17.** The objective is `dcm_stability` — the divergence rate
of the LIPM's divergent component from the centre of pressure — with
`recovery_dcm` paying the same quantity on the window after a push, and
`angular_momentum` and `foot_slip` penalising two things the stabilizer QP's model
does not regulate.

`zmp_tracking` and `com_velocity_tracking` remain, demoted from 0.5 to 0.05: they
score the measured state against mc_rtc's *own plan*, which the QP already
optimises, so as the objective they paid the residual to reproduce what the base
controller is built to deliver. They are a "don't fight the controller" prior now.
Both were sized against the zero-residual baseline and both collapse in the run-up
to a fall (to 0.12 and 0.10 of a possible 1.0), so they remain useful diagnostics.

The earlier claim here that a separate DCM term was unnecessary — that the
divergent mode is a linear combination of the two, so any DCM weighting is
reachable by choosing their weights — was **wrong in a way that mattered**. It is a
linear combination of *measured CoM and CoM velocity*, but those terms score
`measured - planned`, not the measured state itself. No choice of weights over two
plan-tracking errors expresses a plan-independent stability margin.

Constants in `tasks/residual_balance/residual_balance_env_cfg.py`; terms in
`tasks/mdp.py`.

## ZMP_TRACKING_STD

**Removed 2026-08-19** with its reward term; see `Pruning the agreement rewards`.
The sizing argument is kept because it is the template every later `std` followed.

Was `0.05` — sized off the zero-residual baseline rather than picked.
Measured over 64 s x 16 envs, the CoM-to-ZMP offset error runs median 2.1 cm,
p75 4.4 cm, p90 9.0 cm. `std` is where the exponential has fallen to 1/e, so
5 cm scores that baseline 0.68 on average: ordinary walking is well paid, a push
landing (p90) drops payment to 0.02, and there is a third of the term left for
the residual to earn.

**Re-measure if:** the baseline's operating error moves. Tighten `std` and the
signal is mostly noise (0.02 scores 0.41); loosen it and it saturates (0.10
scores 0.84).

## ZMP_TRACKING_WEIGHT

**Removed 2026-08-19.** `zmp_tracking` is no longer a reward term; the quantity
survives as the `zmp_error` metric. See `Pruning the agreement rewards` below.

**History:**
- `1.0` originally.
- Halved when `recovery_tracking` was added. The two are the same quantity and
  the same kernel; splitting the weight keeps the total tracking payment roughly
  where it was while moving half of it onto the steps that follow a push, where
  the residual can actually change the outcome. Nominal steps therefore still pay
  (~0.0125 per step against ~0.001 of penalties), which is what stops the split
  from creating an incentive to end the episode early.

## COM_VELOCITY_TRACKING_STD

**Removed 2026-08-19** with its reward term; see `Pruning the agreement rewards`.
The two-scale reasoning below is why `com_velocity_error` is reported as a plain
norm rather than through a kernel.

Was `0.05` horizontal, `COM_VELOCITY_TRACKING_STD_VERTICAL = 0.005`.
Over 96 s x 16 envs (131 episodes, 58 of them falls) the error runs median
1.2 cm/s and mean 3.3 cm/s.

Two scales, not one, because the axes are not comparable: measured per axis the
horizontal error runs median 1.17 cm/s and the vertical 0.10 cm/s. Under one
shared kernel the vertical channel scores 0.94 and is effectively free — which
silently cost the term its whole reason for existing, since crouch-collapse is a
*vertical* failure. At 5 cm/s horizontal and 0.5 cm/s vertical the two channels
score 0.79 and 0.81 on the baseline, so both keep comparable headroom and neither
saturates the other. The total, 0.80, lands where the old single kernel was
(0.785), so the weight carried over unchanged.

**Why the term is worth having at all:** in the last second before a fall the
error is 0.347 m/s against 0.022 m/s before a time-out — a **16x separation,
against 5x for the ZMP error**. It is the sharpest early warning of the two.

Horizontal stays a 2-norm rather than two more kernels: x and y are
interchangeable for balance, so the term should not care which way the robot is
drifting.

## COM_VELOCITY_TRACKING_WEIGHT

**Removed 2026-08-19**, and replaced by the `com_velocity_error` metric. This is the
term that was negative against baseline in **every** comparison ever run, so it is
kept as a diagnostic. See `Pruning the agreement rewards` below.

**History:**
- `1.0`, kept deliberately equal to the ZMP term because the two are
  complementary halves of one state, not two views of the same thing.
- Halved alongside the ZMP term when `recovery_tracking` arrived — but *not* for
  the same reason. The ZMP half moved to the gated term; there is no gated
  CoM-velocity term for this half to move to, so it was simply removed. Effective
  per-second payment is now ~0.83 for ZMP (0.5 plus ~0.33 from a 2 s window on a
  5-7 s cadence) against 0.5 for CoM velocity, which down-weights the sharper of
  the two signals.
- 2026-08-14 — reviewed and kept at 0.5 as a deliberate de-emphasis. The
  alternatives, if it is revisited: back to 1.0, or add a
  `recovery_com_velocity` mirroring `recovery_tracking`'s gate.

## dcm_stability

**Current:** weight `1.0`, `DCM_STD = 0.05` (measured, see below). The task's
primary dense reward since 2026-08-17, replacing the plan-matching pair.
**Command-relative since 2026-08-19** — the correction below is the whole of that
change.

**What it scores.** With the divergent component of motion
`xi = com_xy + com_vel_xy / omega` and `omega = sqrt(g / com_z)`, the linear
inverted pendulum gives `d(xi)/dt = omega * (xi - CoP)`. So `norm(xi - CoP)` is
not a proxy for instability — it **is** the divergence rate, in metres. But
steady walking at `v` *requires* `xi - CoP = v / omega`: that offset is what
produces the motion, not an error. The scored quantity is therefore

```text
error = norm( (com_vel_xy - commanded_com_vel_xy) / omega - (CoP - com)_xy )
```

with the command taken from the controller's own control robot
(`controller_vector("control_com_vel")`, the same reference `com_velocity_error`
uses). The reward is `exp(-(error/std)^2)`, gated on the feet carrying load
exactly as `zmp_tracking` is. Standing is unaffected by the correction — its
command is zero to four decimals — so this is purely a walking-side change.

**What it bought, measured 2026-08-19** by
[validate_dcm_objective.py](evaluation.md#validate_dcm_objectivepy), 16 envs x
1200 steps per regime, zero residual, 17,599 grounded samples:

| walking arm | mean | median | p75 | p90 | p99 |
| --- | --- | --- | --- | --- | --- |
| corrected | 0.0388 | 0.0317 | 0.0483 | 0.0666 | 0.1501 |

At `DCM_STD = 0.05` the walking score rises from **0.538 to 0.611** while standing
stays at 0.785, so the standing edge falls from **+45.9% to +28.4%**. A second run
that day, same settings, read +51.2% -> +34.4%. The correction removes roughly a
third of the bias and **does not** meet the 10%-band gate the proposal set; see
*dcm_stability standing bias* below for why that band was the wrong test and what
the right one says.

**The test that does decide it** bins the walking arm by how far the measured CoM
speed sits from the commanded one — the gradient a residual can actually move:

| measured - commanded, m/s | n | corrected | legacy |
| --- | --- | --- | --- |
| slower than -0.05 | 242 | 0.0936 | 0.0624 |
| -0.05 .. -0.02 | 1189 | 0.0624 | 0.0581 |
| -0.02 .. +0.02 | 15506 | **0.0322** | **0.0399** |
| +0.02 .. +0.05 | 291 | 0.0596 | 0.0789 |
| faster than +0.05 | 371 | 0.1842 | 0.2256 |

Lagging the command by >0.05 m/s now costs **2.91x** the on-command error, against
**1.56x** before. That is the defect closing: the term used to be nearly
indifferent to a robot dragging its own gait, and now charges it. The minimum sits
at the commanded speed rather than at zero velocity, which is the whole point.

**The reference is not perfectly exogenous, and that is the residual risk.** A
full-scale alternating residual moved the commanded speed by -9.2% in one run and
+0.5% in the other, against error differences of 2-3x across the bands above. So
the residual can nudge the target it is scored against, but two orders less than
it can move the state. Re-check with `--residual-level` if `residual_scale` grows.

**Why it replaced plan-matching.** `zmp_tracking` and `com_velocity_tracking`
score the measured state against mc_rtc's *own plan*, which the stabilizer QP is
already solving for — so the residual was paid to reproduce what the base
controller is built to deliver, and across four runs at three scales and two
discount horizons it never once beat a zero residual on them. This scores the
robot's actual dynamic margin instead, which the plan cannot buy itself.

**That case is weaker than it looked, and this section was written before the
evidence against it.** The fifth run — `zeroinit-4ev`, the first with a healthy
learning rate — **did** beat the baseline on plan-matching. At `model_1000`,
n=136/arm: per-step `zmp_tracking` **+3.7%** (p = 0.003), positive in all four
episode-length bands including survivors-only, with survival **+11.5pp**
(p = 0.029). So the plan-matching objective is improvable after all; "four runs
never beat it" was measuring a pinned optimiser, not an impossible objective.

What is still true is that the win was **small and transient**. By `model_2900` the
same measurement read **-5.5%** (p = 1.3e-09) and survival had fallen to +5.8pp
(p = 0.14), matching a training curve whose best smoothed `zmp_error` was at
iteration 1320 and which decayed for 1600 iterations afterwards. Read together:
the objective has a little headroom, a healthy optimiser finds it inside ~1300
iterations, and then overtraining gives it back.

**So the open question this term was introduced to settle is not settled.** Whether
`dcm_stability` beats a plan-matching run *capped at ~1300 iterations* is untested,
and that is the comparison that decides whether this redesign was needed.

The premise that there was simply nothing to win is **wrong**, and that is worth
recording. Weights are per second and scaled by `step_dt = 0.02`, so each
tracking term's ceiling is `0.5 * 0.02 = 0.01` per step. The zero-residual
baseline reaches 0.00664 (66%) on ZMP and 0.00790 (79%) on CoM velocity, which
back-solves to a 0.032 m operating error — consistent with
[residual-authority.md](residual-authority.md). A third of the ZMP reward is
unclaimed. The residual was not failing to find headroom that does not exist; it
was failing to find headroom that does, presumably where the QP is constrained or
where its model disagrees with the sim.

**It reuses `_ZmpSensors` entirely.** `measured_offset` already returns
`CoP - CoM` in xy, so `xi - CoP = com_vel_xy/omega - measured_offset` and no new
sensor plumbing was needed. `measured_offset` is now the memoised call (five terms
read it per step), rather than `offset_error` as before. The command-relative
form only subtracts one more term inside `dcm_offset`, which is why `dcm_error`,
`dcm_stability` and `recovery_dcm` all moved together and all now take
`action_name`.

## DCM_STD

**Current:** `0.05` — measured off the zero-residual baseline, not copied from
`ZMP_TRACKING_STD` despite landing on the same number. **Re-estimated 2026-08-19
against the corrected error and kept**, which is not the same as retaining it by
default:

| corrected walking error | mean | median | p75 | p90 | p99 |
| --- | --- | --- | --- | --- | --- |
| 16 envs x 1200 steps, n=17,599 | 0.0388 | 0.0317 | 0.0483 | 0.0666 | 0.1501 |

| `std` | baseline mean score | headroom | score at p90 | score at a 0.10 m push peak |
| --- | --- | --- | --- | --- |
| 0.04 | 0.512 | 0.488 | 0.062 | 0.002 |
| **0.05** | **0.611** | **0.389** | **0.174** | **0.018** |
| 0.06 | 0.686 | 0.314 | 0.291 | 0.062 |
| 0.08 | 0.786 | 0.214 | 0.500 | 0.210 |

The 2026-08-17 criteria were a baseline mean score near 0.57 with real headroom
left, and a p90 disturbance collapsing the payment. The corrected error is a
little smaller than the old one, so `0.05` now scores 0.611 with 39% headroom
rather than 0.571 with 43% — inside the same band. Restoring 0.571 exactly would
take `std = 0.046` (the printed `q / sqrt(-log s)` line reads 0.0423 off the
median, which is the score-of-the-median, not the mean-of-the-score). That is
within the noise of the choice and would confound the objective change it is
meant to isolate, so it was not made.

**`0.08` is what the 10% standing-edge band would require, and it is too wide.**
It pays 0.50 at the nominal p90 and 0.21 at a push peak — the early-warning
property this term exists for, gone. The band was rejected on that trade; see the
standing-bias section.

The original measurement, kept because it sized the constant: 2026-08-17, zero
action, 16 envs x 2500 steps, 40000 grounded samples (fully grounded throughout),
on the **pre-correction** error:

| mean | median | p75 | p90 | p99 |
| --- | --- | --- | --- | --- |
| 0.0435 | 0.0331 | 0.0618 | 0.0864 | 0.1889 |

Scoring that distribution through `exp(-(e/std)^2)` — note this is the mean of the
score, not the score of the mean, which differ substantially here:

| `std` | baseline mean score | headroom | score at p90 |
| --- | --- | --- | --- |
| 0.03 | 0.399 | 0.601 | 0.000 |
| 0.04 | 0.492 | 0.508 | 0.009 |
| **0.05** | **0.571** | **0.429** | **0.051** |
| 0.06 | 0.637 | 0.363 | 0.126 |
| 0.08 | 0.737 | 0.263 | 0.312 |

`0.05` is chosen on the same two criteria that sized `ZMP_TRACKING_STD` (which
scores its baseline 0.68 with a p90 landing at 0.02): a push landing must collapse
the payment, and there must be real headroom left for the residual to earn. At
0.05 a p90 disturbance pays 0.051 — the same order as the ZMP term's 0.02 — while
leaving **43% of the term unclaimed**, more than the ZMP term's 32%. `0.06` scores
the baseline closer to the ZMP precedent but lets a p90 landing still pay 0.126,
which blunts the early-warning property this term exists for.

**Re-measure if:** `PUSH_VELOCITY` moves, or the baseline's operating error does.

`dcm_error` is the metric for this. Read it as `dcm_error / zmp_grounded`, for the
same reason `zmp_error` needs that denominator.

## angular_momentum

**Current:** weight `-0.005`, from `mdp.angular_momentum_l2`.

Centroidal angular momentum about the root, penalised as `-norm(L)^2`. The
stabilizer QP does not directly regulate it, so unlike the tracking terms this is
something the residual can improve without competing with the controller.

It reads the `root_angmom` subtree-angular-momentum sensor, which
`robots/additional_sensors_configuration.py` has been adding to every robot MJCF
and which nothing had ever read.

**The weight is measured, and the first guess was 10x too large.** A zero-residual
baseline (16 envs x 1500 steps) carries `norm(L)^2 = 6.64`, so the originally
proposed `-0.05` cost **-0.00664 per step — 58% of `dcm_stability`'s +0.01140.** It
would have cancelled most of the objective, and worse, it is a penalty the base
controller incurs for its *own* natural gait, which is the
[known wrong sign](#known-wrong-sign) failure in a new place. At `-0.005` it costs
0.00066 per step, ~6% of the objective: a regularizer rather than a competitor.

**Re-measure if:** the robot changes. `norm(L)` scales with mass and gait speed, so
the weight is not portable across HRP5P / JVRC1 / RHPS1.

## foot_slip

**Current:** weight `-1.0`, from `mdp.foot_slip`.

Squared tangential sole velocity, summed over feet, counted only while a sole
carries at least `min_normal_force`. Contact slip is a violation the QP's own
model cannot see, which is again what makes it worth paying for.

It reads the `left_foot_lin_vel` / `right_foot_lin_vel` velocimeters — the other
pair of sensors added by `additional_sensors_configuration` and never read.

**This term is deliberately near-inert, and the weight is set for that.** The
baseline barely slips: measured `sum(v_tangential^2) = 0.0004`, i.e. ~0.02 m/s. The
originally proposed `-0.1` made it -8e-7 per step — four orders of magnitude below
the objective, so it could never do anything. `-1.0` keeps the baseline cost
negligible (-8e-6) while making a *real* slip bite, because the penalty is
quadratic: a 0.3 m/s slide on one sole costs 0.0018 per step, ~16% of
`dcm_stability`. It is a guard against a failure that is not currently happening,
not a shaping term.

**Re-measure if:** ground friction changes, or the baseline's slip level does.

## recovery_dcm

**Current:** `1.0`, with `RECOVERY_TRACKING_STD = DCM_STD` and
`RECOVERY_WINDOW_S = 2.0`. The disturbance-gated half of the payment.

**2026-08-17 — the term is now `mdp.recovery_dcm`, not `recovery_tracking`.** The
gate machinery is unchanged (`_age_since_push`, the same window, the same
schedule-independence argument below); only the scored quantity moved from
plan-matching to the DCM offset above. Everything recorded here about *why the
gate exists and why it is 2.0 s* still applies.

**Why the gate exists:** measured with episode length divided out, the per-step
tracking rate is the same under a trained policy as under none (0.01183 vs
0.01196, SE 0.00015) — **the reward was blind to the policy**. Almost every step
of an episode is nominal walking, where mc_rtc tracks its own plan and a residual
has nothing to add; those steps average the informative ones away. This pays on
the window after a push only.

Gating on the push schedule is safe in a way that gating on *measured error*
would not be: the schedule is drawn by the event manager and is entirely
independent of the policy, so the agent cannot arrange to be paid more often. A
gate keyed on its own tracking error would be exactly that — an incentive to
enter the high-paying state.

**Re-measure if:** you rely on `RECOVERY_WINDOW_S = 2.0`. See below.

**History:**
- The window was measured, not guessed (`scripts/probe_residual_authority.py`,
  32 envs x 10 min, 529k grounded samples). Mean ZMP error against its settled
  level of 0.0412 m: 4.0x at the kick, 2.3x at 0.2 s, ~1.5x from 0.4 to 1.5 s,
  then decaying through 1.25x at 1.80 s to 1.06x by 2.4 s. Two seconds covers the
  elevated stretch; much longer and it re-admits the nominal steps this exists to
  exclude, which are ~90% of an episode at a 5-7 s push interval.
- **2026-08-14 — that profile was suspect.** It was taken before the probe learned
  to tell a real push from a warm-up-suppressed timer tick, and before it stopped
  binning the ~4 s post-reset posture settle as push recovery. Both pollutions
  landed in the early bins, the ones that sized this window.
- **2026-08-15 — re-measured after the fix, and 2.0 s holds.** 32 envs x 10 min,
  zero residual, ~8600 samples per 100 ms bin, settled level 0.0364 m:

  | since push | 0.0s | 0.2s | 0.5s | 1.0s | 1.3s | 1.8s | 2.0s | 2.3s |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | mean error | 0.237 | 0.083 | 0.054 | 0.055 | 0.069 | 0.047 | 0.043 | 0.042 |
  | vs settled | 6.5x | 2.3x | 1.5x | 1.5x | 1.9x | 1.3x | 1.2x | 1.15x |

  The kick is sharper than the polluted profile showed (6.5x, not 4.0x) and the
  error is still ~1.2x settled at 2.0 s, so the window is if anything slightly
  short rather than long. Note the **secondary bump at 1.2-1.4 s** (1.8-1.9x),
  absent from the old profile — most likely the first footstep after the push,
  and a reason not to shorten the window below ~1.5 s.
- **2026-08-19 — `2.0` holds on the corrected DCM error too**, which is what the
  term now scores. From `validate_dcm_objective.py`, 16 envs x 1200 steps, ~135-165
  samples per 100 ms bin, settled level 0.035 m:

  | since push | 0.0s | 0.2s | 0.5s | 0.9s | 1.4s | 2.0s | 2.1s | 2.5s |
  | --- | --- | --- | --- | --- | --- | --- | --- | --- |
  | mean error | 0.068 | 0.100 | 0.066 | 0.097 | 0.077 | 0.053 | 0.033 | 0.034 |

  The same secondary bump appears, later and larger (0.8-1.0 s, 2.8x settled), and
  the error drops to settled exactly at the 2.0 s boundary. The peak is at 0.2 s
  rather than at the kick because the corrected error measures the *disagreement
  with the command*, which the impulse takes a moment to produce.

## termination_penalty

**Current:** `-200.0`. Sized by *gradient* scale, not just by discounted horizon.
The manager multiplies every weight by `step_dt`, so a fall lands as one step of
`weight * 0.02` against dense steps of ~0.03. At -200 a fall is worth ~4, a couple
of seconds of dense reward, which is all the 2 s discounted horizon can see
anyway.

**History:**
- `-2000` — a single -40 sample among -0.03 ones, a 1000x outlier inside a
  minibatch that then gets advantage-normalized. Measured across every run to
  2026-07-31 the cost was total: `Loss/value` never converged (0.7-5.5) and the
  adaptive KL schedule pinned the learning rate to rsl_rl's 1e-5 floor from
  iteration 0 and never left it — 20899 iterations of no learning.
- 2026-08-03 — reduced to -200.

## collapsed

**Current:** `mdp.collapsed` — root height below `0.7 * nominal` **and** trunk tilt
within `FALL_LIMIT_ANGLE`, so it is exactly the upright crouch-collapse that
`fell_over` (tilt past 45 degrees) does not catch. Before 2026-08-19 it was the
height predicate alone.

**The union is unchanged, so the learned task is unchanged:**

```text
old = fell_over OR height_low
new = fell_over OR (height_low AND NOT fell_over) = fell_over OR height_low
```

Reset timing, value targets, `termination_penalty`, fall hazard and survival all
stay where they were. What changes is only that a toppled robot — usually both
tilted and low — now increments one counter instead of two.

**Why it mattered.** `TerminationManager` ORs the non-timeout predicates and
`is_terminated` reads that aggregate, so the overlap never doubled the `-200`.
But it made the per-cause shares non-additive: `fell_over` + `collapsed` could
exceed the fall rate, and the question the two counters exist to answer — is this
policy trading a topple for a crouch — could not be read off them.

**Historical numbers in `RUN_COMPARISONS.md` are the old, overlapping labels.**
Their absolute shares are not comparable with post-2026-08-19 ones without
re-evaluating. Fall hazard is computed from the union of the balance-failure
predicates, never by summing the categories, and stays comparable either way.

## zmp_error and zmp_grounded

**Current:** two metrics, read as a ratio.

Every `Episode_Reward/*` curve is an episode *sum*, and those turned out to
correlate with episode length at **r = +0.98**: they move when the robot survives
longer, not when it tracks better, so they cannot answer "is the control
improving". `zmp_error` is in metres and length-independent, so it can.

But `MetricsManager` averages as `sum / step_count` over *every* step, and a step
with the feet unloaded contributes 0 m — there is no centre of pressure to place,
and dividing by a vanishing normal force would put the ZMP anywhere. So
`zmp_error` alone *falls* when the robot spends more time off the ground, which
is backwards for a tracking error. `MetricsTermCfg.reduce` offers no masked mean,
so the denominator is published separately.

**Read the tracking quality as `zmp_error / zmp_grounded`.** `zmp_grounded` is
also a health curve in its own right: a robot lifting off, stumbling or on its
way down spends more steps ungrounded, and that shows up there before it shows up
in a termination.

## zmp_tracking

How the two sides of the comparison are built.

The **controller side** is `planned_zmp`: the centroidal ZMP of the QP's own
solution, i.e. the ZMP the motion mc_rtc commands this period implies (see
`mc_rtc_controller_host._planned_zmp`).

The **sim side** is the centre of pressure of the foot wrenches, summed and moved
onto the ground plane exactly as `mc_rbdyn::zmp` does —
`zmp = p + n x moment_p / (n.f)`, which for a flat floor is `(-M_y, M_x) / F_z`.
MuJoCo reports the wrench transmitted *at* the sensor site; the reaction the
ground applies to the robot is its negation, the same `fs *= -1` the I/O binding
applies before handing these to mc_rtc. `subtree_com` of the root body is the
whole robot's CoM, and it is MuJoCo's own, so it needs no sensor and carries no
estimation error.

Both are taken **relative to their own CoM**, and that is not cosmetic: the
controller places its plan against the state its observers estimate
(KinematicInertial legged odometry), which drifts from MuJoCo's ground truth over
an episode. Differencing absolute positions would charge the residual for that
drift; the CoM-to-ZMP offset is drift-free and is anyway the quantity the LIPM
relation the plan comes from is written on.

Payment is zero while the feet carry less than `min_normal_force` (20 N): a robot
in the air has no centre of pressure to place, and dividing by its vanishing
normal force would put the ZMP anywhere.

`com_velocity_tracking` needs no such drift correction: KinematicInertial
integrates *position* from the anchor frame, so position is what drifts, while
velocity is differential and directly comparable. It reads `subtree_linvel`, which
MuJoCo fills because the robots carry a subtree sensor (the RL-only `root_angmom`),
which is what makes `mj_subtreeVel` run.

## _ZmpSensors

Split out of `zmp_tracking` so the reward is not the only way to reach the raw
number: the reward reports `exp(-(error/std)^2)`, a bounded kernel output that
says nothing about metres.

`offset_error` is memoised for the current step, because three terms score the
same quantity — `zmp_tracking`, `recovery_tracking` and the `zmp_error` metric —
and computing it is not free: the `site_xmat`/`site_xpos` gathers, a batched
`(num_envs, k, 3, 3) @ (num_envs, k, 3, 1)`, the cross products and the
action-term lookup, at 50 Hz x num_envs.

`common_step_counter` is a sound cache key because terminations, rewards and
metrics all run in one phase of `ManagerBasedRlEnv.step` off the same `sim.data`.
The tolerances are in the key too, so a term configured with different ones gets
its own value rather than someone else's. One shared instance per env exists
because three separate ones would mean three separate memos, defeating the point.

## steps_since_push

Bounded by `episode_length_buf` on purpose. Class-based event terms get no `reset`
callback from mjlab's `EventManager` — it only re-samples the interval timers — so
`last_push_step` survives an episode boundary. Rather than reach for a reset hook
that does not exist, a push is simply not counted unless it happened after this
episode started. Envs with no push yet read a huge age (`NEVER_AGE`).

Reading `last_push_step` rather than watching the interval countdown is the only
way to get this right; see
[evaluation.md](evaluation.md#binning-by-time-since-a-push).

Interval events fire *after* the reward is computed (see
`manager_based_rl_env.step`), so the first step `recovery_tracking` can pay on has
an age of 1, never 0.

## Residual harm at gamma=0.99

> **SUPERSEDED 2026-08-19. The headline figures in this section do not reproduce.**
> They come from a comparison at **6 envs per arm**, and `encoder_bias` is a per-env
> `startup` draw, so episodes within an env are not independent and the effective
> sample size tracks the env count rather than the episode count. Re-measured at
> **28 envs per arm** (n = 168), the same checkpoint reads:
>
> | | 6 envs/arm (below) | 28 envs/arm |
> | --- | --- | --- |
> | tracking aggregate | **-10.1%** (p = 2e-06) | **-0.9%** (p = 0.37) |
> | `zmp_tracking` | -10% | -1.4% (p = 0.15) |
> | `com_velocity_tracking` | -8% | -1.5% (p = 0.01) |
> | `recovery_tracking` | -17% | +2.0% (p = 0.63) |
> | `fell_over` | +17.8pp | +7.7pp (p = 0.06) |
> | hazard ratio | — | 0.81 |
>
> **The claim this section is named for — that the residual is measurably worse than
> no residual at gamma = 0.99 — is not supported at proper power.** The direction is
> unchanged and the topple increase survives as a trend, but the magnitude was an
> artefact of six environments. Resampling puts the sign-error rate at 6 envs/arm at
> **15.3%**; the floor is 16 envs/arm. Full record and method in the untracked
> `RUN_COMPARISONS.md`.
>
> The reasoning downstream of this section — the credit-assignment hypothesis that
> produced `gamma = 0.997` — was independently supported by the correlation statistic
> and by the later runs, so it is not withdrawn. Only these numbers are.

**Original text, kept for the record.** From
`model_3050` of the 2026-08-14 run, against the zero-residual baseline in the
same paired run (n=48/54, 24 envs x 20 min), with episode length divided out so
duration is not a confound:

| per-step rate | baseline | policy | delta | p |
| --- | --- | --- | --- | --- |
| `zmp_tracking` | 0.00687 | 0.00619 | -10% | <0.001 |
| `com_velocity_tracking` | 0.00799 | 0.00735 | -8% | <0.001 |
| `recovery_tracking` | 0.00282 | 0.00235 | -17% | 0.020 |

| failure mode | baseline | policy | delta | p |
| --- | --- | --- | --- | --- |
| `fell_over` | 6.2% | 24.1% | +17.8pp | 0.013 |
| `collapsed` | 77.1% | 66.7% | -10.4pp | 0.244 |
| survival | 20.8% | 24.1% | +3.2pp | 0.696 |

So it converts crouch-collapses into topples at unchanged survival, and tracks
~10% worse per step on the objective it is optimising. The episode-sum rewards
the script prints do not show this cleanly, because they track episode length;
the per-step rates are what resolve it (see
[evaluation.md](evaluation.md#per-step-rates-beat-survival)).

**It got there gradually.** Over training the policy intervened steadily *more* —
`residual_magnitude` -0.0305 -> -0.0450 between iterations 500 and 3100, +48% —
while `fell_over` climbed 0.19 -> 0.23 and reward and episode length peaked
around iteration 1400 and drifted down.

**Within its own episodes, intervening more goes with dying sooner.**
`corr(per-step |residual|, episode length) = -0.60` over 68 policy episodes, and
episodes reaching the cap carry a smaller residual than those that do not
(0.00149 vs 0.00168, p < 0.001).

**Caveat, and it is not small:** that correlation is not proof of causation. A
robot already in trouble produces extreme observations and therefore larger
actions, so the arrow could point the other way. It is consistent with the
residual causing falls, not demonstrative of it.

**The hypothesis this suggests** is a credit-assignment one, and it is recorded
under `gamma` in [ppo.md](ppo.md): at `gamma = 0.99` and `step_dt = 0.02` the
horizon is 100 steps, **2 s**, against episodes of ~50 s. A residual can improve
the ZMP match now and topple the robot five seconds later without the discounting
ever presenting the bill. The signature fits — `zmp_error` improved monotonically
while `fell_over` rose and survival did not.

**Tested 2026-08-15, and the hypothesis held on the failure mode.** At
`gamma = 0.997` the topple signature is gone: `fell_over` is **19.6% for both
arms** (against 6.2% baseline / 24.1% policy at γ.99), `collapsed` fell 70.5% ->
57.1%, and survival rose 18.8% -> 29.5% (p = 0.061). The policy now converts
crouch-collapses into *survivals* rather than into topples. n = 112/arm at
`model_1150`.

**The tracking deficit did not go away** — it was measured at
`residual_scale = 0.03` in the same run and widened to -16.7%, which is the
scale's doing rather than the horizon's
([residual-authority.md](residual-authority.md#residual_scale)). Whether γ.997 at
scale 0.01 closes it is the open question; that run is what tests it.

## Rejected shaping ideas

Recorded so they are not reinvented.

**Paying for the controller still generating a gait** (`controller_reference_motion`,
tanh of the joint-velocity reference) is anti-correlated with what we want: over
96 s x 16 envs the reference norm runs 1.78 rad/s in the second before a fall
against 0.64 overall, because a falling robot's controller thrashes. The term
would pay *more* for the run-up to a fall. It exists in `mdp.py` but is not wired
into the task, and should not be without re-measuring.

**A support-region margin on the measured ZMP** is close to a tautology, since a
centre of pressure lies inside the contact hull by construction. Only the
*commanded* ZMP can leave it.

**A commanded-vs-measured joint position term** does not measure "is the
controller's plan being executed" under this coupling: the joints are
position-controlled with stiff PD, so the measured angle follows the commanded one
to within 0.04 rad even while the robot topples (measured), and since the command
is reference-plus-residual such a term reduces to a second penalty on the
residual. Whole-body failure shows up in the base — attitude, height, travel —
which is what the task's terminations and `base_progress_tanh` read instead.

## action_l2

**Current:** weight `-0.1`, on the raw action **clamped to `RAW_CLIP = 1.0`**.

The clamp is not cosmetic — it fixes a runaway that destroyed a run. The residual
is hard-clipped, and because the env cfg sets `clip` equal to `scale`, the clip
binds at a raw action of exactly 1.0. Past that a larger raw action has **no
physical effect whatsoever**, so an unclamped quadratic penalty keeps charging more
for a difference the robot cannot feel.

**How it failed.** `Run 2026-08-17_15-38-02_zeroinit-4ev` was healthy for 2940
iterations and then diverged inside ten:

| iteration | `Train/mean_reward` | `Episode_Reward/residual_magnitude` | `Loss/value` | `Episode_Metrics/zmp_error` |
| --- | --- | --- | --- | --- |
| 2940 | 28.2 | -0.066 | 0.047 | 0.0644 |
| 2950 | 24.1 | -0.412 | 15.9 | 0.0634 |
| 2970 | -90.0 | -3.91 | 23.7 | 0.0703 |
| 2999 | -290.8 | -0.249 | 32.7 | 0.0660 |

The policy's mean wandered out to a raw action of roughly **16** — sixteen times
the point where the clip binds. `zmp_error` never moved, because the *physical*
residual was bounded the whole time. What diverged was the penalty: it reached
-25 per step, which handed the critic targets two orders of magnitude outside its
range (`Loss/value` 0.047 -> 32.7), and the adaptive schedule could not pull the
rate down fast enough at 4 events per iteration
([ppo.md](ppo.md#num_learning_epochs)).

**Why the clamp is safe.** It is inert in the healthy regime. At `model_2900` the
per-step penalty was 0.00137, i.e. `action_l2` ~ 0.685 over 12 joints, so the mean
per-joint action was ~0.24 — well inside 1.0. The clamp changes the reward *only*
where the policy has already left the region its actions can affect, which is
exactly the pathology.

`residual_rate` had the same exposure until commit `8b80696`; the task now uses
`mdp.action_rate_l2`, which clamps both the current and previous raw action at the
same `RAW_CLIP` before differencing them.

## torque_margin

**Current:** weight `-0.05` **provisional**, ramped x4 at iteration 500 and x10 at
1000 by the `torque_margin_weight` curriculum. `TORQUE_SOFT_RATIO = 1.0`.

**Why it exists: the measurement was already in the repo and nothing acted on it.**
[residual-authority.md](residual-authority.md#residual_scale) records that
`residual_scale = 0.01` through the real `PDgains_sim.dat` gains is **22-27% of
every leg joint's hardware limit**, and that a saturated residual takes ankle pitch
to **0.64** of its limit. Nothing enforced any of it. The position clip bounds the
*offset*, not the torque it produces, and `EFFORT_LIMIT` is deliberately `inf`
(mc_mujoco parity, see `pd_actuator_configuration`) so MuJoCo will not clamp
either. A residual that outgrows the actuators is invisible in sim and divergent
on the robot.

Both halves already existed unwired: `mc_rtc_robot_configuration.get_effort_limits`
reads per-joint limits straight from the mc_rtc `RobotModule`, and
`entity.data.qfrc_actuator` gives the realised joint torque.

**Shape**, after leo_mjlab's `raw_torque_peak_penalty`:

```
cost = sum_j log1p(relu(peak|tau_j| / limit_j - SOFT_RATIO))
```

- **Peak over the decimation window, not the mean.** What sizes an actuator is the
  worst instant, and at 20 substeps per policy step that instant is invisible to
  anything sampled at the policy rate. The action term peak-holds it; note
  `apply_action` runs *before* `sim.step`, so the accumulator trails by one substep
  and the reward folds in its own read to cover the last of the window.
- **`SOFT_RATIO = 1.0`, the limit itself.** leo_mjlab ran 0.7 and moved to 1.0
  deliberately — charging below the limit made the policy timid.
- **`log1p`, not square.** A soft knee, so one saturated joint cannot swamp the
  objective the way `angular_momentum` did at its first weight.
- **Residual joints only.** The documented hazard is the leg joints, which are
  exactly the twelve the residual drives; charging the other 41 adds an offset the
  policy can barely influence.

**Measured 2026-08-18, and the sizing rule is not the one used for the shaping
terms.** This is a *guard*, like `foot_slip`: the right baseline cost is **zero**,
not the ~5% used for `angular_momentum`. At `-0.2` with the warm-up gate the
zero-residual baseline pays exactly **0.000000** per step.

Over 6 envs x 700 steps, the max-over-joints ratio to the hardware limit:

| | median | p90 | p99 | max | steps over limit |
| --- | --- | --- | --- | --- | --- |
| settled | 0.133 | 0.203 | 0.229 | **0.40** | **0.00%** |
| first 0.5 s | 0.128 | 0.177 | 1.80 | **22.1** | 1.03% |

The worst settled joint is `LAP` at 0.40, which matches
[residual-authority.md](residual-authority.md#residual_scale)'s independent finding
that ankle pitch is the binding joint. So the baseline has ~2.5x of headroom and
the term never fires on it — it exists for the regime that produced the 2026-07-31
failure at 220-270% of limit.

**`warmup_steps = 25` is load-bearing, not hygiene.** The reset teleport drives a
*substep* transient of **22x the limit** — `kp` reaches 36000 on the knees, so a
1 rad settling error is ~36000 Nm. No policy-rate sample ever sees it (sampled at
the policy rate the same run maxes at 0.40), but the peak-hold does, and it is not
the residual's doing. Without the gate the baseline paid 0.27% of the objective
in pure reset artefact. The action term also zeroes its accumulator on reset, which
is necessary but not sufficient: the first policy step still spans the transient.

**Re-measure if:** the robot changes, `residual_scale` grows, or the PD gains move.
The limits come from the mc_rtc `RobotModule` so they follow the robot, but the
*ratio* depends on the gains that produce the torque.

## torque_margin_weight

**Current:** mjlab's `reward_curriculum`, with weights `-0.05`, `-0.20`, and
`-0.50` at common policy-step counts 0, 48000, and 96000 respectively.

A safety penalty at full strength from iteration 0 charges mc_rtc's own torques
before the residual has done anything — the residual starts at exactly zero under
`ZeroInitMLPModel`, so every newton-metre early in training belongs to the base
controller. leo_mjlab ramps its equivalent from -0.05 to -1.00 over 6000 iterations
for the same reason, stated as keeping the policy from becoming "too timid".

`common_step_counter` increments once per vectorised policy step, **not once per
environment**. Multiplying the thresholds by `num_envs` delayed the old ramp by
128x, so it never fired in practical runs. The thresholds are now invariant to
both environment count and rollout length. With the current 256-step rollout they
land near iterations 188 and 375; with the historical 96-step rollout they land
at 500 and 1000.

The curriculum manager calls from `_reset_idx`, but the reward configuration is
global. The first reset after a threshold changes the scalar for every env; this
is not a staggered per-env curriculum.

## com_velocity_error

**Current:** a metric, in m/s, replacing the `com_velocity_tracking` reward.

The norm of `measured CoM velocity - control_com_vel`. Reported raw rather than
through an exponential kernel: as a diagnostic the metres-per-second figure is the
useful quantity, and the two-scale split that `COM_VELOCITY_TRACKING_STD` needed
only existed to keep a *reward* from hiding crouch-collapse.

**Why it is worth logging at all.** This is the only quantity that came out negative
against the zero-residual baseline in **every** comparison ever run on this task —
across three rewards, two horizons, three residual scales, and with and without the
coherence gate. That makes it the best available detector of a policy fighting the
plan, which is exactly what stops being visible once the agreement rewards are gone.

## Pruning the agreement rewards

**Removed 2026-08-19:** `zmp_tracking` and `com_velocity_tracking` are no longer
reward terms. Both scored `measured - planned` against mc_rtc's own plan, which the
stabilizer QP already optimises. Both are retained as metrics (`zmp_error`,
`com_velocity_error`), so nothing stops being observable.

**Two independent lines of evidence picked out the same two terms.**

Measured share of the total absolute dense reward on the zero-residual baseline, and
the policy's own delta against that baseline:

| term | share | policy delta | verdict |
| --- | --- | --- | --- |
| `dcm_stability` | **61.4%** | +4.2% (p = 0.007) | working |
| `termination_penalty` | 12.8% | -24.5% | working |
| `recovery_dcm` | 11.0% | +9.3% (p = 0.03) | working |
| `angular_momentum` | 5.9% | -17.2% (p = 5e-04) | working |
| `com_velocity_tracking` | 4.5% | **-7.2 / -5.6 / -6.7 / -3.6%** | negative everywhere |
| `zmp_tracking` | 3.8% | +1.3 / -1.4 / -0.2 / +0.3% | null |
| `upright` | 0.5% | -16.0% (p = 0.04) | working |
| `foot_slip` | 0.1% | noise at 1e-5 | inert guard |
| `torque_margin` | 0.0% | exactly 0 | inert guard |

The terms that are zero-sum-or-worse **are** the agreement terms, and together they
are only **8.3% of the dense signal**.

**Why removing them is low-risk rather than a gamble.** `com_velocity_tracking` is
negative against baseline in every comparison across both reward eras, often at
p < 1e-07 — the policy actively sacrifices a term it is paid for. A prior that can be
given up that freely is not binding, so removing it should change behaviour little.
That is the argument *for* deletion: it was paying for nothing.

**What is deliberately kept.** The *raw* controller signals stay as observations —
`controller_ref_pos`, `controller_ref_vel`, `controller_planned_zmp`,
`controller_planned_com_vel`, `controller_pos_error`. Feeding the policy the plan is
useful; *paying* it to match the plan is not. `recovery_dcm` also stays: it is gated
on the post-push window but scores the DCM-to-CoP distance, which is
plan-independent, so it is not an agreement term despite the gating.

`foot_slip` and `torque_margin` stay at 0.1% and 0.0%. They are guards, not shaping:
they cost nothing and fire only on a real contact-slip or hardware-limit excursion.
Removing them would save nothing and lose the detection.

**This does not fix the wrong sign.** The agreement pair carried a +33.6% standing
bias; `dcm_stability` carries **+56.5%** and is 61.4% of the signal. The prune
removes 8.3% of wrong-signed reward and leaves the larger share untouched — see
`dcm_stability standing bias` below, and the velocity-referenced fix proposed there.

## dcm_stability standing bias

**Measured 2026-08-18, and it is real: `dcm_stability` scores a standing robot
+56.5% above a walking one.** Four times the old ZMP term's edge, in the term that
now carries ~95% of the dense signal.

4 envs x 700 steps per regime, zero residual, identical but for the controller
(`Enabled: Posture` against `LogisticController_ismpc`, passed through
`_make_env_cfg(mc_rtc_yaml=...)` so no config file was touched):

| regime | `dcm_stability` | `zmp_tracking` | `com_velocity_tracking` | `norm(alpha)` |
| --- | --- | --- | --- | --- |
| walking | 0.5702 | 0.6665 | 0.8059 | 0.728 |
| standing | 0.8923 | 0.8901 | 0.9327 | 0.000 |
| **edge** | **+56.5%** | +33.6% | +15.7% | |

**Why, and it is structural rather than incidental.** In the LIPM,
`d(xi)/dt = omega * (xi - CoP)`: a robot walking forward *requires* the DCM to lead
the CoP, and that offset is the quantity producing the motion. It is not an error.
Standing has `com_vel ~ 0`, so `xi ~ com` and the scored distance collapses to a
balanced stance's CoM-to-CoP offset. The term therefore pays most where the robot
does least.

**The old terms had the same defect an order of magnitude smaller.** `zmp_tracking`
was demoted 0.5 -> 0.05 partly for a documented +14% edge; the direct measurement
here puts it at +33.6%, so that figure was itself understated. But at weight 0.05
it contributes ~6% of the dense signal, while `dcm_stability` + `recovery_dcm` carry
~2.0 of ~2.1. **The coefficient moved the wrong way when the objective changed.**

**It remains latent, for the reason the old note gives.** `residual_scale = 0.01`
cannot cancel a swing trajectory (~0.5 rad), and the residual cannot change what
mc_rtc commands — the gait is the controller's, not the policy's. The reachable
consequence is not "the robot stands" but "the policy resists the gait as hard as
its authority allows".

**Which the coherence gate now partly absorbs.** Resisting the gait *is* opposing
`alpha`, and that is exactly what `GATE_STRENGTH` withholds authority from — the
gate was measured attenuating 59% of steps, rising to 65% when the gait is fastest.
The two changes were made for unrelated reasons and the second happens to blunt the
first.

**The principled fix, when the objective is next revisited:** score the DCM offset
against the offset the *commanded* velocity implies rather than against zero. In
steady LIPM walking `xi - CoP ~ v / omega`, so penalising
`abs(norm(xi - CoP) - norm(v_cmd) / omega)` targets zero offset when standing and
`v/omega` when walking, and the bias disappears. Not done here: it is a change to
the objective, and three of them are already stacked in the run testing the gate.

**Done 2026-08-19, as a vector difference rather than the scalar one above** (the
vector form also penalises an offset pointing the wrong way). Two things the
re-measurement settled, and they point in opposite directions:

1. **The gradient the residual can move is fixed.** Lagging the commanded speed by
   >0.05 m/s went from costing 1.56x the on-command error to costing 2.91x. The
   term no longer sits near-flat while a policy drags its own gait — the table is
   under `dcm_stability`.
2. **The standing edge did not disappear: +45.9% -> +28.4% at `DCM_STD = 0.05`.**
   The proposal's acceptance band was 10% and this misses it.

**Why the remaining edge is not the same defect.** What is left is not a velocity
preference — the command is subtracted — but a difficulty difference between two
regimes: a standing robot's CoP sits almost exactly under its DCM (median error
**0.0031 m**), while a walking one carries an irreducible ~0.032 m tracking error
that agrees with the operating error in
[residual-authority.md](residual-authority.md). A residual under a *walking*
controller cannot collect the standing score by any action; it can only move along
the within-episode gradient, which now points at the command.

**And closing the last of it costs more than it is worth.** The band is met only
at `std >= 0.08`, where a p90 disturbance still pays 0.50 and a push peak 0.21.
That trades the term's early-warning property for a cross-controller comparison no
policy can act on, so `0.05` stays. Re-open this if a *commanded-velocity
curriculum* is ever added, where standing and walking become states of one
episode.

## Known wrong sign

Both tracking terms score a standing robot slightly above a walking one (0.75 vs
0.66 for the ZMP term), which is the wrong sign for this task. It stays
theoretical only because the residual is hard-clipped to an authority that cannot
cancel a swing trajectory — revisit it if `residual_scale` grows. See
[residual-authority.md](residual-authority.md).

**Largely defused 2026-08-17** by the demotion to 0.05: those two terms were ~90%
of the dense signal, so the wrong sign was on almost every nominal step. At a
tenth of the weight the mis-ranking is a tenth as strong.

**But it moved rather than went away, and it got worse.** The check deferred here
was run on 2026-08-18 and `dcm_stability` carries the same defect at **+56.5%**,
against this term's directly measured +33.6%, while carrying ~95% of the dense
signal instead of ~6%. See `dcm_stability standing bias` above. The 0.75/0.66
figure quoted in this section also understates the ZMP term's own edge.
