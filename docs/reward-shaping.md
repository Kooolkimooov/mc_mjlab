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

**Current:** `0.05` — sized off the zero-residual baseline rather than picked.
Measured over 64 s x 16 envs, the CoM-to-ZMP offset error runs median 2.1 cm,
p75 4.4 cm, p90 9.0 cm. `std` is where the exponential has fallen to 1/e, so
5 cm scores that baseline 0.68 on average: ordinary walking is well paid, a push
landing (p90) drops payment to 0.02, and there is a third of the term left for
the residual to earn.

**Re-measure if:** the baseline's operating error moves. Tighten `std` and the
signal is mostly noise (0.02 scores 0.41); loosen it and it saturates (0.10
scores 0.84).

## ZMP_TRACKING_WEIGHT

**Current:** `0.05`, a prior rather than the objective — see `dcm_stability`.

**History:**
- `1.0` originally.
- Halved when `recovery_tracking` was added. The two are the same quantity and
  the same kernel; splitting the weight keeps the total tracking payment roughly
  where it was while moving half of it onto the steps that follow a push, where
  the residual can actually change the outcome. Nominal steps therefore still pay
  (~0.0125 per step against ~0.001 of penalties), which is what stops the split
  from creating an incentive to end the episode early.

## COM_VELOCITY_TRACKING_STD

**Current:** `0.05` horizontal, `COM_VELOCITY_TRACKING_STD_VERTICAL = 0.005`.
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

**Current:** `0.05`, matching the ZMP term's demotion.

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

**What it scores.** With the divergent component of motion
`xi = com_xy + com_vel_xy / omega` and `omega = sqrt(g / com_z)`, the linear
inverted pendulum gives `d(xi)/dt = omega * (xi - CoP)`. So `norm(xi - CoP)` is
not a proxy for instability — it **is** the divergence rate, in metres. The reward
is `exp(-(error/std)^2)`, gated on the feet carrying load exactly as
`zmp_tracking` is.

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
read it per step), rather than `offset_error` as before.

## DCM_STD

**Current:** `0.05` — measured off the zero-residual baseline, not copied from
`ZMP_TRACKING_STD` despite landing on the same number.

Measured 2026-08-17, zero action, 16 envs x 2500 steps, 40000 grounded samples
(fully grounded throughout). The DCM-to-CoP distance runs:

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

## RECOVERY_TRACKING_WEIGHT

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

**A trained residual measured worse than no residual at all.** From
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

Note `residual_rate` (mjlab's `action_rate_l2`) is still on the unclamped raw
action and has the same shape of exposure, though its weight is 10x smaller and it
did not drive this failure.

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

**Current:** the task's first and only curriculum term, from `mdp.reward_weight`
(ported from leo_mjlab's `velocity/mdp/curriculums.py`).

A safety penalty at full strength from iteration 0 charges mc_rtc's own torques
before the residual has done anything — the residual starts at exactly zero under
`ZeroInitMLPModel`, so every newton-metre early in training belongs to the base
controller. leo_mjlab ramps its equivalent from -0.05 to -1.00 over 6000 iterations
for the same reason, stated as keeping the policy from becoming "too timid".

**It fires from `_reset_idx`**, so envs cross a stage boundary as they reset rather
than together. That is fine for a stage ramp and is why the boundaries are stated
in environment steps (`iterations * num_steps_per_env * num_envs`), not iterations.

## Known wrong sign

Both tracking terms score a standing robot slightly above a walking one (0.75 vs
0.66 for the ZMP term), which is the wrong sign for this task. It stays
theoretical only because the residual is hard-clipped to an authority that cannot
cancel a swing trajectory — revisit it if `residual_scale` grows. See
[residual-authority.md](residual-authority.md).

**Largely defused 2026-08-17** by the demotion to 0.05: those two terms were ~90%
of the dense signal, so the wrong sign was on almost every nominal step. At a
tenth of the weight the mis-ranking is a tenth as strong. It is not *gone* —
`dcm_stability` has the same shape of risk and has not been checked for it, which
is worth doing once `DCM_STD` is measured.
