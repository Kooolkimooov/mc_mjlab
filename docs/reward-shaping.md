# Reward shaping

The reward must pay for tracking the controller's plan, not for surviving. With
`gamma=0.99` the discounted horizon is ~100 steps (2 s), so the termination
penalty shapes only the last couple of seconds before a fall (`0.99^400 ~ 0.02`)
and the dense terms do all the work.

Those are `zmp_tracking` and `com_velocity_tracking`: the CoM-to-ZMP offset and
the CoM velocity, the two halves of the planar LIPM state the plan is written on.
There is deliberately no separate DCM term — the divergent mode
`com + comVel/omega` is a linear combination of those two, so any DCM weighting
is already reachable by choosing their weights.

Both were sized against the zero-residual baseline, and both collapse in the
run-up to a fall (to 0.12 and 0.10 of a possible 1.0), which is what makes them
the early warning the sparse penalty cannot be.

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

**Current:** `0.5`. Weights are per second — the manager scales by `step_dt` — so
at 0.5 over the 90 s walk window a perfectly tracking episode earns 45 and the
baseline ~31, against the -4 a fall costs. Dense shaping that ranks good balance
without ever out-paying survival.

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

**Current:** `0.5`.

**Re-measure if:** you are wondering whether this should be back at 1.0. It
probably should.

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

## RECOVERY_TRACKING_WEIGHT

**Current:** `1.0`, with `RECOVERY_TRACKING_STD = ZMP_TRACKING_STD` and
`RECOVERY_WINDOW_S = 2.0`. The disturbance-gated half of the ZMP payment.

**Why it exists:** measured with episode length divided out, the per-step
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

## Known wrong sign

Both tracking terms score a standing robot slightly above a walking one (0.75 vs
0.66 for the ZMP term), which is the wrong sign for this task. It stays
theoretical only because the residual is hard-clipped to an authority that cannot
cancel a swing trajectory — revisit it if `residual_scale` grows. See
[residual-authority.md](residual-authority.md).
