# Hard constraints and the decision journal

Constraints are numbered, never renumbered, and never deleted — one that stops
being true is marked `RETIRED` with the measurement that retired it. Every
configuration change cites the constraints it touches, and **a rejected path is
not retried until the journal below says what changed since.**

Each constraint states the rule, the measurement that made it a rule, and what
would invalidate it. The numbers are the identifiers: `grep -rn C4 docs/` finds
one, and a code comment citing its anchor is dead-link-checked by
`scripts/check_prose.py` on every edit.

Grouped by what they constrain: C1-C3 the shape of an objective, C4-C5 what may
be resumed and what may be believed, C6-C8 what a measurement is allowed to say.

---

## C1 no cost on an event the policy can stop producing may rise

**Current:** a penalty attached to an event the policy controls the occurrence of
may not be increased — not by weight, not by ceiling, not by threshold. The
demand is always satisfied by ceasing to produce the event. To *increase* a
quantity, pay for the wanted state per second instead of fining its absence at an
event.

**Why:** five independent attempts in `leo_mjlab` to make landings cheaper — via
contact balance, capture-point placement, free velocity, flat touchdown, and
swing height at triple weight — each produced the same answer: air time
0.487 s -> 1.512 s, falls 0.008 -> 0.830, and the landing cost not even improved
(impact 0.0533 -> 0.0685). A gait clock paying 3.16/s for kept rhythm did not
make the escape unprofitable; the policy paid the clock instead. What worked was
`swing_height_bonus_dense`, paying height *per second of flight*: clearance
0.0058 -> 0.0217 m where three penalties against unreachable targets had moved it
nowhere. See [leo-mjlab-review.md](leo-mjlab-review.md).

**Locally:** `foot_slip` (`-1.0`) is the one term of this shape; `dcm_stability`
and `recovery_dcm` are bonuses, which unloading forfeits rather than escapes. At
today's weights the escape saves at most 1.0/s and gives up 4.0/s, so it is
unprofitable. Measured 2026-09-14, the escape does not occur at all: grounded
fraction is 100.0% over n=16,000 under a zero residual. So the constraint is not
live here today, and
[`GROUNDED_FRACTION_MARGIN`](evaluation.md#grounded_fraction_margin) stands as
the tripwire that would catch it if authority grows.

**Re-measure if:** any contact-gated weight changes, or the residual's authority
grows.

## C2 a target is placed against the distribution its own gate admits

**Current:** a kernel target sits 1.3-1.5x above the median of the samples its
gate admits; a guard threshold sits far above the p99 and passes by reading zero.
Below the measurement a clamped ratio pins at 1.0 and the gradient is exactly
zero; far above it the kernel floors and the gradient vanishes again.

**Why:** two `leo_mjlab` rejections were the same arithmetic — a target lowered
onto the measurement cut traction instead of concentrating it, and a target 2.2x
above everything the gait did was a constant.

**The local case shows how easily the wrong distribution is picked.**
`DCM_STD = 0.10` reads 3.36x against nominal walking, but that is
`dcm_stability`'s distribution and its weight is `0.0`, so it decides nothing. It
was then compared against the *endpoints* of `recovery_dcm`'s time-binned mean
error (0.033-0.100 m), which made it look like 1.0-1.9x and in band. Both
readings are wrong. The rule asks for the **median of the samples the gate
admits**, measured directly: 3,200 admitted samples of 16,000, median error
**0.0411 m**, so the constant is **2.43x** — out of band, on the floored side.
Full derivation at [reward-shaping.md#target-placement](reward-shaping.md#target-placement)
and the measurement at [reward-shaping.md#recovery_dcm](reward-shaping.md#recovery_dcm).

**Re-measure if:** a kernel's `std`, a guard's threshold, or the gate in front of
either changes.

## C3 in an additive mix a saturated half is a constant per-event bonus

**Current:** when two normalized halves are summed, one of them pinned at its
clamp does not transfer its weight to the other — it becomes a constant paid per
event, which rewards producing more events.

**Why:** `0.5 * (d_ratio**2 + p_ratio**2) / step_dt` with the distance half
saturated shortened the step period 0.204 -> 0.164 s, the opposite of the demand,
because the payment rate `0.5/T + 0.5*T/T_target**2` is minimal at the target and
rises as `T` shortens. The docstring asserting that saturating one half moves all
the weight onto the other was simply wrong.

**Re-measure if:** a reward term is built by summing normalized components.

## C4 the only valid resume point is a checkpoint whose sweep exists

**Current:** a full resume requires a qualification verdict written beside the
checkpoint. Never take a checkpoint without checking the state of its run at that
iteration; never resume from one no deterministic sweep has judged.

**Why:** four consecutive `leo_mjlab` diagnoses blamed a reward change for what
was a poisoned starting point, and three checkpoints of that campaign — taken
150 iterations after a resume, or just as a run turned brittle — contaminated
everything restarted from them. One of them cost constraint C1 a false sixth data
point: two different configurations reproduced the same collapse trajectory to
three decimals, so the cause was neither of them.

**Locally:** enforced by `qualification_sidecar.enforce`, called from
`ResidualBalanceOnPolicyRunner.load`. Actor-only loads warn instead of refusing,
because the qualifier itself must be able to load an unjudged checkpoint.
`MC_MJLAB_ALLOW_UNQUALIFIED_RESUME=1` downgrades the refusal.

**Re-measure if:** the qualifier's gates change enough that an old verdict means
something different.

## C5 training-scale metrics are not deployment-scale metrics

**Current:** only falls are unambiguous enough at training scale to justify
stopping a run; every other guard is relative drift from a post-attach baseline.
A policy is judged good by the deterministic sweep, never by the training curves.

**Why:** exploration noise inflated `leo_mjlab`'s leg-torque clipping by roughly
10x — 0.32 in the training loop against 0.07 in the sweep for the same policy —
and three healthy runs were nearly killed by caps calibrated on training numbers
as though they were acceptance thresholds. Locally this is already the design of
`training_watchdog.py`; it is recorded here so it is not undone.

**Re-measure if:** the watchdog gains an absolute threshold.

## C6 a criterion with no measurement is NOT MEASURED

**Current:** an unmeasured criterion blocks promotion but is never reported as a
failure, and anything selecting the next target tests `verdict == "fail"` rather
than "did not pass". NaN marks a missing sample and `inf` a genuine unbounded
bound; neither is ever a verdict.

**Why:** `leo_mjlab`'s sweep converted NaN to 0.0 and printed "FAIL, foot lift
+100%" for a policy whose foot lift had never been measured, while
`bad = [r for r in rows if not r[4]]` promoted unmeasured criteria as the next
target, since `not None` is true. This repository had the same class of defect
live: a missing `recovery_dcm_error` divided a zero default and scored as a
*perfect* recovery, and NaN comparisons silently passed six further gates.

**Locally:** enforced in two layers — `REQUIRED_METRICS` / `REQUIRED_TERMINATIONS`
checked before a rollout starts, then tri-state `Criterion` records in
`promotion()`. See [evaluation.md#unmeasured-is-not-a-verdict](evaluation.md#unmeasured-is-not-a-verdict).

**Re-measure if:** a new gate is added to `promotion()`.

## C7 the name of a metric is not the quantity it measures

**Current:** before a criterion is trusted, check that the quantity it reads is
the one its threshold was written for, and that the metric is published where the
reader looks.

**Why:** four `leo_mjlab` metrics collapsed on inspection. A touchdown-velocity
threshold was applied to the pre-contact velocity *peak*, which is strictly
larger, so the criterion could never pass and named itself the next target on
every sweep. Flatness was scored by counting contact patches the solver declares
loaded, where a sole parallel to the ground within 16 um scored 2.15 of 4. A
yaw-tracking column read NaN on every row of every sweep ever run, because the
command manager publishes it in the episode-end extras and the sweep reads the
per-step log. And `first_contact` fired on every regained contact, so solver
chatter billed the maximum error and dominated the term meant to raise the foot —
a minimum flight time of 0.05 s moved its measured peak 0.0009 -> 0.0076 m.

**Locally:** `Episode_Reward/*` sums correlate with episode length at r = +0.98,
and `zmp_error` must be read as `zmp_error / zmp_grounded`; both are instances of
this constraint.

**Re-measure if:** a metric is renamed, or a gate is pointed at a new quantity.

## C8 a single small-sample failure rate is not a verdict

**Current:** a fall rate from one small-`n` run does not decide anything. Paired
arms, a fixed episode count per environment, and a reported interval are what
make a difference readable.

**Why:** three `leo_mjlab` probes of *identical* configuration measured falls at
0.0156, 0.0195 and 0.0352 on 256 environments — a factor 2.3 on the criterion
that ranks before all others. They moved their default to 1024. This machine is
RAM-bound well below that, so the compensating mechanism is the paired design in
`compare_to_baseline.py` and `qualify_checkpoints.py`, not sample count.

**Re-measure if:** the environment count or episode allocation changes.

---

## Decision journal

Format: aim -> change -> measurement -> verdict, citing the constraints involved.
A rejected row is not retried without a later row saying what changed.

| # | date | aim | change | measurement | verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | 2026-09-14 | C6 | tri-state criteria, pre-rollout metric inventory, strict lookups in `qualify_checkpoints.py` | contract suite: every gate reports `not_measured` when its metric is dropped | **KEPT** — the qualifier could previously be fooled by a metric that was never measured |
| 2 | 2026-09-14 | C4 | qualification sidecar beside each checkpoint; full resume refuses an unqualified one | contract suite: missing, stale-digest and failing verdicts all refuse; actor-only warns | **KEPT** |
| 3 | 2026-09-14 | C1 | `grounded_fraction` criterion against the paired zero-residual arm | margin **not yet measured**; see [evaluation.md#grounded_fraction_margin](evaluation.md#grounded_fraction_margin) | **PROVISIONAL** — the gate exists, its threshold does not |
| 4 | 2026-09-14 | C2 | `placement_verdict`, kernel and guard roles, gate-conditioned quantiles in the audit | policy zero, 16 envs x 1000 steps: `recovery_dcm` admits 3200/16000 samples, gated median error **0.0411 m**, so `DCM_STD = 0.10` is **2.43x** | **OUT OF BAND** — floored side. `DCM_STD` 0.10 -> **0.06** (1.46x); one constant, one live term, weight unchanged. Predictions registered at [reward-shaping.md#recovery_dcm](reward-shaping.md#recovery_dcm); awaiting the run |
| 6 | 2026-09-15 | C2 | trained the `DCM_STD = 0.06` arm, 500 iterations, then swept `model_100..499` x {nominal, finite_impulse} x 2 seeds x 32 envs | paired recovery-DCM gain vs the zero-residual arm: **-0.24%, -1.00%, -0.10%, -1.46%, -2.96%** by checkpoint; grounded fraction 1.0000 throughout | **NULL** — the direction is right and grows with training, but never reaches the 5% bar. No checkpoint eligible. Note there is **no `DCM_STD = 0.10` control arm**, so this measures the policy against zero residual, not `0.06` against `0.10` |
| 7 | 2026-09-15 | C7 | read the two headline gate failures rather than their labels | `finite_impulse` hazard is **1.000 for BOTH arms** at every checkpoint, so `hazard_ratio` is exactly 1.000 and the `> 0.90` gate fails mechanically; `foot_slip` upper-CI ratios are 0.051-0.062 against a 0.050 limit on an absolute difference of ~1e-6 | **GATE DEFECT** — see [evaluation.md#saturated-hazard-cannot-discriminate](evaluation.md#saturated-hazard-cannot-discriminate). Neither failure is a statement about the policy |
| 5 | 2026-09-14 | C1 | measure whether the contact escape exists before shaping against it | policy zero, same run: `zmp_grounded` **100.0%**, n=16000, no spread | **NOT MEASURABLE** — the escape does not occur at this authority, so the grounded-floor arm would test an immunity nothing can observe. Arm withdrawn, [`GROUNDED_FRACTION_MARGIN`](evaluation.md#grounded_fraction_margin) kept as the instrument that would catch it |
