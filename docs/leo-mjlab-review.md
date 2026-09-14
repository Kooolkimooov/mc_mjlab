# `leo_mjlab` 150-commit review

**Current:** `1fcdcee..bdce238` (107 commits) was reviewed on 2026-08-26;
`bdce238..579183f` (43 more, to 2026-09-02) was reviewed on 2026-09-14 and is
recorded in "Second range" below. Both are treated as experimental evidence, not
as a source tree to merge.

**Re-measure if:** that checkout advances beyond `579183f`, mjlab changes how it
serializes configs or restores `common_step_counter`, or the residual task stops
using manager-based rewards and curricula.

**History:**

- 2026-08-26 — mapped every reusable idea into the adoption matrix below and
  started the first implementation tranche: effective-config manifests and
  resume curriculum continuity.
- 2026-08-26 — completed the second tranche with a live reward-call recorder,
  policy-zero/checkpoint audit, shape and finite-value contracts, conditional
  denominators, and effective curriculum-weight histories.
- 2026-08-26 — implemented the third tranche as additive qualifier outputs:
  startup/recovery/sustained windows, body-frame push directions, attributed
  terminations, grounded tracking, and per-joint authority/effort diagnostics.
- 2026-08-26 — completed the fourth tranche with PID-bound heartbeats,
  preserve-before-stop requests, GPU/checkpoint/worker/qualification guards,
  and distinct completion, failure, unexpected-exit, and watchdog-stop verdicts.
- 2026-09-14 — reviewed the 43 commits added since `bdce238`; they add one hard
  rule (the landing-cost escape), two calibration rules, and five measurement
  failures, all recorded under "Second range".
- 2026-08-26 — completed the fifth tranche with held-out two-pass advancement,
  reset-cohort rehearsal, three-regression rollback, preserved stage winners,
  and checkpointed stage state.

## Scope and reading rule

The range adds about 1.19 million lines, but 1.18 million are three committed
trainer logs. The reusable substance is in nine experiment notes, 25 automation
scripts, the RHPS1 ablation matrix, and changes to rewards, observations,
curricula, events, assets, and PPO settings. Its robot, objective, and policy are
different from this repository's mc_rtc-residual problem. Therefore the methods
and failure modes transfer more strongly than the final numerical settings.

The most important finding is methodological: several apparent reward or code
regressions were eventually traced to a wrong baseline, a wrong checkpoint, an
inert term, an unresolved site, a conditional metric read as unconditional, or
a value supplied by a callable default that was absent from the YAML dump. This
repository should make those classes of error mechanically difficult before it
adds a larger curriculum.

## Recommended implementation order

| Order | Tranche | Exit condition |
| --- | --- | --- |
| 1 (implemented) | Effective configuration and resume integrity | every new checkpoint contains a canonical audit record, a semantic resume contract, a policy-interface contract, and the active curriculum is reapplied after restoring the global counter |
| 2 (implemented) | Reward audit and shape contracts | policy-zero and trained rollouts expose raw value, effective weight, weighted rate, active fraction, quantiles, and tensor shape for every reward |
| 3 (implemented) | Stratified qualification metrics | reports separate startup/sustained behavior, nominal/disturbed regimes, direction/axis, termination cause, per-joint authority, grounded masks, and recovery windows |
| 4 (live checked) | Unattended-run watchdog | a run can warn, preserve a checkpoint, or stop on sustained degradation without killing the inherited checkpoint or mistaking a stopped trainer for watchdog failure |
| 5 (rejected at stage 0) | Achievement-gated curriculum | difficulty advances only after repeated held-out qualification, retains rehearsal of earlier stages, and rolls back on regression |
| 6 | Objective changes | one isolated hypothesis at a time, calibrated against raw magnitudes and accepted only by the baseline-relative qualifier |

These statuses describe code and evidence separately; they do not claim policy
capability. Tranches 1 and 2 are implemented because later measurements are not
trustworthy if a run can silently use different defaults, resume at the wrong
curriculum stage, or optimize an inert or incorrectly shaped reward. Tranches 3
and 4 passed their live output contracts. Tranche 5 has a code path, but the real
run stayed at stage 0 and confirmed that qualification still needs a manual loop.

## Adopt now: experiment identity and resume integrity

| Idea from the range | Local interpretation | Decision |
| --- | --- | --- |
| Prove the policy-zero baseline rather than trusting a named run | generate or verify a clean zero-residual checkpoint and store its identity in each comparison | adopt; existing clean-checkpoint tooling is the base |
| Treat the checkpoint as a first-class experimental input | validate policy interface, base-controller files, and semantic training config separately | adopt now |
| Read effective callable defaults, not only YAML fields | record explicit parameters and signature defaults from live manager terms | adopt now |
| Audit live manager configs after entity resolution | serialize the configs actually held by the managers, including resolved scene entities and instantiated class terms | adopt now |
| Diff training, play, and qualifier configs | keep an actor-interface contract strict while permitting evaluator-only corruption and runtime changes | adopt now; add a human diff CLI later |
| Preserve curriculum position over resume | restore the counter and immediately recompute mutable curriculum targets | adopt now |
| Detect a checkpoint inherited from a different experiment | reject semantic full-resume mismatches even when model tensor shapes happen to fit | adopt now |
| Record source identity for hidden semantics | record defining callable files and core manager/runner modules without making source bytes the semantic resume contract | **done 2026-08-27**; audit only, excluded from both enforced contracts |
| Keep operational changes from invalidating evaluation | exclude viewer, environment count, controller-worker count, console output, corruption, and observation latency from the actor-only interface | adopt now |
| Keep old checkpoints evaluable | warn when a legacy checkpoint lacks the new manifest; retain the existing controller-provenance check | adopt now |

The complete audit record intentionally differs from the enforced contracts.
`record_sha256` answers “was anything different?”; `training_sha256` answers
“is this a semantically equivalent continuation?”; `policy_interface_sha256`
answers “does this actor still consume and produce the same quantities?”

## Adopt next: reward auditing and objective contracts

| Idea from the range | Local interpretation | Decision |
| --- | --- | --- |
| Map every effective reward contribution | report raw mean, effective weight, weighted rate, episode contribution, active fraction, and sign | implemented |
| Resolve effective curriculum weights | reward reports must use the live manager weight at the sampled step, not the initial YAML value | implemented |
| Calibrate changed rewards with paired policy-zero rollouts | compare old/new term values on identical states before spending a training run | next tranche |
| Give stateful reward variants separate instances | never share landing counters, debouncers, histories, or class-term caches across an A/B pair | next tranche |
| Assert reward output shape | require `(num_envs,)`; a broadcasting bug can produce plausible aggregate curves | implemented |
| Resolve scene entities through the manager | audit terms after `SceneEntityCfg.resolve`, not with independently guessed site or joint ids | implemented |
| Make time units explicit | distinguish per-step, per-second, per-episode, touchdown-event, and conditional-contact objectives | implemented for the current reward family; extend with new event terms |
| Account for `dt` once | calibrate the manager's weighted rate and integrated episode contribution; do not hand-correct by an assumed control frequency | implemented |
| Expose inactive or inert terms | flag zero active fraction, unchanged target state, or zero gradient-relevant contribution | nonzero output and zero weighted contribution implemented; target-state and gradient probes remain later |
| Track conditional denominators | pair contact-only, grounded-only, recovery-only, and touchdown-only metrics with sample counts | grounded and recovery implemented; add term-specific denominators with new gated rewards |
| Separate price from ceiling | tune constraint-like shaping by violation frequency and magnitude before changing its weight | adopt as tuning protocol |
| Use feasibility as the first gate | reject reward proposals whose target is outside controller authority or actuator limits | adopt; integrate residual-authority and baseline-deviation probes |

The RHPS1-specific conclusions—exact foot-height targets, torque weights,
single-support prices, stride periods, and touchdown thresholds—must not be
copied. The transferable result is the calibration method and the discovery
that a term's name often did not match the physical event it measured.

## Adopt next: diagnostic and qualification coverage

| Idea from the range | Local interpretation | Decision |
| --- | --- | --- |
| Split startup from sustained behavior | score controller transient, recovery window, and post-recovery hold separately | next tranche |
| Split nominal from randomized behavior | paired seeds with nominal physics and each randomization family isolated | next tranche |
| Sweep checkpoints, not only the last one | catch improvement followed by collapse or policy drift | next tranche |
| Evaluate directions and axes separately | forward/backward/lateral/yaw pushes and CoM deviations need separate cells | next tranche |
| Attribute failure causes | report tilt, low height, controller failure, worker failure, timeout, and unloaded feet independently | next tranche |
| Log per-joint torque and clipping | add per-joint residual saturation and actuator-demand ratios, plus leg/arm aggregates | next tranche |
| Log command-tracking error | for the walking base controller, compare realized motion with the controller's requested motion | next tranche |
| Measure sole geometry, not only foot-center height | lowest sole point, tilt, support loading, and touchdown impact are the physically relevant landing quantities | adopt when foot diagnostics are expanded |
| Measure hover and flight explicitly | report double-flight, single support, grounded fraction, flight duration, and touchdown rate | adopt when gait qualification is expanded |
| Use fixed-camera video as evidence | save a deterministic side and three-quarter view with ground-relative motion visible | adopt in evaluator video tranche |
| Keep videos short and frequent enough to diagnose | a small clip at qualification checkpoints is more useful than a rare long render | adopt; retain storage limits |
| Add narrow probes before a clean run | test authority, observation drift, natural command tracking, and plant response without training | adopt as preflight suite |
| Compare rollout and training-loop measurements | detect wrapper, reset, randomization, and command-injection discrepancies | adopt after reward audit |
| Verify play corruption is actually disabled | actor-only evaluation may disable noise, but that change must be intentional and visible | covered by audit record, not actor-interface rejection |
| Treat sensor bias coherently | bias coupled velocity and gravity channels together rather than perturbing one inconsistent signal | later randomization tranche |

## Adopt later: curriculum design

| Idea from the range | Local interpretation | Decision |
| --- | --- | --- |
| Start from the last verified feasible behavior | use zero residual or a qualified checkpoint, never an unverified late checkpoint | adopt |
| Gate progression on achievement | advance on held-out survival/recovery/tracking thresholds, not global step alone | primary curriculum design |
| Require repeated passes | use hysteresis and multiple evaluation windows so one lucky batch cannot advance | primary curriculum design |
| Retain earlier-stage rehearsal | mix easier disturbances after advancement to reduce forgetting | primary curriculum design |
| Roll back on sustained regression | return to the last passed stage while preserving the last good checkpoint | primary curriculum design |
| Increase one physical challenge at a time | separate push magnitude, direction breadth, randomization, observation latency, and residual authority | primary curriculum design |
| Ramp continuously inside a stage | avoid a single step discontinuity in disturbance or objective coefficients | adopt where a continuous parameter exists |
| Keep standing practice | mix no-push/standing episodes because recovery policies can lose nominal balance | adopt |
| Curriculum both target and guard | pair a harder recovery demand with fixed torque, clipping, and collapse gates | adopt |
| Persist relative parameter bases | staged relative changes must apply to an immutable starting value, not compound on the last call | adopt with unit contracts |
| Count curriculum calls deliberately | choose environment steps, policy steps, episodes, or events explicitly; never infer that `common_step_counter` means calls | adopt with naming/unit checks |
| Change deployment scale separately | a cap that is safe during training may be too permissive at deployment; record both | later, after authority qualification |

The local task's frozen and gradual schedules remain diagnostics. The separate
achievement task consumes the tranche-3 qualifier at iteration boundaries,
keeps standing and prior stages in its reset cohorts, and checkpoints the state
that the tranche-4 watchdog can preserve.

## Unattended experiment operations

| Idea from the range | Local interpretation | Decision |
| --- | --- | --- |
| Attach a watchdog to an existing trainer | accept an explicit PID/run directory and verify both before monitoring | implemented |
| Use warn/preserve/stop levels | distinguish a noisy sample from sustained degradation | implemented |
| Guard degradation, not inherited state | establish a post-resume baseline window before applying stop rules | implemented for workers and qualification |
| A stopped trainer is not watchdog failure | report clean completion separately from crash/OOM/watchdog termination | implemented |
| Checkpoint before intervention | request or copy a recoverable checkpoint before stopping a deteriorating run | implemented at the iteration boundary |
| Queue runs behind GPU availability | preserve the exact resolved command/config and launch in tmux | adopt in Python tooling |
| Emit a machine-readable verdict | every rung records pass/fail/ambiguous, metrics, checkpoint, and next action | implemented for watchdog decisions |
| Isolate ablations | change one semantic unit at a time; cumulative ladders confound interactions | adopt |
| Allow paired or weighted variants | if a weight is the question, run identical state samples before multiple trainings | adopt |
| Preflight environment viability | policy zero and controller-only rollouts must pass before PPO starts | adopt |

## Do not port directly

| Item | Reason |
| --- | --- |
| RHPS1 reward weights, step periods, height targets, impact limits, and torque ceilings | different robot, actuators, controller, and objective |
| The 1,600-line environment-variable ablation switchboard | valuable as a lab notebook, but too implicit and too large for a durable task API; use typed named configs and manifests |
| Shell chains as the experiment database | fragile after crashes and difficult to audit; retain tmux ownership but put orchestration state in structured Python/JSON |
| Committed multi-hundred-thousand-line trainer logs | store summaries, tables, run ids, and artifacts outside source control |
| Static “corner count” or solver-proxy contact metrics | measure sole pose/loading and the physical event directly |
| A termination price chosen from episodic sums | episode length dominates sums; use rates, hazards, and matched exposure |
| A curriculum adopted before its gate exists | step schedules can automate a bad objective faster, but cannot establish improvement |
| A final-checkpoint-only verdict | can miss the best checkpoint and hide late instability |
| Cumulative ablation verdicts | interactions prevent attribution; isolate first, then test the minimal combination |
| Training from a suspect checkpoint | the range repeatedly showed that checkpoint identity can dominate several reward changes |

## Concrete backlog after tranche 1

1. Add `scripts/diff_effective_manifest.py` so rejected resume contracts and
   intentional evaluator differences are easy to inspect without loading a
   controller.
2. Re-measure curriculum promotion/rollback thresholds after qualification runs
   establish variance beyond the initial two-seed minimum.
3. Run objective ablations only after these gates pass; begin with the failure
   variable identified by the baseline-deviation map, not with a borrowed gait
   reward.

## Second range: `bdce238..579183f`

Forty-three commits, 2026-08-26 to 2026-09-02. The range is narrower than the
first and more conclusive: it ends with a policy that passes six acceptance
criteria out of six, and it converts several of the first range's observations
into stated rules. `docs/objectifs_et_decisions.md` there is the artifact worth
copying in form — objectives, seven numbered hard constraints, and a decision
journal where every row is *aim → change → measurement → verdict*, with the
rule that a rejected path is not retried until the document says what changed.
Constraint C7 below was violated five times before it was written down as
absolute; a numbered constraint list is what stops the sixth attempt.

### The landing-cost escape

Their C7, and the single most transferable result: **no cost attached to an
event the policy can stop producing may be increased** — not by weight, not by
ceiling, not by threshold. Five independent attempts to make landings cheaper
(contact balance, capture-point placement, free velocity, flat touchdown, and
swing height at triple weight) were each answered by the policy refusing to
land: air time 0.49 s → 1.55 s, falls 0.008 → 0.83. A gait clock paying 3.16/s
for kept rhythm did not make the escape unprofitable; the policy paid the clock
instead. A sixth apparent occurrence was retracted — it was a poisoned
checkpoint, not the ceiling (see below).

What worked instead pays for the *state* rather than penalizing the *event*:
`swing_height_bonus_dense` pays foot height per second of flight, so not landing
earns nothing extra, and `descent_speed_cost` charges downward velocity during
flight rather than at impact. Foot clearance moved 0.0058 → 0.0217 m after three
penalties against unreachable targets had moved it nowhere.

Local reading: `foot_slip`, `dcm_stability` and `recovery_dcm` are all gated on
ground contact, so all three pay zero with the feet unloaded. This repository
already measured that escape from the other side — `zmp_error` falls when the
robot spends more time off the ground, which is why `zmp_grounded` exists as its
denominator. The rule to carry over is that raising any contact-gated penalty's
weight requires watching the grounded fraction in the same run, and that a term
meant to *increase* a quantity should pay for it continuously rather than fine
its absence at an event.

### Clamped targets and saturated halves

Two rejected changes turned out to be the same arithmetic error, and it applies
to any kernel or ratio in this repository's reward set.

- A clamped target must sit **1.3 to 1.5 times above the measured value**. Below
  the measurement the ratio pins at 1.0 and the gradient is exactly zero; far
  above it the kernel floors and the gradient vanishes again. `periodlive` was
  rejected for lowering a target onto the measurement (`clamp(elapsed/target)`
  0.52 → 1.0, traction cut rather than concentrated), and a separate term sat at
  a target 2.2x above everything the gait did, making it a constant.
- In an **additive** mix, a saturated half does not transfer its weight to the
  other half — it becomes a constant bonus per event, which rewards producing
  more events. `0.5*(d_ratio**2 + p_ratio**2) / step_dt` with the distance half
  pinned shortened the step period 0.204 → 0.164 s, the opposite of the demand.
  Both halves must pull the same way at the operating point.

Tranche 2's audit already reports raw magnitudes per term; the addition is to
use them to *place* a target before a run rather than to explain one after it.
`DCM_STD` and `torque_margin`'s `soft_ratio` are the local values this applies
to.

### `step_dt`, third occurrence

`RewardManager` returns `raw * weight * dt`, so a term paid once per event
becomes a rate proportional to the event *count*: a bonus per landing rewards
landing more often, a cost per landing rewards landing less. A touchdown bonus
ran at 1.3% of its intended size for two whole experiments because of it, and
the same defect had already been recorded twice on other terms. This
repository's reward terms are continuous and per-second, so it is currently a
contract to keep rather than a bug to fix: any future event-paid term needs an
explicit unit statement, which tranche 2 already asks for.

### Checkpoint hygiene, and when to sweep

Four consecutive diagnoses blamed a reward change for what was a poisoned
starting point. Three checkpoints of that campaign — taken 150 iterations after
a resume, or just as a run turned brittle — contaminated everything restarted
from them, and one of them cost the C7 rule a false sixth data point. The rule
they settled on: **the only valid resume point is a checkpoint whose
deterministic sweep exists**, and never take a checkpoint without checking the
state of its run at that iteration.

Related, and it upgrades tranche 3's "sweep checkpoints, not only the last one"
from a nice-to-have: the mid-run checkpoint passed all six criteria while the
*same run* 3600 iterations later failed two of them (falls 0.0361, foot lift
0.0258). Three runs in a row showed that shape — late iterations convert margin
into stride and pay for it in falls. Sweeping only at apparent convergence
selects the worse policy.

Locally, `scripts/qualify_checkpoints.py` already sweeps every saved checkpoint
rather than the last one, which is the harder half
(`improvement-roadmap.md#checkpoint_qualification`). What the range adds is the
*resume* side: `residual_balance_runner.py` validates that a checkpoint's config
is a semantically equivalent continuation, but nothing records whether that
checkpoint was ever qualified. Writing the qualifier verdict beside each
checkpoint, and warning on a full resume from an unqualified one, is the cheap
version of this rule.

### Five ways a measurement lied

All five are cheap to guard against and expensive to discover late.

- **A criterion with no measurement is `NOT MEASURED`, never `FAIL`.** Their
  sweep converted `nan` to 0.0 and printed "FAIL, foot lift +100%" for a policy
  whose foot lift had never been measured — the metric was emitted only by terms
  that later configurations add. One layer up, `bad = [r for r in rows if not
  r[4]]` promoted an unmeasured criterion as the next target, since `not None`
  is true; the fix is `is False`.
- **A metric published only through `reset()` never appears in a non-resetting
  rollout.** A yaw-tracking column read `nan` on every row of every sweep ever
  run, including the ones used to judge policies, because the command manager
  returns it in the episode-end extras and the sweep reads the per-step log.
  The tensorboard tag existed, which is what kept it invisible.
- **The name is not the quantity.** A threshold written for touchdown velocity
  was applied to the pre-contact velocity *peak*, which is strictly larger, so
  the criterion could never pass and named itself the next target on every
  sweep. Flatness was scored by counting contact patches the solver declares
  loaded — a sole parallel to the ground within 16 µm scored 2.15 out of 4,
  because 130 µm of toe-versus-heel offset per milliradian of ankle pitch lifts
  two patches clear of the detection threshold. Sole tilt in radians is the
  solver-independent quantity.
- **Contact chatter can dominate an event-gated term.** `first_contact` fires on
  every regained contact, so a foot losing and regaining contact for one step
  counted as a landing with a peak of zero — and with cost summed as
  `error**2 * first_contact`, each micro-contact billed the maximum. The term
  meant to raise the foot was being set by solver chatter; gating on a minimum
  flight time of 0.05 s moved its measured peak 0.0009 → 0.0076 m. Any local
  term keyed on a contact transition needs the same debounce.
- **Sample size decides whether a fall rate means anything.** Three probes of
  identical configuration measured falls at 0.0156, 0.0195 and 0.0352 on 256
  environments — a factor 2.3 on the criterion that ranks before all others.
  They moved the default to 1024. This repository is RAM-bound well below that,
  so the transferable form is negative: a single small-`n` fall rate is not a
  verdict, and `compare_to_baseline.py`'s paired arms, fixed-`K` episodes per
  environment and reported intervals are the compensating mechanism.

### Training scale is not deployment scale

Their deterministic sweep read leg-torque clipping at 0.07 where the training
loop read 0.32 for the same policy — roughly a factor 10, all of it exploration
noise. Three healthy runs were nearly killed by watchdog caps calibrated on
training-scale numbers as though they were acceptance thresholds. The division
of labour they settled on: the watchdog says "this run is going wrong", through
*relative* drift from a post-attach baseline, and only falls are unambiguous
enough at training scale to justify stopping a run; the sweep says "this policy
is good enough". `training_watchdog.py` already guards on drift from a
post-attach baseline, so this range confirms the local design rather than
changing it.

### Smaller items

- **Self-collision distances must match the deployed QP's.** Their model was up
  to three times more permissive than the controller would be at run time
  (thighs 0.02 m against the QP's 0.06), which is consistent with the blocked
  lateral step they were chasing. Locally the QP owns collision avoidance and
  mjlab collisions are off by default, so this only becomes live if a preset is
  ever enabled for a robot.
- **`--agent.max-iterations` counts additional iterations, not an absolute
  target.** A 450-iteration probe asked for 18450 more and ran a day and a half.
  Probes of unequal length are comparable to nothing, which is the whole point
  of the method.
- **Wait on the run you launched.** A checkpoint glob across *all* run
  directories matched a previous run's file, so a probe's stop condition was
  true before training began and the sweep that followed measured the wrong run.
- **Manager overhead can dominate a single-environment step.** Two thirds of
  their `env.step` was managers, not physics, with rewards alone at 9.5 ms;
  stripping evaluation-only terms bought 36% more frames. Solver iteration
  counts changed nothing. Low relevance here, where the controller solve
  dominates, but it is a reminder to measure the split before optimizing the
  simulator.
