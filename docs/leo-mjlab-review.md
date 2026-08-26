# `leo_mjlab` 107-commit review

**Current:** the exact fast-forward range `1fcdcee..bdce238` in
`~/git/leo_mjlab` was reviewed on 2026-08-26. It contains 107 commits and is
treated as experimental evidence, not as a source tree to merge.

**Re-measure if:** that checkout advances beyond `bdce238`, mjlab changes how it
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
| 1 (done) | Effective configuration and resume integrity | every new checkpoint contains a canonical audit record, a semantic resume contract, a policy-interface contract, and the active curriculum is reapplied after restoring the global counter |
| 2 (done) | Reward audit and shape contracts | policy-zero and trained rollouts expose raw value, effective weight, weighted rate, active fraction, quantiles, and tensor shape for every reward |
| 3 (done) | Stratified qualification metrics | reports separate startup/sustained behavior, nominal/disturbed regimes, direction/axis, termination cause, per-joint authority, grounded masks, and recovery windows |
| 4 | Unattended-run watchdog | a run can warn, preserve a checkpoint, or stop on sustained degradation without killing the inherited checkpoint or mistaking a stopped trainer for watchdog failure |
| 5 | Achievement-gated curriculum | difficulty advances only after repeated held-out qualification, retains rehearsal of earlier stages, and rolls back on regression |
| 6 | Objective changes | one isolated hypothesis at a time, calibrated against raw magnitudes and accepted only by the baseline-relative qualifier |

Tranches 1 and 2 are implemented because later measurements are not trustworthy
if a run can silently use different defaults, resume at the wrong curriculum
stage, or optimize an inert or incorrectly shaped reward. Tranche 3 passed its
live output contract. Tranche 4 is the next recommended code change.

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
| Record source identity for hidden semantics | hash defining callable files and core manager/runner modules in addition to package versions | adopt now |
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

The local task already has frozen and gradual diagnostic schedules. Those remain
diagnostics. They should not become the main curriculum until tranche 3 provides
stable promotion metrics and tranche 4 provides rollback automation.

## Adopt later: unattended experiment operations

| Idea from the range | Local interpretation | Decision |
| --- | --- | --- |
| Attach a watchdog to an existing trainer | accept an explicit PID/run directory and verify both before monitoring | adopt |
| Use warn/preserve/stop levels | distinguish a noisy sample from sustained degradation | adopt |
| Guard degradation, not inherited state | establish a post-resume baseline window before applying stop rules | adopt |
| A stopped trainer is not watchdog failure | report clean completion separately from crash/OOM/watchdog termination | adopt |
| Checkpoint before intervention | request or copy a recoverable checkpoint before stopping a deteriorating run | adopt where runner hooks permit |
| Queue runs behind GPU availability | preserve the exact resolved command/config and launch in tmux | adopt in Python tooling |
| Emit a machine-readable verdict | every rung records pass/fail/ambiguous, metrics, checkpoint, and next action | adopt |
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
2. Add a structured watchdog that monitors the qualifier metrics, GPU memory,
   controller-worker restarts, trainer liveness, and checkpoint age.
3. Define curriculum promotion/rollback thresholds only after two independent
   seeds establish baseline variance for the new stratified metrics.
4. Implement stage state in checkpoints, sample easier stages after promotion,
   and validate uninterrupted-versus-resumed stage traces.
5. Run objective ablations only after these gates pass; begin with the failure
   variable identified by the baseline-deviation map, not with a borrowed gait
   reward.
