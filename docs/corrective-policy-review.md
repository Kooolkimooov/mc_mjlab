# Corrective review and policy-optimization scope

This is the disposition of the branch review at `2a15d18`. It separates useful
infrastructure from demonstrated capability, records every raised defect, and
defines the narrow policy correction that may run before the broader backlog.

## Branch verdict

**Current:** the reviewed branch comprised 19 commits and `+9,173/-276` lines
over three days. Static verification was clean, but 13 training arms and three
paired checkpoint qualifications produced no policy improvement. The branch
therefore delivered measurement and experiment infrastructure, not a promoted
policy. `docs/improvement-screens.md` states that result correctly.

The later achievement run did not change the verdict. It completed 500 PPO
iterations in 3 h 46 min with 3,623 timeouts, two collapses, and no controller or
worker failures. Stage 0 never passed. At `model_180`, finite-impulse hazard
improved from 9.375% to 6.25%, but recovery DCM error worsened by 0.172 mm;
robust hazard worsened from 78.125% to 90.625%, and the overall hazard ratio was
1.107. Late checkpoints remain unqualified, but screening them before correcting
the actor objective would spend compute on the known failure mode.

**Re-measure if:** a changed actor objective passes its deterministic contracts
and a policy-zero smoke run shows the intended active-sample fraction.

**History:**
- 2026-08-27 — accepted the review's central conclusion and stopped further
  checkpoint screening.

## Confirmed strengths

The review identified real work worth retaining:

- `SquashedGaussianDistribution.kl_divergence` uses latent-Gaussian KL, which is
  invariant under the bijective tanh transform.
- `project_residual` accounts for controller-side clipping by subtracting the
  clamped nominal target.
- `recovery_authority.measure` uses the correct CoP moment translation and DCM
  construction.
- `RolloutAdaptivePPO` fixes repeated per-minibatch learning-rate adaptation.
- `ControllerPool.close` fixes worker thread-pool and failed-build leaks.
- `verify_improvement_contracts.py` provides deterministic executable contracts.
- `recovery_detector.json` records its calibration seed, environment count,
  duration, duty, recall, and quantiles.
- `apply_qualification` is pure over frozen state, report validation is strict,
  evidence deduplication is semantic, and reset-cohort rehearsal avoids a global
  mid-episode difficulty jump.
- Achievement stage is independent of `common_step_counter`, its environment
  config removes the old torque-margin ramp so difficulty changes one variable,
  and rollback deliberately preserves policy and optimizer state.

These are engineering results. None is evidence that a residual policy improves
the controller.

## effective_training_manifest.py

**Current:** source-file and runtime-module hashes are folded into
`training_sha256`, so a comment-only source edit or dependency rebuild can block
full resume with no override. At the same time, the action-defining contents of
`etc/recovery_detector.json` are absent from controller provenance; only its path
appears in the environment config. The enforced identity is therefore stricter
on incidental source bytes than on an external file that controls action
authority.

**Decision:** deferred from the policy-only tranche. Split semantic enforcement
from audit identity, add an explicit and logged drift override, and content-hash
the detector before the next resumable long run.

## requested_action_l2

**Current:** the measured mean residual authority was 7.338%, leaving about 92.7%
of rollout steps unable to affect simulation. The old reward charged only the
executed residual, so inactive requests had no cost while ordinary PPO still
mixed their advantages into the actor update. This supplies variance rather
than causal action signal and is a plausible mechanical explanation for the 13
statistically indistinguishable arms.

The correction keeps all steps for critic targets but zeroes inactive advantages
before the PPO surrogate. Active advantages are normalized over active samples
and multiplied by total/active count so a 7.338% duty cycle does not shrink the
actor gradient by about 13.6x. The transformed entropy term remains all-step on
purpose: for a tanh Gaussian it regularizes the state-dependent mean toward zero
while maintaining the one shared exploration standard deviation. The action term
retains the transition's gate across an automatic terminal reset, so fall samples
are classified using the authority that actually produced them.

`residual_magnitude` and `residual_rate` now price the bounded normalized request
on active steps. They no longer get artificially cheaper during a fractional
authority ramp. Separate metrics retain both requested and executed magnitudes.

**Re-measure if:** recovery gating, action distribution, entropy structure, or
rollout duty changes.

**History:**
- 2026-08-27 — selected active-only surrogate learning over a local 300-line PPO
  fork; upstream critic, entropy, minibatch, recurrent, RND, and symmetry paths
  remain intact.

## Screen evidence

**Current:** the screen table reported six decimal places across a 0.3% ZMP
spread using one seed and one run per arm. Although the document calls those
training diagnostics rather than policy evidence, it shortlisted ankle authority
from those same digits. Paired qualification later rejected every shortlisted
checkpoint, confirming that the rank was not decision-grade evidence.

**Decision:** retain the raw measurements and rejection, but do not rank a new
training matrix until the active-only actor objective has a policy-zero parity
check and at least two independent training seeds.

## get_residual_joints

**Current:** intersecting all six RobotModule bound maps can silently remove a
joint from the action space. Only HRP5P was screened; JVRC1 and RHPS1 have not
been checked against the resulting action dimensions.

**Decision:** deferred. Make omitted joints explicit and validate all registered
robots before claiming cross-robot support for a trained policy.

## project_residual

**Current:** clamping nominal `q` and `alpha` changes the zero-residual controller
path relative to strict mc_mujoco parity and invalidates older baseline numbers.
The safety projection itself is correct, but its placement changed the baseline.

**Decision:** deferred. Establish a named parity path before interpreting a new
policy-versus-controller result.

## randomize_current_pd_gains

**Current:** the randomizer probes private `_kp` and `_kd` fields and otherwise
falls back to entity actuator gains. The torque action deliberately zeros those
entity gains, so a private-field rename would turn randomization into a silent
no-op.

**Decision:** deferred. Replace the private probe with a required action-term
protocol and fail when the selected control mode cannot be randomized.

## RecoveryFilter

**Current:** `max_active_s` can force authority directly to zero while a walking
reference is still slewing, causing a one-step command snap. The ordinary cutoff
path is safe because authority has already decayed near zero.

**Decision:** deferred. Route expiry through the same decay/slew contract before
training the walking-reference action again.

## _apply_datastore_commands

**Current:** a datastore value captured at burst onset can be restored up to two
seconds later, overwriting an intervening FSM update.

**Decision:** deferred. Restore relative to a live controller value or require a
callback contract that owns the override lifecycle.

## WALKING_REFERENCE_SCALE

**Current:** `(0.20, 0.15, 0.30)` permits a `-0.2 m/s` longitudinal delta against
the installed controller's `+0.1 m/s` command, including reverse walking. That is
not residual-scale authority.

**Decision:** deferred with the walking-reference action; it is not part of the
policy correction run.

## RecoveryCalibration

**Current:** dataclass/JSON fallback defaults disagree with the shipped file:
`max_active_s` is 1.75 versus 2.0 seconds and `rearm_s` is 0.25 versus 0.5
seconds. Omitting a JSON key silently selects a different detector.

**Decision:** deferred. Require every calibrated temporal key or make the code
defaults exactly match the versioned file.

## _select_residual_joints

**Current:** `leg[-2:]` and `leg[2:-1]` assume a six-DoF leg without asserting
that topology.

**Decision:** deferred. Validate the expected limb length and named joints before
slicing.

## residual_balance task registrations

**Current:** four residual-balance ids are registered by default: position,
torque, ankle, and ankle achievement. Together with the two zero-residual play
ids, mc_mjlab exposes six tasks. Ten completed ablations no longer build during
`import mjlab`; `MC_MJLAB_REGISTER_ARCHIVED_TASKS=1` restores their original ids
for old-checkpoint compatibility. The screen launcher now contains only the
supported standard and ankle comparison.

**Re-measure if:** a retained task is rejected, a new policy direction qualifies,
or an archived checkpoint needs a compatibility path beyond its original id.

**History:**
- 2026-08-27 — reduced 14 default residual registrations to four while retaining
  ten historical ids behind an explicit compatibility switch.

## verify_improvement_contracts.py

**Current:** the repository has a deterministic assertion suite despite guidance
that still says there is no test suite. The suite is the only executable coverage
for several training contracts.

**Decision:** this policy tranche extends it with masked-advantage assertions and
records its command in both working-instruction files, replacing their inaccurate
claim that no test suite exists. The demo remains the live controller verification.

## documentation identifiers

**Current:** the review found 20 headings that did not resolve to current code,
including roadmap pseudo-identifiers and stale reward names. The controller-pool
construction timeout also lost its local reason/link while retaining the tuned
formula. `vars(mujoco)["mjtSensor"]` suppresses typing without explaining the
binding mismatch, and durability claims rely on file-only `fsync` without a
parent-directory sync.

**Decision:** recorded here but deferred; these do not change the actor objective.
The correction tranche does not expand task registrations or experiment scripts.

## run_improvement_screens.py

**Current:** the script defaults to launching 13 training arms at 128 environments
and 30 workers. It is user-invoked, but the cost is not surfaced in the README or
guarded by the repository's per-training-run confirmation practice.

**Decision:** deferred. Do not use the matrix launcher for the policy correction;
run one explicit short seed, inspect it, and request authorization before any
long or multi-arm program.

## achievement_finite_impulse_curriculum

**Current:** `stages` stores stage indices where the base class interprets its
first element as a global-step threshold. The override makes that type pun inert
today, but the manifest records misleading parameters and a future route through
the base implementation would select late difficulty almost immediately.

**Decision:** deferred from policy optimization, required before another
achievement run.

## AchievementCurriculumBridge

**Current:** the runner only polls a report that an external human or process must
produce. The minimum advancement cost is two reports times four scenarios times
two seeds: 16 qualifier seed-scenarios per stage, at least 48 for three-stage
mastery, and 24 more for a three-report rollback at each stage.

The completed run exercised one real report ingestion: model 140 was accepted as
`regression_pending` at an iteration boundary. It did not exercise advancement,
checkpoint preservation, or rollback. Model 180 completed after the trainer had
already exited and was never consumed, directly confirming the unattended-loop
gap.

**Decision:** deferred. Add a request watcher and explicit checkpoint scheduling
before calling the curriculum unattended.

## AchievementState.from_dict

**Current:** construction occurs before schema validation, so an unknown future
field can raise `TypeError` instead of the intended unsupported-schema error.

**Decision:** deferred with the achievement corrections.

## last_good_checkpoint

**Current:** a stage-0 regression reset clears `last_good_checkpoint` while the
qualified stage-0 path remains in `qualified_checkpoints`.

**Decision:** deferred with the achievement corrections.

## _preserve_checkpoint

**Current:** file data is synced through a read-only handle and the parent
directory is not synced, so the documentation overstates durability. The same
pattern appears in the watchdog.

**Decision:** deferred with the achievement corrections.

## rehearsal_weights

**Current:** a four-element tensor is rebuilt and copied host-to-device at every
environment reset.

**Decision:** deferred; cache one tensor per stage when the event term is built.

## MATCHED_TASK_ID

**Current:** `Position-Ankle-Matched-Impulse` is the first configuration built to
be *measurable*, not just plausible. It changes one thing against the ankle
screen arm: the disturbance distribution now covers the magnitudes the qualifier
scores. Everything else — authority set, observations, reward weights, the
torque-margin ramp, PPO settings — is the ankle control's.

Two supporting corrections landed with it, both against the same measurement:
burst onset no longer pays the residual magnitude cost twice
(`requested_action_rate_l2`), and the paired interval no longer hides a sampling
verdict behind a policy verdict (`clusters_for_confidence`).

**Why this is expected to move the gates.** The three 2026-08-25 rejections all
produced a real recovery-DCM improvement of `-4.79%` to `-7.70%` and failed on
confidence, not on sign. The best of them needed 23 clusters and had 16. Matching
the training distribution should raise the effect; clustering by
`seed x environment` at 16 environments and two seeds supplies 32. Separately,
the stage-0 achievement run showed the hazard gate is reachable only if the
robust band is trained: finite-impulse hazard fell to `6.25%` while robust rose
to `90.625%` on the same checkpoint.

**Acceptance, in order.** Nothing here promotes a policy on training curves.

1. Static gates: format, lint, the assertion suite, the 56-diagnostic type
   baseline, prose hook, and `list-envs` discovery. Done.
2. One short live smoke: confirm the task builds, controller workers start, and
   `Episode_Metrics/impulse_speed` rises against the ankle arm. If it does not
   rise, the mixture is not reaching the sampler and nothing after this matters.
3. One seed to the ankle screen's 188-iteration budget, read against the
   archived ankle arm on the same diagnostics.
4. Paired qualification of its best and late checkpoints at 16 environments and
   two seeds, which is 32 clusters.

**What falsifies it.** Robust-scenario hazard not improving against the ankle
control, or `nominal` gait gates regressing — the mixture spends 20% of episodes
standing precisely to protect those, and losing them would mean the robust share
is too large rather than that the hypothesis is wrong.

**Re-measure if:** the qualifier's push magnitudes, the promotion hazard gate, or
the authority set changes.

**History:**
- 2026-08-27 — built after the branch review traced the 13 indistinguishable
  arms to a training support that never reached the scored magnitudes.

## Policy correction acceptance

**Current:** no new policy is promoted by this code change. The static gate,
live-smoke gate, and isolated learning seed pass their mechanical contracts;
paired controller-relative triage remains.

Before a training matrix:

1. formatting, lint, the assertion suite, the 56-diagnostic type baseline,
   prose hook, and task discovery pass;
2. two one-iteration live smokes logged finite losses and active actor fractions
   of 9.77% and 9.28%; the terminal-safe reset smoke logged requested residual
   L2 0.003109, requested rate L2 0.006423, executed residual L2 0.00000232, no
   fall, collapse, controller failure, or worker failure, and KL 0.021937;
3. one 188-iteration seed completed 6,160,384 transitions without infrastructure
   failure; 12.553% of samples updated the surrogate, but executed residual L2
   rose 65.5% and both stability errors were slightly worse than the old ankle
   screen;
4. paired evaluation of `model_100.pt` must beat the controller baseline before
   spending a second independent training seed or attempting promotion.

Manifest, baseline-parity, walking-reference, achievement automation, task
cleanup, and documentation repairs remain explicit blockers for their respective
features, not silent additions to this policy-only tranche.

**Re-measure if:** gate calibration, rollout length, policy objective, action
regularization, or diagnostic reduction changes.

**History:**
- 2026-08-27 — completed the seed-42 ankle comparison in 67 minutes; the actor
  objective is active, but training curves provide no improvement evidence.
- 2026-08-27 — passed deterministic contracts and live collection, update, and
  reset paths with four environments and two controller workers.
