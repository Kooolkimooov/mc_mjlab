# Residual locomanip implementation and training campaign

## Progress ledger

Implementation status and experimental outcome are independent. All timestamps
use Asia/Tokyo (UTC+09:00). Raw traces and checkpoints stay outside Git.

| Stage/substep ID | Status | Implementation/validation evidence | Experiment outcome | Artifacts | Next action/blocker | Last updated |
| --- | --- | --- | --- | --- | --- | --- |
| S0 plan | complete | Complete approved plan transcribed before source changes; commit containing this entry | not measured | this document | Freeze references | 2026-10-26 14:20 +09:00 |
| R1 review corrections (S2/S4/S8/S10) | complete | `df799f8`, `0f3f29a`; targeted/full Python tests, 17 native tests, lint, architecture and strict prose passed; four archived probes repaired | same-batch 14,305 frames; reordered 14,300 per arm: exact noise agreement; all six outcomes agree | `review-reanalysis/`, `review-noise/`, `review-noise-reordered/`, `review-noise-repeatability.json` | Use corrected evaluator for training validation | 2026-10-27 09:04 +09:00 |
| S1 references | complete | Inventory frozen at `f577463`; `original`, `no_magnitude` (study control) and `arms_2x` (production) all load through the provenance-checking runner; no reference unavailable | nominal: each completes 600-650 kg, fails 700+ | `references.json`, `phase-nominal/` | none | 2026-10-26 16:58 +09:00 |
| S2 qualification | complete | Mass dtype fix `bb98c3b`; `compare_study_runs.py`; six zero sources over batch 6/reversed batch 3 with recycled partial resets | zero vs zero: no outcome disagreement; spreads 0.0135 m, 0.04 s | `zero-smoke/`, `zero-reordered/`, `zero-plain-timed/`, `zero-repeatability.json` | Validation/held-out cohorts once candidates exist | 2026-10-26 15:20 +09:00 |
| S3 PD telemetry | complete | `pd_telemetry` `f963cb2`: motor table and chain trajectory identical, stepped-only samples, partial resets (7 tests) | live: no outcome disagreement, spread 0.017 m; +3% step time; rate traces sum exactly | `zero-telemetry/`, `zero-telemetry-rate/` | Used by S4 | 2026-10-26 15:20 +09:00 |
| S4 phase diagnosis | complete | Corrected `df799f8` reports: nominal 192, encoder 216, encoder-650 36 episodes; one retained trace per outcome | combined -1: nominal 12/12 complete, encoder 22/24 complete with two controller failures; earlier universal completion claim withdrawn | `review-reanalysis/`; original raw traces preserved | Negative arm/waist direction remains useful; all failures count in selection | 2026-10-26 20:27 +09:00 |
| S5 focused exploration | active | `StudySpec` arms-waist/coordinated `071c50c`, init fix `ed098ff`; live: level-0 requests equal zero and identical joint requests agree across all three tasks | not trained | `s5-mapping/`, `review-reanalysis/s5-mapping/` | Train after fresh control (S9 items 2-3) | 2026-10-26 20:27 +09:00 |
| S6 observation history | active | history-10/20 `6287f6f`; live family smoke: both build, recycle and match zero at level 0 (0.0042 m, 0.06 s); `model_1180` rejected | not trained | `history-smoke/` | Train in S9 items 4-5 | 2026-10-26 20:27 +09:00 |
| S7 objective audit | rejected (current evidence) | `059681f`; audit of 541 matched failure/success pairs from S4 nominal | current defect 354/541 (discounted); candidate leaves 312/541, 0/541 only without residual costs | `phase-nominal/objective.json` | Re-run when learned policies complete heavy loads | 2026-10-26 16:55 +09:00 |
| S8 robustness/transitions | active | scenarios `97244ba`; worker replacement and handover passed; `df799f8` restores first-step QP failures and keys noise by condition/time | installed-gain sensitivity motivates later tests; old noisy comparisons are sensitivity observations, not matched evidence | `scenario-smoke/`, `sensitivity/`, `worker-replacement/`, `review-noise/` | Robustness training branches depend on S9 validation | 2026-10-26 20:27 +09:00 |
| S9 campaign | active | `0f3f29a` coordinator; control initialization/PPO smoke completed two iterations; prior family/reset smokes passed | full run pending launch | `review-train-smoke/`; launch commands below | Launch seed-42 control with periodic validation | 2026-10-27 09:04 +09:00 |
| S10 identification | active | `mc_mjlab.identification` `57c92d8`: shared log, fits within synthetic tolerances (6 tests), robot-only hashed calibration gate | simulator 700 kg trace fitted on first half, held-out RMSE recorded; no robot data | `identification/zero-700-fresh.json` | Calibration procedure note; robot data external | 2026-10-26 15:55 +09:00 |

## Decisions and starting evidence

**Current:** Optimize safe completion under heavy payloads. Safety precedes
capacity, reward and tracking. HRP5P position-residual locomanip is the campaign
task; JVRC1 gets compatibility checks only. Walking must be preserved. Precise
cart placement remains diagnostic. Use original hardware-derived scales and
magnitude/rate costs of -0.05 for the study control; preserve production
registration, existing task IDs, supported checkpoints, provenance enforcement
and the frozen `locomanip_safety` task. Keep the 2x-arm checkpoint as a reference;
further scale increases are not planned.

**Re-measure if:** controller, robot, gains, dependency, actuator, observation or
evaluation semantics change. Historical references require their matching
configuration and original provenance; never bypass provenance enforcement.

**History:**
- 2026-10-26 — approved campaign transcribed here. October 26 measurements in
  [locomanip.md](locomanip.md#residual_scale_gains) found improved DCM from
  doubled arm authority without establishing a completion improvement. Preserve
  the original measurements and sample sizes in that topic note.

The eight incorporated ideas are focused exploration, coherent observation
history, exact pre-clamp PD telemetry, phase-resolved diagnosis, outcome-based
qualification, isolated robustness tests, transition/reset validation and
preparation for hardware dynamics identification. Physical trials are not
required; missing hardware data remains explicitly unmeasured.

## Record keeping and execution rules

Update the opening ledger after each implementation substep, validation result,
launch, completion, blocker or direction change. Commit updates with each source
milestone or result record. Before ending a session record the exact next action
and running tmux jobs. A stage exits only with evidence for its exit conditions;
implemented code alone is not a policy improvement. A skipped branch cites the
decision rule and measurement that made it unnecessary. Preserve rejected
hypotheses and identify what changed before a retry. Amendments are dated, name
affected stages, and retain original decisions in History.

Every experiment entry records hypothesis/single intended change; control and
hard constraints; exact command and source revision; configuration/provenance
hashes and seed; run directory and tmux location; checkpoint IDs and hashes;
budget and actual progress; evaluation conditions and sample sizes; exclusions,
uncertainty and artifacts; verdict and next action. Long runs also record launch
configuration, milestones, latest inspected checkpoint and next evaluation point.
Use Current / Re-measure if / History for measurement notes. Link topic notes
instead of duplicating conflicting conclusions.

Commit every source/configuration dependency before dependent runs. Leave the
machine-local `[tool.uv.sources]` override uncommitted. Use `uv run`; launch long
jobs in tmux, reusing existing sessions/windows where available. Use generated
task IDs from `list-envs`, never hand-assemble them. Run targeted checks while
developing, then applicable Python/native, lint/type, prose and generated
architecture checks. Account for pre-existing diagnostics explicitly.

## S0 — Commit the complete plan

Create this authoritative document and link it from `docs/README.md`. Include
decisions, interfaces, stages, dependencies, budgets, scenarios and promotion
criteria. Exit: plan and initial ledger committed before source changes or runs.

## S1 — Freeze references

Record repository/dependency revisions, installed/user controller configurations,
controller binaries, PD gains and effective configurations. Identify and hash
zero-residual baseline, original-scale `model_1180`, no-magnitude-cost
`model_1660`, and 2x-arm `model_1280`. Build the original-scale study control
without changing production registration. Load historical checkpoints only with
matching configuration; mismatched provenance makes a reference unavailable
until restored. Update [Leo review notes](leo-mjlab-review.md) with the newly
reviewed range and this campaign link. Exit: reproducible references, explicitly
documenting unavailable inputs.

## S2 — Qualification before training variants

Extend the matched-condition evaluator with a public interface taking independent
candidate/reference task and checkpoint identifiers, a cohort definition and an
output directory. Output cohort manifest, episode records, summaries, optional
traces and qualification verdict. Run separate environments for different action
dimensions; condition IDs match exogenous inputs independently of row ordering
and batch size.

Full mass grid (kg): **400, 450, 500, 550, 600, 625, 650, 675, 700, 725, 750,
775, 800, 850, 900**. Transition subset: **600–800 in 25 kg increments**.
Validation: seeds **1001–1002**, **8 distinct encoder draws per mass and seed**.
Final held-out: seeds **2001–2003**, **32 distinct draws per mass and seed**.
Include fresh episodes and episodes following a completed **600 kg zero-residual
warm-up**. Nominal evaluation disables corruption and encoder bias; repeated
identical nominal states measure repeatability, not independent sample size.

Record actual mass, perturbations, lifecycle cohort, termination cause,
completion, displacement and phase exposure. Timeouts are measured
non-completions. Report worker failures separately and retry infrastructure
failures at most twice. Missing/non-finite measurements remain `not_measured`.
Report paired uncertainty, per-mass results, equal-transition-weight and
full-range-standardized results. Cluster repeats of one environment/input draw.
Only report a completion boundary when supported, otherwise a success curve.
Exit: zero-versus-zero agrees within measured repeatability, including partial
resets and reordered batches.

## S3 — Exact PD-demand telemetry

Add an optional repository-owned instrumented PD actuator. Capture after actuator
delay and before effort clipping: targets and measured position/velocity; P, D
and feed-forward contributions; unclipped demand and clipped command; policy
step peak absolute demand; saturation duration and longest continuous saturation;
contributions/timestamp at the total-demand peak. Read live gains/limits. Keep
independent P/D maxima separate from their values at the total peak. Reset
accumulators at policy boundaries, clearing only selected rows. Include final
substeps and terminal transitions. Aggregate arms, waist, legs and wrists; full
rate traces only for selected diagnostic episodes. Measure overhead because
overriding actuator computation disables fusion; ordinary training retains the
original path. Exit: identical-command torque equivalence and no systematic live
outcome change with instrumentation.

## S4 — Phase-resolved authority and failure mechanisms

Extend position probes with repetitions, seeds, lifecycle cohorts and profile
scales. Compare zero, reference policies and constant arm/waist/combined requests
at normalized **-1, -0.5, 0.5, 1** and initial masses **600, 650, 700, 750, 800,
900 kg**. Nominal first, then matched encoder draws. Segment by actual controller
readouts: before bilateral Hold, Hold entry, sustained Hold, exit/release and
terminal outcome. Preserve raw labels and freshness; retain windows **1 second
before to 2 seconds after** events. Measure delivery, targets, joint response,
raw demand, clipping, contacts, cart lag, height, tilt, slip, displacement and
reward contributions. Separate operating peaks and fall transients but retain
both. Exit: useful joint groups and failure onset timing evidenced; correlation
is not labelled causation.

## S5 — Focused exploration

Add typed specifications and explicit registrations in `locomanip_study`, using
the existing naming helper and standard train/play. Control: current residual
joints with original scales. Arms/waist: individual shoulders, elbows and waist,
others follow mc_rtc. Coordinated: two bounded policy outputs, shared normalized
arm request and shared waist request. Resolve measured groups by joint name;
expand through each original hardware scale and apply existing physical bounds
after expansion. Preserve exact zero equivalence. Separate policy actions from
joint requested/executed values; magnitude/rate costs use expanded normalized
joint requests; actor receives executed joint residuals. PPO, observations,
payload sampling and rewards stay fixed. Exit: mappings verified and controlled
training/evaluation results; restricted actions pass the same safety gates.

## S6 — Coherent observation history

Two full-action control variants: **10 frames (~0.2 s)** and **20 frames (~0.4 s)**.
History covers continuous actor base velocities, gravity, height/vertical
velocity, joint position/velocity/torque, executed residuals, planned CoM velocity
and sole velocities. Phase/completion stay instantaneous. True payload and hand
contacts stay privileged. Critic gets matching shared-channel histories. Preserve
noise-before-history, partial-reset isolation and resolved dimensions/order in
checkpoint contracts. Train from scratch, without widening or optimizer surgery.
Measure whether payload-dependent action changes precede outcome improvements.
Exit: isolated history results. Train a combined action/history variant only
when both components show independently useful validation results.

## S7 — Conditional objective experiment

Audit raw/weighted rewards on matched successful, failing and low-motion traces.
Compare undiscounted totals and discounted returns from start and Hold entry;
separate duration, deterministic behavior, exploration costs and DCM gains
without completion. Require a consistent measured ranking defect before a reward
experiment. Sole candidate: replace DCM reward with existing `is_alive` weight
**1.0**; retain termination, posture, slip, residual costs and diagnostic DCM.
Frozen safety remains unchanged. Replay over traces first: safe successful motion
must outrank observed failure and low-motion alternatives. Otherwise reject
without adding terms. Train separately from action/history. Exit: one qualified
objective variant or documented rejection.

## S8 — Isolated robustness and transitions

Typed scenarios: encoder bias/actor white noise separately and together;
persistent base-velocity bias; consistent IMU mounting rotation; correlated
velocity estimation error; PD gain variation; extra actuator delay; foot/cart
friction separately. Sensitivity settings (not hardware measurements):

| Mechanism | Settings |
| --- | --- |
| Velocity bias | +/-0.02, +/-0.05 m/s; each axis separately |
| IMU rotation | +/-1, +/-2 degrees; roll/pitch separately |
| Correlated velocity error | 0.02/0.05 m/s RMS; 0.1/0.5 s correlation |
| PD gains | 0.9/1.1 multipliers |
| Added actuator delay | 2, 4, 10 ms |
| Friction | 0.9/1.1 multipliers |

Scale immutable installed gains, never armature defaults. Reset persistent state
only for affected rows. Declare observation corruption while preserving interface
validation. Keep asynchronous controller delay. Exercise zero-to-residual
handover, grasp, Hold entry, release, recycled episodes and worker replacement.
During action holds, observations report delivered actions. Train at most two
justified variants: one isolated mechanism, then supported mixture; preserve
nominal rehearsal. Exit: measured robustness gain without nominal safety loss,
or documented sensitivity limit.

## S9 — Bounded training campaign

Common budget: **256 environments, 64 policy steps/rollout, 2,000 iterations,
existing PPO, checkpoints every 20 iterations**. Short initialization/reset smoke
checks precede each configuration family. Seed 42 isolated screens, then selected
configuration and control replicated on seeds **43 and 44**. Run order:

1. Fresh control.
2. Arms/waist.
3. Coordinated actions.
4. Ten-frame history.
5. Twenty-frame history.
6. At most one independently supported action/history combination.
7. At most one admitted objective variant.
8. At most two justified robustness variants.
9. Selected configuration and control on each additional training seed.

Maximum **13 full runs**. Document omitted branches. Evaluate every **200
iterations** on validation cohorts and inspect neighboring saved checkpoints
around the best eligible interval. Do not stop healthy runs based on early reward
or exploration-scale torque. Select by safety, safe completion, then physical
demand/residual use. Freeze configuration/checkpoints before held-out evaluation.
Exit: replicated qualified candidate or complete negative result.

## S10 — Hardware dynamics identification preparation

Shared simulator/future robot log: timestamps, frames, units, joint ordering,
controller targets, encoders, available link/IMU state, efforts and contacts.
Implement static-push/free-decay analysis, small residual step/frequency-response
probes, per-joint delay/low-pass fits and oscillation frequency/damping estimates.
Validate synthetic and simulator traces; fit on one segment set and predict
held-out segments. Opt-in lag/compliance models bind to calibration artifacts and
remain disabled without measured calibration. Distinguish motor encoders/link
motion in compliance and check numerical stability. Do not copy RHPS1 constants
or duplicate the controller already in training. Exit: usable tools/documented
calibration procedure; absent robot data remains unmeasured.

## Mechanism tests and promotion gates

Verify exact telemetry torque/peak/duration/live gains/terminal samples;
grouped mappings, joint costs and zero equivalence; history ordering/partial
resets; condition matching through batches/retries/arm reordering; distinct
completion/timeout/physical/infrastructure outcomes; perturbation resets and
installed gains; rejection of incompatible action/history checkpoints; synthetic
identification within declared tolerances. Live checks cover zero-residual
walking, full manipulation, light/transition/heavy loads, recycled episodes and
controlled worker replacement. Report deterministic and sampled actions separately.

A promoted candidate must satisfy all of these against zero and learned reference:

1. Complete valid matched measurements.
2. Physical limits and controller fault containment preserved.
3. No failures introduced in nominal cases completed by both baseline references.
4. No empirical increase of combined physical/controller failure in any predefined
   mass stratum; report tilt/collapse separately but judge their union.
5. Held-out safe completion gain over original-scale reference with paired **95%**
   interval excluding zero.
6. At least **95%** reference displacement on jointly completed matched cases,
   without more incomplete/timeouts.
7. Safety checks pass on both additional training seeds.

Inconclusive evidence never promotes. If none passes, retain production and
record a negative result. Campaign completion requires every stage evidenced as
complete, rejected, justified skipped or explicitly dependent on external data;
all jobs accounted for; committed final verdict and exact reproduction and
continuation path.

## Experiment ledger

**Current:** Training is authorized by the user's instruction to address the
review and continue the plan. Review fixes, live regressions and the control
initialization smoke passed; launch the fresh full-budget control next. Existing measurements remain in
[locomanip.md](locomanip.md); they motivate the campaign, not its qualification.

S9 launch commands in `mc_mjlab_training:2.1` (task defaults
already equal the budget: 256 envs, 64 steps, 2,000 iterations, saves every 20,
seed 42; about 6.4 h each from the 2x-arm run's 2,000 iterations in 6:24):

1. `uv run train Mc-Mjlab-Locomanip-Study-Locomanipcontroller-Hrp5P-Control`
2. `uv run train Mc-Mjlab-Locomanip-Study-Locomanipcontroller-Hrp5P-Arms-Waist`
3. `uv run train Mc-Mjlab-Locomanip-Study-Locomanipcontroller-Hrp5P-Coordinated`
4. `uv run train Mc-Mjlab-Locomanip-Study-Locomanipcontroller-Hrp5P-History-10`
5. `uv run train Mc-Mjlab-Locomanip-Study-Locomanipcontroller-Hrp5P-History-20`

Item 7 (objective variant) is skipped: S7 rejected the candidate. Items 6, 8 and
9 depend on validation results of 1-5.

**Re-measure if:** any reference or evaluator semantics change.

**History:**
- 2026-10-26 19:05 — session end: no job running; `mc_mjlab_training` windows
  `2` (training), `3` (study) and `4` (checks) are idle shells, `1` is
  TensorBoard. Exact next action: obtain approval, then launch S9 item 1
  (`uv run train Mc-Mjlab-Locomanip-Study-Locomanipcontroller-Hrp5P-Control`) in
  `mc_mjlab_training:2.1`; evaluate every 200 iterations with
  `scripts/evaluate_locomanip_study.py` on the validation cohort.
- 2026-10-26 14:30-17:45 — evaluation-only runs in `mc_mjlab_training`
  windows `2.1`, `study` and `checks` (zero smoke, telemetry, S4 probes, S5/S6
  smokes, handover, worker replacement, scenarios). Window `1.1` remains
  TensorBoard.
- 2026-10-26 14:20 +09:00 — source inspection only. Initial tmux inventory was
  inaccessible inside the sandbox; inventory requires local socket access before
  scheduling jobs. Local `pyproject.toml` dependency override is pre-existing.

## Review continuation: R1 and S9

**Current:** Fix revision `df799f8` preserves first-step controller failures,
selects the retained attempt's trace, records per-arm channel names, keys
observation corruption by condition/channel/episode time, and requires finite,
verified calibration artifacts. No reference or production task changed.

Corrected archives are under `logs/locomanip-study/review-reanalysis/`; originals
remain unchanged. For each of `phase-nominal`, `phase-encoder`,
`phase-encoder-650`, `s5-mapping`, reproduce with:

```sh
uv run --no-sync python scripts/repair_locomanip_reports.py logs/locomanip-study/NAME --output logs/locomanip-study/review-reanalysis/NAME --names logs/locomanip-study/phase-nominal/names.json
```

The four reports retain respectively 192/216/36/32 episodes and
127/134/14/16 combined failures. Six first-step QP failures are retained in the
650 kg rerun. Combined -1 completes 12/12 nominal cases, 22/24 encoder cases
(two controller failures), and 4/4 in the separate 650 kg rerun. These are small
diagnostic cohorts, not qualification. All four replayed alive-objective audits
reject the candidate. `reanalysis.json` records input identities/hashes;
`episodes.json` binds each record to its trace and channel metadata.

Noise regression command (`df799f8`, reference/candidate are the same original
checkpoint, hash `dd5530445dfa2f6e49e03a42ddd5687e637f9bacce069fe438e22e0640df6300`):

```sh
uv run --no-sync python scripts/evaluate_locomanip_study.py --cohort docs/locomanip-study-smoke.json --scenario velocity-error/0.05/0.5 --candidate-checkpoint logs/rsl_rl/Mc-Mjlab-Locomanip-Locomanipcontroller-Hrp5P-Position-Residual/2026-10-21_10-52-44_push-friction/model_1180.pt --reference-checkpoint logs/rsl_rl/Mc-Mjlab-Locomanip-Locomanipcontroller-Hrp5P-Position-Residual/2026-10-21_10-52-44_push-friction/model_1180.pt --batch-size 6 --workers 3 --trace --output logs/locomanip-study/review-noise
```

This completed with valid matched data, no worker/early-controller failures and
14,305 common trace frames whose `observation_noise/*` arrays agree exactly.
Six conditions per arm cover 400/700/900 kg fresh and recycled; this tests
reproducibility, not independent statistical improvement. The reordered run
used the same command with `--batch-size 3 --reverse` and output
`logs/locomanip-study/review-noise-reordered`, in `mc_mjlab_training:3.1`.
It completed with valid matched data and no worker/early-controller failures.
Each reordered arm matches 14,300 common baseline frames exactly on all noise
channels. All six conditions retain their completion/failure/timeout outcomes
across all four arms; maximum time spread is 0.120003 s and displacement spread
0.018695 m. `review-noise-repeatability.json` preserves per-condition counts.

The bounded coordinator `scripts/run_locomanip_study.py` launches one standard
train command and validates checkpoints 200 through 1800 every 200 iterations,
then the final `model_1999.pt` (2,000 zero-indexed training iterations). Every
validation uses the full 480-condition encoder-draw cohort per arm, compared to
the original reference. Failed/missing evaluations are recorded for attention;
they do not stop healthy training. Validation does not imply promotion. Status
and exact commands, revisions, checkpoint/report hashes live in `status.json`.
Neighbouring checkpoints, zero-reference checks and held-out qualification
remain subsequent selection work. Three coordinator tests verify backlog
draining, missing checkpoints and ambiguous run-directory rejection.

The two-iteration initialization/PPO smoke completed in
`mc_mjlab_training:2.1` at 2026-10-27 09:01, using source `0f3f29a`:

```sh
uv run --no-sync train Mc-Mjlab-Locomanip-Study-Locomanipcontroller-Hrp5P-Control --env.scene.num-envs 4 --env.actions.mc-rtc-residual.num-workers 2 --agent.max-iterations 2 --agent.logger tensorboard --log-root logs/locomanip-study/review-train-smoke --agent.run-name control-s42-smoke
```

The run is `review-train-smoke/Mc-Mjlab-Locomanip-Study-Locomanipcontroller-Hrp5P-Control/2026-10-27_09-01-21_control-s42-smoke`,
512 transitions, actor 531/critic 660 observations, 35 action outputs. It saved
`model_1.pt`, SHA-256
`3455368fcbc1e3ad28f7a18faec631db4d75c3868bf52d64c171a0dd76ca1fc5`;
effective manifest hash
`ef44981d3d50558a1e1639bc1472efee7cdaf0159d9701ea4ef2cb6ef92e3bec`;
base-controller manifest hash
`8170dc0981c52225d31a0292c1091cba17ff38dd73dac09adf7aca234e64cbdf`.
This short smoke verifies initialization and PPO only; the earlier family
smokes and recycled evaluation cover episode reset behavior.

Then launch the fresh control in the same pane:

```sh
uv run --no-sync python scripts/run_locomanip_study.py --variant control --seed 42 --reference logs/rsl_rl/Mc-Mjlab-Locomanip-Locomanipcontroller-Hrp5P-Position-Residual/2026-10-21_10-52-44_push-friction/model_1180.pt --output logs/locomanip-study/campaign-control-s42
```

Hypothesis: establish a reproducible fresh control at original hardware scales
and both residual costs -0.05; no intended change from the approved control.
Budget is 256 environments, 64 steps, 2,000 iterations, saves every 20. The
other configurations remain unlaunched; the 13-full-run cap is unchanged.

**Re-measure if:** evaluator, corruption clock, action mapping, runtime or
calibration validation changes.

**History:** Before the review, retry-excluded early failures and merged traces
could overstate completion and distort reward/phase summaries. The original
universal-completion statement is preserved in earlier commits and superseded
by the counts above. The earlier noise outcomes cannot be repaired into matched
draws retrospectively.

- 2026-10-27 09:04 — full precommit Python tests passed on both fix commits;
  `cmake --build build --target check-native`: 17/17 passed;
  `uv run --no-sync ruff check`, architecture `--check`, strict prose passed.
  Full `ty check` retains 165 diagnostics; the new coordinator, repair and
  coordinator tests pass targeted type checking. No new source changes since
  these checks. All study panes are idle following completed regressions/smoke;
  `mc_mjlab_training:1.1` remains TensorBoard. Next: launch the command above in
  pane `2.1`, inspect initialization/checkpoint 0, record the actual run identity.

## Amendments

- 2026-10-26 20:17 — S2/S4/S8/S10 review corrections: supersede the 18:25
  classification amendment below. Every QP failure counts as a controller
  failure, including the first policy step (ten controller periods can already
  observe residual-driven motion). Early timing is diagnostic only; retries are
  worker-only. Retained outcomes select one attempt's trace and per-arm column
  metadata. White and OU observation errors become condition/term/time keyed;
  old noisy comparisons are sensitivity observations, not matched evidence.
  Calibration rejects nonfinite parameters/timesteps and revalidates its artifact
  before enabling a model. Existing artifacts remain intact; corrected reports
  go in separate reanalysis directories. S2/S4 completion claims above refer to
  the earlier measurements and are under revalidation; no policy is promoted.
- 2026-10-26 20:17 — S9: the user's continuation instruction removes the previous
  session's approval blocker. The original 13-run limit and conditional branches
  remain. Begin with one fresh seed-42 control after the fixes are verified.

- 2026-10-26 18:25 — S2, S4 (classification, not a decision change): a
  `controller_failed` reported on the first policy step is recorded as
  `controller_start_failure`, retried at most twice like a worker failure and
  reported separately, because mc_rtc's first solve after a reset sees no
  residual. Evidence: draw 650/1 failed `QP failed to run()` right after
  `LMC::Initial_` in 17/18 probe episodes across all arms, 0/1 in a 2-row batch
  and 2/4 in a 24-row rerun with identical bias (max 0.0099 rad) and payload.
  Original rule: "Report worker failures separately and retry infrastructure
  failures at most twice"; it now names both kinds. Applied from `5913f0c`;
  records before it carry the failure as `E` (controller) outcomes.

## Reference identities and first evaluator milestone

**Current:** Reference inventory in `logs/locomanip-study/references.json`, created
with `uv run python scripts/freeze_locomanip_references.py --output
logs/locomanip-study/references.json` at source `f577463`. All three embedded
controller-provenance comparisons pass; actual runner loads remain pending.
The inventory records full configuration/provenance hashes and source revisions.
Local mjlab HEAD is `ba314ea2f37b724684af21b63a1c5dbe114a6cd7`.

| Reference | Run under `logs/rsl_rl/Mc-Mjlab-Locomanip-Locomanipcontroller-Hrp5P-Position-Residual/` | Checkpoint SHA256 |
| --- | --- | --- |
| original `model_1180.pt` | `2026-10-21_10-52-44_push-friction` | `dd5530445dfa2f6e49e03a42ddd5687e637f9bacce069fe438e22e0640df6300` |
| no magnitude `model_1660.pt` | `2026-10-22_10-00-47_no-magnitude-penalty` | `ad5113b21d6088e91036c208dbb15301feaeebf09fcb392409a0a66a7b8407ef` |
| 2x arms `model_1280.pt` | `2026-10-23_17-39-18_arm-scale-2x` | `61cc88a47078d3cbbf66bf75d82701db7002fec3f6662c5e02c17aeac1b518c3` |

The public study evaluator accepts candidate/reference task IDs and checkpoints,
`--cohort` JSON, `--output`, batching, bounded worker retries, traces and reverse
ordering. Default validation and `--held-out` select the planned cohorts. Omitted
checkpoints mean exact zero action. Both arms run independently, so action widths
can differ. Corruption is disabled; encoder draws are condition keyed. Recycled
rows first complete a nominal 600 kg zero-action episode, then reset individually
into their target payload/draw. Failed warm-ups are not target measurements.
Timeouts are measured non-completions; only worker failures are retried. Reports
cannot promote while campaign-wide gates remain pending. No boundary is fitted
until outcome evidence supports it.

**Re-measure if:** any file identity, rollout reset behavior or cohort changes.

**History:**
- 2026-10-26 — targeted cohort/control tests: 10 passed. Paired statistics are
  invariant to record ordering; lifecycle repeats share clusters; nominal repeats
  receive no population confidence interval. Missing/nonfinite values cannot pass.
  Generated architecture and strict prose checks passed at S1. The first type
  pass found four annotation errors; corrected before live execution.
- Next live command after committing this milestone:
  `uv run python scripts/evaluate_locomanip_study.py --cohort
  docs/locomanip-study-smoke.json --batch-size 6 --workers 3 --output
  logs/locomanip-study/zero-smoke`. Repeat with `--reverse --batch-size 3` into
  `logs/locomanip-study/zero-reordered`. These are repeatability tests, not
  qualification samples. Launch in existing `mc_mjlab_training:2.1`.

## Zero-policy repeatability

**Current:** Smoke cohort `docs/locomanip-study-smoke.json` (400/700/900 kg, seed
1001, one nominal draw, fresh and recycled: 6 conditions) at `bb98c3b`. Four
zero-policy sources, `zero-smoke` (batch 6) and `zero-reordered` (`--reverse
--batch-size 3`) with two arms each, agree on every outcome and termination
label. 400 kg completes at 49.26-49.28 s; 700 and 900 kg end `fell_over` at
46.18-46.22 s in both lifecycles. Spreads across the four sources: displacement
<= 0.0109 m, end time <= 0.040 s; adding `zero-plain-timed` (`cc7a8a2`, batch 6)
as two more sources keeps every outcome and widens displacement to 0.0135 m.
Runs are not bitwise repeatable, so live
equivalence checks are judged against these spreads. Report:
`logs/locomanip-study/zero-repeatability.json`, from `uv run python
scripts/compare_study_runs.py logs/locomanip-study/zero-smoke
logs/locomanip-study/zero-reordered`.

**Re-measure if:** rollout resets, payload application, batching or the
controller change.

**History:**
- 2026-10-26 14:30 — first live attempt stopped at the first completed warm-up:
  `RuntimeError: Float did not match Long`, integer JSON masses compared against
  the applied float mass. Fixed in `bb98c3b`; the failed directory is kept as
  `zero-smoke-failed-long-mass`.
- 2026-10-26 14:35-14:50 — the two runs above, in `mc_mjlab_training:2.1`.

## pd_telemetry

**Current:** `instrument_pd` replaces the 53 per-joint `IdealPdActuatorCfg`s with
one `InstrumentedPdActuator`. Its motors are created in the parts' order with
each part's limit/armature/friction (HRP5P motor table identical, test-checked),
and its torque is bitwise the ideal law (chain trajectory test). A sample counts
only in `update()`, after `sim.step()`, so the extra `write_data_to_sim` of a
reset never enters a window; windows open on each policy step's first substep and
episode resets clear only their rows. The evaluator flag `--pd-telemetry` adds
per-group (arms, waist, legs, wrists) episode peaks, contributions at the peak,
independent P/D maxima, saturation joint-seconds and longest run;
`--pd-rate-trace` saves one condition's full 1 kHz samples. Training keeps the
fused path. Smoke cohort, zero policy, `f963cb2`: `zero-telemetry` versus
`zero-plain-timed` has no outcome or termination disagreement, displacement
spread 0.017 m and end-time spread 0.04 s (plain runs among themselves: 0.0135 m,
0.04 s). Overhead 31.64 versus 30.70 ms per env step at 6 envs (9855/9856
steps), +3%.

**Re-measure if:** mjlab's actuator call order, the PD law, fusion or the
actuator configs change.

**History:**
- 2026-10-26 — an earlier per-joint draft would have sampled reset-time writes
  and cost ~37k kernel launches per policy step; replaced before any run.
- 2026-10-26 14:57-15:02 — first zero-policy demand on the smoke cohort: at 700
  kg LEP reaches 1.00x its limit near 38.5 s and LWRR 1.2x with ~1 s saturated
  near 37.5 s, about 8 s before the 46.2 s fall; 900 kg arms reach 1.06x. 400 kg
  stays below 0.38x on every group.
- 2026-10-26 15:02-15:12 — `zero-telemetry-rate` (`--trace --pd-rate-trace
  seed=1001/mass=700/repeat=0/lifecycle=fresh`): each arm's file holds 46,180 x
  13 x 53 samples spanning 0-46.179 s of the 46.18 s terminal episode, so final
  substeps are included; P + D + feed-forward equals raw demand exactly and
  clipped equals clamp(raw) on every sample. About 59 MB per file.

## Phase-resolved authority

**Current:** Nominal probe `logs/locomanip-study/phase-nominal` (`868cffb`;
masses 600, 650, 700, 750, 800, 900 kg; fresh and recycled; seed 1001, one
nominal draw; 192 episodes, all segmented). Outcome per mass, fresh|recycled
(C complete, F tilt fall, X collapse):

| Arm | 600 | 650 | 700 | 750 | 800 | 900 |
| --- | --- | --- | --- | --- | --- | --- |
| zero | CC | XX | FF | FF | FF | FF |
| original / arms_2x | CC | CC | XX | FF | FF | FF |
| no_magnitude | CC | CC | FF | FF | FF | FF |
| arms -1 | CC | CC | CC | CC | CC | XX |
| arms -0.5 | CC | CC | CC | CX | FF | FF |
| waist -1 | CC | CC | CC | XX | FF | FF |
| waist -0.5 | CC | CC | XX | FF | FF | FF |
| combined -1 | CC | CC | CC | CC | CC | CC |
| combined -0.5 | CC | CC | CC | CC | CC | FF |
| any +0.5 / +1 | FF | -- | FF | FF | FF | FF |

(waist +0.5 alone completes 600 kg; every other positive request fails
everywhere.) The controller timeline is identical in every arm: fresh bilateral
Hold at 11.42-11.46 s, release at 44.44-44.46 s (medians). All 127 failures end
after the scheduled release, 1.44-4.66 s (median 1.70 s); none before it.
Sustained-Hold peak tilt separates outcomes almost completely (successes median
0.110 rad, range 0.066-0.214; failures 0.602, 0.210-0.608), as does cart error
(0.391 m, 0.280-0.576 against 0.983 m, 0.564-1.066). Against matched zero, arms -1
lowers sustained-Hold cart error by 0.353 m and tilt by 0.342 rad on average; the
references by 0.06-0.09 m and 0.07-0.09 rad. These are interventions on the
request, but the leaning-on-the-cart mechanism they suggest is a correlation,
not an established cause. Useful groups: arms and waist, negative direction;
legs and wrists moved nothing in the earlier authority probe (`8ac6bb0`).

Matched encoder draws (`phase-encoder/`, `62da14b`; seed 1001, two draws per
mass, fresh|recycled, positive levels dropped after failing every nominal
condition). Per mass, draw 0 then draw 1:

| Arm | 600 | 650 | 700 | 750 | 800 | 900 |
| --- | --- | --- | --- | --- | --- | --- |
| zero | CF\|CF | FE\|FE | FF\|FF | FF\|FF | FF\|FF | FF\|FF |
| original | CC\|CX | XE\|XE | FF\|FF | FF\|FF | FF\|FF | FF\|FF |
| no_magnitude | CF\|CF | FE\|FE | FF\|FF | FF\|FF | FF\|FF | FF\|FF |
| arms_2x | CC\|CC | XE\|XC | FF\|FF | FF\|FF | FF\|FF | FF\|FF |
| arms -1 | CC\|CC | CE\|CE | CC\|CC | CC\|CC | CX\|CX | CC\|CC |
| arms -0.5 | CC\|CC | CE\|CE | FX\|FX | FF\|FF | FF\|FF | FF\|FF |
| waist -1 | CC\|CC | CE\|CE | FX\|FX | FF\|FF | FF\|FF | XF\|XF |
| combined -1 | CC\|CC | CE\|CE | CC\|CC | CC\|CC | CC\|CC | CC\|CC |
| combined -0.5 | CC\|CC | CE\|CE | FC\|FC | CC\|CC | FF\|FF | CC\|CC |

E is a QP failure at the first step for draw 1 at 650 kg in 17 of 18
arm-episodes; it is intermittent and precedes any residual, so it is now an
infrastructure start failure (see Amendments) and the 650 kg column is rerun.
Encoder bias costs the references and zero their 600 kg margin on draw 1; combined
-1 still completes every other condition from 600 to 900 kg.

**Re-measure if:** the controller's manipulation schedule, scales or payload
model change.

**History:**
- 2026-10-26 18:10-18:24 — draw 650/1 diagnosis: identical bias (max 0.0099 rad)
  and payload completed under zero in a 2-row, 2-worker batch
  (`draw650-diagnosis/`); stance joints sit at least 0.122 rad from their limits.
  A 24-row, 4-worker rerun with full controller output (`encoder24-diagnosis/`)
  failed it in 2 of 4 episodes with `[error] QP failed to run()` right after
  `LMC::Initial_`: intermittent and policy independent, so classified as an
  infrastructure start failure (see Amendments).
- 2026-10-26 18:26-19:03 — 650 kg rerun with retries (`phase-encoder-650/`,
  `5913f0c`), draw 0 then draw 1, fresh|recycled: zero FC|FS, original XC|XS,
  no_magnitude XC|FC, arms_2x XC|XS, arms -1 CC|CS, arms -0.5, waist -1, combined
  -1 and combined -0.5 all CC|CC. Draw 1 completes for every arm once mc_rtc
  starts; 4 of 36 rows (draw 1, recycled: zero, original, arms_2x, arms -1)
  still failed the first solve after three attempts and stay unmeasured. The
  start failure is frequent for this initial condition, not random noise; it is
  a controller start fragility to report, not a policy outcome.
- 2026-10-26 16:52-18:10 — command: `uv run python
  scripts/probe_locomanip_phases.py --output logs/locomanip-study/phase-encoder
  --references logs/locomanip-study/references.json --encoder-draws --seeds 1001
  --draws 2 --groups arms,waist,combined --levels -1,-0.5 --arms
  zero,original,no_magnitude,arms_2x,control/arms/-1,control/arms/-0.5,control/waist/-1,control/combined/-1,control/combined/-0.5`.
- 2026-10-26 15:16-16:50 — command: `uv run python
  scripts/probe_locomanip_phases.py --output logs/locomanip-study/phase-nominal
  --references logs/locomanip-study/references.json --workers 4`
  (`mc_mjlab_training:study`). All three references loaded through the runner.

## Objective ranking audit

**Current:** On the S4 nominal traces (541 matched failure/success pairs, no
timeouts without completion), failures score at least the matched success in
220/541 pairs undiscounted and 354/541 discounted from the start under the
current objective: a consistent defect. Without residual costs it persists
(411/541 discounted), so the DCM term itself prefers the failing runs. The
candidate (DCM replaced by `is_alive` at 1.0, costs kept) still leaves 312/541;
without residual costs it leaves 0/541 on every measure. The remaining defect
comes from the saturation-level residual cost the constant-request successes
pay (about -24.6 per episode for arms -1), which a learned success would not
necessarily pay. Under the plan's rule (the candidate as specified must clear
the defect) the objective experiment is **rejected on this evidence**; no term
was added.

**Re-measure if:** learned policies complete the heavy cohort, giving
successful traces without saturation-level requests; or rewards change.

## locomanip_study variants

**Current:** `StudySpec` registers five ids through `get_task_name`: control,
arms-waist, coordinated, history-10 and history-20, each one change from the
original-scale control. Arms-waist narrows `residual_actuator_names` to
`[LR](S[CPRY]|EP)` and `W[PRY]` (13 of 35 joints); the rest track mc_rtc.
Coordinated keeps those joints behind two outputs (shared arm, shared waist):
`GroupedResidualJointPositionAction` copies each output onto its joints before the
shared scale and projection, so costs see the expanded joint requests and the
actor the executed joint residuals. History variants give every continuous actor
channel and its critic copy 10 or 20 frames; `manipulation_phase` and
`task_complete` stay instantaneous and privileged critic terms are unchanged.
mjlab applies noise before history and resets history per row. Loading
`model_1180` into history-10 or arms-waist raises `ReferenceUnavailable` (state
dict shapes differ).

**Re-measure if:** joint names, residual joints, the observation set or mjlab's
history semantics change.

**History:**
- 2026-10-26 15:54-16:47 — `s5-mapping` (`ed098ff`, 700/800 kg, fresh and
  recycled, nominal, 4 conditions per arm): level 0 on arms-waist and on
  coordinated matches zero (identical outcomes; displacement within 0.0123 and
  0.0115 m, time within 0.04 s, inside the zero-run spread); arms -1 through
  control, arms-waist and coordinated agree to 0.0012 m and 0.02 s; waist -1
  through control and coordinated to 0.0108 m. Command: `uv run python
  scripts/probe_locomanip_phases.py --output logs/locomanip-study/s5-mapping
  --masses 700,800 --lifecycles fresh,recycled --profiles
  control,arms-waist,coordinated --groups arms,waist --levels 0,-1 --arms
  zero,control/arms/-1,control/waist/-1,arms-waist/arms/0,arms-waist/arms/-1,coordinated/arms/0,coordinated/arms/-1,coordinated/waist/-1`.
- 2026-10-26 17:06-17:40 — history family smoke (`history-smoke/`, `0dbd1b4`,
  700 kg, fresh and recycled, nominal): history-10 and history-20 at level 0 match
  zero (identical outcomes, displacement within 0.0042 m, time within 0.06 s);
  arms -1 completes both conditions under both, as under the control.
- 2026-10-26 15:48 — first live build of the coordinated term failed: mjlab's
  `BaseAction` sized buffers from `action_dim` before grouping existed
  (`AttributeError: _residual_action_dim`). Fixed in `ed098ff`; config tests could
  not reach it because the term needs live controllers.

## objective_audit

**Current:** `audit_locomanip_objective.py <probe>` ranks matched episodes per
condition: safe successes against failures and timeouts without completion, by
undiscounted total, discounted return from the start and from fresh bilateral
Hold, each with and without a steady-state bootstrap on truncated endings
(completion and time limit are both `time_out=True`). Objectives: current,
candidate (DCM replaced by `is_alive` at 1.0), and both without residual costs.
Admission rule: a defect in more than half of the pairs of some measure under the
current objective, and none left under the candidate. The termination penalty is
scaled by the 0.02 s step: one fall costs -4.0, not -200.

**Re-measure if:** reward terms, weights, step size, gamma (0.997) or the
termination time-out flags change.

**History:**
- 2026-10-26 — on `phase-smoke` (700 kg), zero and `original` failures outrank
  constant arms -1 successes under both objectives because the constant request
  pays -24.6 residual magnitude over 49 s; without residual costs the current
  objective still prefers the failures on every discounted measure while the
  alive objective does not. Constant requests are not a learned policy; the
  verdict waits for the S4 nominal traces.

## study_scenarios

**Current:** `--scenario` applies one sensitivity setting to both arms: encoder
bias (with encoder-draw cohorts), actor white noise, both, base-velocity bias
(+/-0.02, +/-0.05 m/s per axis), IMU mounting rotation of base velocities and
gravity (+/-1, +/-2 deg roll or pitch), Ornstein-Uhlenbeck velocity error
(0.02/0.05 m/s RMS, 0.1/0.5 s), installed PD gains x0.9/1.1 (scaled once after
the action term installs `PDgains_sim`, never the armature defaults), actuator
delay 2/4/10 ms, foot friction x0.9/1.1 (foot geoms and floor together, since
contacts take the larger coefficient) and cart pair friction x0.9/1.1.
Observation corruption is declared through `noise`, which the policy interface
excludes, so checkpoints still validate. White and correlated noise use the
global generator and are not condition keyed. The controller's own IMU feed is
not rotated.

**Re-measure if:** observation terms, actuator configs or contact definitions
change.

**History:**
- 2026-10-26 16:48-17:06 — Hold handover (`handover/`, `07b2f84`, 700/800 kg,
  fresh and recycled, nominal): executed residual is exactly 0 before fresh
  bilateral Hold in every handover episode. Arms -1 from Hold entry only
  completes all four conditions (sustained-Hold tilt 0.084-0.137 rad, against
  0.594-0.602 for zero); `original` from Hold entry reproduces its nominal
  outcomes (700 kg collapse, 800 kg tilt fall). Command: `uv run python
  scripts/probe_locomanip_phases.py --output logs/locomanip-study/handover
  --masses 700,800 --lifecycles fresh,recycled --references
  logs/locomanip-study/references.json --groups arms --levels -1 --handover
  --arms zero,handover/control/arms/-1,handover/original`.
- 2026-10-26 15:57-16:15 — `check_worker_replacement.py` (`d8df874`): SIGKILL
  of worker `0_1_0` 60 s into the reference arm of the smoke cohort; exactly its
  two 400 kg conditions reported worker failures and were retried in attempt 1;
  12/12 records measured, evaluator exit 0, and outcomes agree with the earlier
  zero runs (8 sources, displacement spread 0.017 m, time 0.06 s). The first
  harness attempt crashed on `/proc` names with spaces and is not a result.
- 2026-10-26 17:05-18:10 — remaining sensitivity (`sensitivity/`, one fresh
  nominal condition per mass): 10 ms delay and foot friction x0.9 leave every arm
  at its nominal limit (references 650 kg, combined -1 all of 600-800); velocity
  bias x +/-0.05 m/s, IMU pitch +/-2 deg, velocity error 0.05 m/s/0.5 s and white
  noise leave `original` and `arms_2x` completing 600 and 650 kg.
- 2026-10-26 17:05- — gain sensitivity (`sensitivity/`, `51c3d61`, fresh,
  nominal, one condition per mass): installed gains x0.9 make zero and both
  references fail at every mass from 600 kg (nominally they complete 600, and the
  references 650), while combined -1 completes 600-800 kg; gains x1.1 let zero
  and the references complete 600-700 kg and fail at 800, combined -1 again
  completes all four. Outcomes are very sensitive to +/-10% on the installed gains.
- 2026-10-26 16:17-17:04 — scenario smoke (`8f81e2b`), zero versus `original`
  at 650 kg fresh (`docs/locomanip-study-scenario-smoke.json`; one condition each,
  so no rate is estimated). `original` completes (49.3 s) under 10 ms delay, foot
  friction x1.1, cart friction x0.9, IMU pitch +2 deg, velocity bias x +0.05 m/s,
  velocity error 0.05 m/s/0.5 s and white noise; installed gains x0.9 make it fall
  (46.1 s). Zero ignores observations, so the observation scenarios leave it at
  its nominal collapse (47.7-48.0 s); cart friction x0.9 lets zero complete. Each
  run's `applied` record shows the change installed (friction 1.0 -> scaled on
  both feet and the terrain; cart pair tangential [0.5, 0.05]).

## identification

**Current:** `IdentificationLog` is the shared format (seconds, radians, N m;
`joint_names` fixes column order; frames declared; strict validation;
compressed `.npz` with a JSON header). Fits: whole-sample delay plus first-order
lag (grid then local refinement), free-decay frequency and damping, static-push
stiffness, single-frequency gain and phase; step and sine probe generators;
held-out prediction error. Synthetic recovery within 1 ms delay, 10% lag, 2%
frequency, 10% damping. `LagCompliance` stays the identity unless given a
calibration whose SHA-256 matches and whose `source` is `robot`; its lag uses
exact discretization (stable for every tau > 0) and compliance subtracts
effort/stiffness from the motor angle. Simulator check on the 700 kg zero trace
(`identify_joint_dynamics.py`, first half fitted, second half predicted): delays
0-4 ms, several time constants at the 0.43 ms floor, held-out RMSE 0.007-0.078
rad, dominated by load droop a pure lag cannot represent.

**Re-measure if:** a robot log is supplied, or the PD path or physics step
changes.

Calibration procedure (not performed; no robot data exists): record each joint
with `step_probe`/`sine_probe` residuals of a few hundredths of a radian while
the robot stands supported, in the shared log with motor and, where available,
link encoders; fit on alternate segments and accept only if held-out RMSE stays
within the fit RMSE; write `{"source": "robot", "joints": {...}}`, hash it and pass
the hash to `load_calibration`.

## S1 implementation record

**Current:** `locomanip_study` registers an independent original-scale control;
production and frozen safety are untouched. `freeze_locomanip_references.py`
records exact checkpoint/configuration hashes, dependency revisions and external
controller provenance without constructing an environment. Provenance comparison
does not count as a successful runner load.

**Re-measure if:** references, installed controller inputs or dependencies change.

**History:**
- 2026-10-26 — `uv run pytest -q tests/test_locomanip_study.py`: 1 passed;
  original authority and both production robot registrations checked. Targeted
  Ruff passed. Plan commit: `a707545`. Existing tmux `mc_mjlab_training:1.1` is
  TensorBoard, not training; `mc_mjlab_training:2.1` is an idle shell. No campaign
  job launched. Next: freeze inventory, then validate actual runner loads with
  separate-arm S2 rollouts.
