# Residual locomanip implementation and training campaign

## Progress ledger

Implementation status and experimental outcome are independent. All timestamps
use Asia/Tokyo (UTC+09:00). Raw traces and checkpoints stay outside Git.

| Stage/substep ID | Status | Implementation/validation evidence | Experiment outcome | Artifacts | Next action/blocker | Last updated |
| --- | --- | --- | --- | --- | --- | --- |
| S0 plan | complete | Complete approved plan transcribed before source changes; commit containing this entry | not measured | this document | Freeze references | 2026-10-26 14:20 +09:00 |
| S1 references | active | Inventory frozen at `f577463`; `original` loads through the provenance-checking runner into the study control | `original` fails 700 kg nominal (`collapsed`, 47.6-47.7 s) | `logs/locomanip-study/references.json`, `phase-smoke/` | `no_magnitude` and `arms_2x` loads run in the S4 nominal probe | 2026-10-26 15:20 +09:00 |
| S2 qualification | complete | Mass dtype fix `bb98c3b`; `compare_study_runs.py`; six zero sources over batch 6/reversed batch 3 with recycled partial resets | zero vs zero: no outcome disagreement; spreads 0.0135 m, 0.04 s | `zero-smoke/`, `zero-reordered/`, `zero-plain-timed/`, `zero-repeatability.json` | Validation/held-out cohorts once candidates exist | 2026-10-26 15:20 +09:00 |
| S3 PD telemetry | complete | `pd_telemetry` `f963cb2`: motor table and chain trajectory identical, stepped-only samples, partial resets (7 tests) | live: no outcome disagreement, spread 0.017 m; +3% step time; rate traces sum exactly | `zero-telemetry/`, `zero-telemetry-rate/` | Used by S4 | 2026-10-26 15:20 +09:00 |
| S4 phase diagnosis | active | `probe_locomanip_phases.py` `8184e6e`: constant arms, raw-phase windows (3 tests); 700 kg smoke segmented 6/6 episodes | pending | `phase-smoke/` | Nominal probe: 16 arms x 600-900 kg x fresh/recycled | 2026-10-26 15:20 +09:00 |
| S5 focused exploration | planned | none | pending | none | Measured groups from S4; typed study tasks | 2026-10-26 14:20 +09:00 |
| S6 observation history | planned | none | pending | none | Isolated 10/20-frame variants | 2026-10-26 14:20 +09:00 |
| S7 objective audit | planned | none | pending | none | Recorded matched trajectories and ranking defect gate | 2026-10-26 14:20 +09:00 |
| S8 robustness/transitions | planned | none | pending | none | Isolated scenarios, then justified training branches | 2026-10-26 14:20 +09:00 |
| S9 campaign | planned | No new training launched | pending | none | S2 qualification and family smoke tests first | 2026-10-26 14:20 +09:00 |
| S10 identification | planned | No robot calibration data supplied | not measured | none | Shared log, synthetic/simulator validation, opt-in models | 2026-10-26 14:20 +09:00 |

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

**Current:** No campaign experiments launched. Existing measurements remain in
[locomanip.md](locomanip.md); they motivate the campaign, not its qualification.

**Re-measure if:** any reference or evaluator semantics change.

**History:**
- 2026-10-26 14:20 +09:00 — source inspection only. Initial tmux inventory was
  inaccessible inside the sandbox; inventory requires local socket access before
  scheduling jobs. Local `pyproject.toml` dependency override is pre-existing.

## Amendments

None. Original decisions are preserved above.

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
