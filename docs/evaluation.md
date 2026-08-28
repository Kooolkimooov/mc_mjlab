# Evaluation

How to find out whether a residual policy actually beat mc_rtc, and the sampling
traps that make it easy to get a confident wrong answer. Scripts in `scripts/`.

## Why the training curves cannot answer it

`Episode_Reward/*` comes from a *stochastic* policy still carrying its
exploration noise, which on this task is a disturbance in its own right — it goes
through the real PD gains onto the joints doing the balancing. The sums also
correlate with episode length at r = +0.98, so they move when the robot survives
longer rather than when it tracks better.

## audit_rewards.py

Run the reward audit before changing an objective or spending a training run:

```sh
uv run python scripts/audit_rewards.py
uv run python scripts/audit_rewards.py --checkpoint logs/.../model_180.pt
```

It reports each term before weighting, its effective live weight, weighted
per-second rate, integrated window contribution, nonzero fraction, quantiles,
and exact output shape. A checkpoint run splits environments between policy zero
and the trained actor. This catches an inert or mis-scaled objective; it does not
replace `compare_to_baseline.py`, because reward samples are not fixed-episode
qualification outcomes and the two arms do not occupy identical states.

The JSON report embeds the effective reward contract and records
`zmp_grounded` and `recovery_active` beside conditional rewards. Read the full
unit and zero-weight-call semantics in
[reward-shaping.md](reward-shaping.md#reward_audit).

## compare_to_baseline.py

Both arms run in the same env, with the same pushes, terminations and reward
terms, and the policy is evaluated deterministically (`MLPModel.forward` returns
the distribution mean), so a deficit is the learned mean being worse rather than
the dither.

```sh
uv run python scripts/compare_to_baseline.py --checkpoint logs/.../model_499.pt
```

Checkpoints from the `Position-Velocity` task need its additional action and
observation layout reconstructed explicitly:

```sh
uv run python scripts/compare_to_baseline.py \
  --checkpoint logs/.../model_187.pt --walking-reference
```

Reward-shape screens must be scored with their own width. Pass
`--recovery-dcm-std` when it differs from the repository default; this changes
only the reported recovery term, not the policy or simulated behavior.

Two things it does that a naive A/B does not, both learned by getting them wrong
first.

### Both arms at once

The envs are split in half and stepped together in one loop, so the two arms
share the wall-clock window, the machine load and the worker pool. Running one
arm and then the other would put the second on controllers already worn by the
first, and would let anything that drifts during the run land on only one of them.

The cost is that the pairing is no longer within-env: an env belongs to one arm
for the whole run, so `encoder_bias` (a per-env startup constant that moves
survival on its own) differs between the arms rather than cancelling. With envs
split evenly that is a wash in expectation, and the run is half as long.

### Fixed episodes per env, not "everything that finished"

A fall recycles in a few seconds while a survivor runs to the cap, so counting
every episode completed inside a wall-clock budget samples them proportional to
1/duration. **That inflates a failure rate badly — a true 10% can read 67% — and
it inflates it by a different factor for each arm**, which biases the very
difference being measured.

Taking the first K episodes of every env is unbiased because the selection never
looks at how long an episode lasted. The raw counts are printed alongside so the
size of that correction stays visible.

K is counted over **every env the arm owns**, not over the envs that appear in
the results. Seeding it from completed episodes instead silently drops any env
still inside its first episode at the deadline — which is to say the survivors.
That is a selection on duration again, the exact bias this exists to avoid. An
arm with such an env reports K=0 and says so rather than producing a
clean-looking number computed off the fallers alone.

### Resetting without corrupting the observation history

`ManagerBasedRlEnv.reset()` ends with
`observation_manager.compute(update_history=True)`, and `compute_group` appends
to every term's `CircularBuffer` for *all* envs, not just `env_ids` — the
buffer's write pointer is shared across the batch, so it cannot be otherwise.
`step()` has already appended this step's frame, so calling `reset()` in the loop
gives every `history_length=5` term two frames on any step where at least one env
finished, which is most of them. The window then spans half the time it should
and carries near-duplicate frames.

That is not symmetric across the arms: the baseline applies a zero action and
reads no observations at all, so the distortion lands only on the policy — it
biases the difference the script exists to measure. Hence `_reset_done`, which
makes the three calls `step()`'s own `auto_reset` path makes, minus the recompute.

The residual cost, stated rather than hidden: `get_observations()` returns the
cached buffer, so the first action of a new episode is computed from the dead
episode's last observation. One step per episode, against every step of every
episode the other way. From the next step on the histories are exactly what
training produces — the reset zeroed those rows, so the next append backfills all
five slots with the fresh frame.

### Per-step rates beat survival

**Divide episode length out and read that first.** Survival to the cap is the
headline number but it is a badly underpowered one: at ~20% it needs roughly
n = 150 per arm to resolve a 5pp difference. The per-step tracking rates resolved
a 10% deficit at n = 50 with p < 0.001, on the same data where survival came back
p = 0.70.

Measured the hard way on 2026-08-14. Three comparisons of the same training run:

| checkpoint | n/arm | survival verdict | per-step verdict |
| --- | --- | --- | --- |
| `model_1499` (previous run) | 48 | -2.1pp, p = 0.80 | not computed |
| `model_1050` | 18/24 | +0.0pp, p = 1.00 | not computed |
| `model_3050` | 48/54 | +3.2pp, p = 0.70 | **-10%, p < 0.001** |

The first two said "no detectable difference" and were read as parity. The third,
with barely more data, showed the policy was significantly *worse* all along —
the survival statistic simply could not see it. An underpowered null is not
evidence of equivalence, and this task's survival rate is underpowered at any
sample size a 20-minute run can reach.

The episode-sum rewards do not substitute: they correlate with episode length at
r = +0.98, so a policy with shorter episodes shows lower sums whether or not it
tracks worse. Only the length-normalised rate separates the two.

### Reading the output

`auto_reset = False` is essential: `step()` otherwise resets terminated envs *in
place* before returning, zeroing both `episode_length_buf` and the reward
manager's `_episode_sums`, so the episode being measured is erased before the
caller sees `terminated`.

Reward sums correlate strongly with episode length on this task, so a reward
delta that tracks the length delta is not independent evidence.

Quartiles are reported because the mean alone hides the shape: episode length is
strongly bimodal (an early fall against a full survival), so a mean that moves
can mean either "falls got later" or "more episodes reached the cap". `sem` is
what two runs have to differ by to mean anything. Quartiles use the "inclusive"
linear-interpolation convention numpy/pandas use, so they match anything the data
gets pasted into.

`--drop-obs` exists for checkpoints that predate an observation term: the actor's
first layer and its `obs_normalizer` are both sized by the concatenated width, so
adding a term retires every earlier checkpoint with a `size mismatch` at load.
Removing one from the *middle* of a group would silently shift later terms into
the wrong column rather than fail, so it is only sound for terms appended at the
end.

## Never score the last checkpoint

**Score the best-metric checkpoint, and confirm the choice with a second read.**
The last checkpoint is not the best one and can be actively broken: `zeroinit-4ev`
peaked on `Episode_Metrics/zmp_error` at iteration 1320, decayed for 1600
iterations, and then diverged outright in its last 50, so `model_2999` was corrupt
while `model_1000` was the best result this project has produced. Reading only the
final checkpoint would have reported the opposite of the truth in both directions.

The cheap procedure: smooth `Episode_Metrics/zmp_error` over ~60 iterations, take
the argmin, and compare that checkpoint plus one much later. Two reads bracket the
trajectory — a single one cannot distinguish "still improving" from "peaked and
decaying", and those call for opposite decisions about run length.

## qualify_checkpoints.py

The promotion path scores **every saved validation checkpoint**, supplied as
individual paths, a run directory, or globs. It writes raw paired episodes to
`qualification.csv` and configuration, clustered statistics, Holm-adjusted
p-values, gate failures, and the selected checkpoint to `qualification.json`.

```sh
uv run python scripts/qualify_checkpoints.py logs/.../run_dir \
  --seed 42 --seed 43 --seed 44 --out-dir logs/qualification/run_name
```

The default scenarios are nominal walking, the historical velocity kick, a
force-based finite impulse, and a held-out robust impulse with stage-one
friction/gain/strength and delay perturbations. Episode counts are fixed per env
and arm; wall time never decides which episodes enter the result. The robust
scenario is invalid for promotion if more than 5% of its baseline episodes fail
before the first disturbance.

### qualification_strata.csv

The qualifier also writes `qualification_strata.csv` and
`qualification_joints.csv`. They are additional diagnostics; the established
promotion gates still read the original episode aggregates.

Every episode step belongs to exactly one stratum:

- `startup/none` covers episode age through 10 s before a disturbance.
- `recovery/{forward,backward,left,right}` covers the live `recovery_dcm`
  window after a recorded push. Recovery takes precedence on the first pushed
  step, so that step cannot leak into startup or sustained exposure.
- `sustained/none` is every remaining step, including nominal walking after the
  startup transient and the intervals between recovery windows.

Directions use the push velocity in the controller base frame. The largest
absolute horizontal component selects sagittal versus lateral; its sign selects
forward/backward or left/right. The raw rows remain one episode, environment,
pair, arm, and stratum each. JSON differences are paired on those episode keys
and then clustered by environment, or by seed when multiple seeds are present.
Timesteps are never treated as independent samples.

The stratum CSV reports grounded fraction, grounded-conditional DCM and ZMP
errors, controller-command CoM-velocity error, foot slip, recovery gate duty,
detector score, and actuator-demand ratio. A stratum that contains the final
step also carries its exact termination cause; the other strata in that episode
carry zero, which attributes rather than duplicates failures.

The joint CSV reports each residual joint's physical authority scale, requested
and executed normalized RMS, executed physical RMS, active fraction,
near-policy-bound fraction, feasibility-projection fraction, and actuator effort
ratio. Physical RMS and authority are radians in position mode and newton-metres
in torque mode. Per-joint effort is sampled at policy-step endpoints; the scalar
`max_effort_ratio` has substep coverage and remains the hard safety gate.

Read a zero `active_fraction` together with gate duty: it means the joint
delivered no residual, while a nonzero request with a zero gate means the actor
asked but recovery authority correctly withheld it. `com_velocity_error` is the
controller-command tracking diagnostic: lower means realized CoM motion stayed
closer to what mc_rtc commanded in that regime.

Use the zero-residual component gate before adding or widening a robustness
term:

```sh
uv run python scripts/calibrate_robustness_profile.py
```

It runs each component and their combination through the 12-second gait
transition, reports survival and worker failures, and exits nonzero below 95%
survival.

Each env runs a crossover: baseline and policy alternate, with starting arm
balanced across env ids. Disturbance timing, planar direction, equivalent delta
velocity, duration, and application height are deterministic functions of seed,
scenario, env, pair, and occurrence. Both arms of a pair therefore see the same
schedule and the same controller instance and startup encoder bias. Statistics
cluster pairs within env; with multiple evaluation seeds they aggregate envs
within seed and place the confidence interval across seeds.

Safety and nominal gates run before ranking. Passing checkpoints are ranked
lexicographically by recovery, hazard, then residual use. The test set is not an
input to this command: run it only after `selected` is fixed, and only for that
checkpoint.

`scripts/generate_clean_checkpoint.py` creates a provenance-bearing untrained
checkpoint in the current policy format for smoke tests. It does not translate
old Gaussian checkpoints. Those remain valid only in a worktree containing their
original actor, observation layout, and controller inputs; generating new
validation checkpoints from the current tree is the clean migration path.

## source_drift

**Current:** defining-file digests live in the manifest's audit `record` and in
neither enforced contract. `validate_effective_training_manifest` strips
`source_sha256` from both the training contract and the policy interface, then
re-digests both sides rather than comparing the stored digest, and prints the
drifted audit paths when the contract still matches.

**Why.** On 2026-08-27 a stage-0 re-qualification of a checkpoint trained the
same afternoon died with
`Checkpoint policy interface differs from the active run:
observations.groups.actor.terms.controller_ref_pos.callable.source_sha256, ...`.
Nothing about the actor's interface had changed. Unrelated additions to
`tasks/mdp.py` — a new metric and a new event class — changed the digest of the
file that happens to define six observation terms, and that stranded the
checkpoint. Source identity is evidence about semantics; it is not semantics, and
making it the enforced contract means any edit to a shared module retires every
checkpoint that module touches.

The enforced contract still catches what matters: a renamed or relocated
callable changes its qualified name, and term ordering, dimensions, group
settings and effective parameters are all still compared.

**Re-digesting both sides is also a strengthening.** Validation previously
compared the `*_sha256` fields stored in the checkpoint. Those are derived
values, so a stale or forged digest could mask a real difference or invent one.
The payload now decides.

**A field added after a checkpoint was written is not an interface change.**
On 2026-08-28 the action term gained `controller_scalars`, and every checkpoint
trained before it became unqualifiable: the saved manifest cannot contain a key
that did not exist. `validate_effective_training_manifest` now drops keys the
active contract *gained* before comparing, and names them. Removed keys and
changed values still fail, and observation ordering, dimensions and effective
parameters are compared separately, so an addition that really does move the
actor interface still shows up there.

**Re-measure if:** the manifest schema changes or a new field is added that
should be enforced rather than recorded.

**History:**
- 2026-08-27 — moved source hashes to audit-only after they stranded a
  same-day checkpoint; the branch review had flagged this as the most dangerous
  defect on the branch and `leo-mjlab-review.md` had already recorded the
  decision to exclude them.

## invalidated_scenarios

**Current:** a scenario whose baseline or policy arm lost a controller worker is
removed from every substantive gate, not just flagged. `promotion()` reports the
invalidation, replaces the recovery-DCM verdict with
`recovery DCM unreadable, scenario invalidated`, computes the hazard ratio over
the surviving scenarios and says how many those were, and records the excluded
names under `invalidated_scenarios`.

This cannot promote anything: an invalidation already appends its own reason, so
`eligible` is `False` before any of this runs. The change is entirely about what
the report *claims*.

**Why.** The 2026-08-27 `model_140` verdict read
`finite_impulse: controller worker failure invalidated the run` and
`finite impulse: recovery DCM improvement is below 5%` side by side. The second
is computed from the first's discarded episodes. Read quickly it says the policy
failed the recovery gate, when the truth is that the run has no recovery
measurement at all — the same confusion between a measurement verdict and a
policy verdict that `clusters_for_confidence` above exists to prevent.

**The failures are wedges, not crashes.** All three in that run reported
`exit code None`, which is the pool's timeout killing an unresponsive worker
rather than a signal. That is the documented mc_rtc mode: a controller whose MPC
has collapsed can hang permanently inside `reset()`. Harder impulse bands reach
that state more often, so a qualification run over the matched distribution is
more exposed to it than one over `[0.10, 0.25]`. Two of four scenarios were lost
at 16 environments and 6 workers.

**Re-measure if:** the pool timeout, worker count, or impulse distribution
changes.

**History:**
- 2026-08-27 — separated invalidation from verdict after a worker wedge produced
  a recovery-gate reason on a scenario that had no valid episodes.

## clusters_for_confidence

**`--num-envs` defaults to 16, not 8.** The paired interval has
`seeds x environments` clusters, so the old 8-environment single-seed default
gave 8 — a third of the 23 that the best measured recovery effect needs. Sixteen
is also the documented ceiling for running beside a training job.


**Current:** the paired interval is a Student-t interval over
`(seed, environment)` clusters, and every summary carries its `clusters` count.
When the recovery-DCM improvement clears its 5% magnitude bar but its interval
still touches zero, `promotion()` names the cluster count that would resolve it
instead of reporting an undifferentiated gate failure.

**Two defects this replaced.** A single cluster produced `sem = NaN`, so
`ci_high >= 0.0` evaluated False and the confidence half of the recovery gate
silently passed; the standard error is now infinite, which fails closed. And the
paired statistic collapsed to seed means whenever more than one seed ran, so the
achievement mode's mandatory two seeds turned 32 environment clusters into 2 —
where a 95% normal interval has roughly 24% coverage, because the correct
critical value at one degree of freedom is `12.71`, not `1.96`.

**It did not promote anything retroactively.** Re-scored with the t interval,
all three checkpoints that the 2026-08-25 paired qualification rejected still
fail, and by wider margins:

| checkpoint | mean, m | relative | normal `ci_high` | t `ci_high` | clusters needed |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard 180 | -0.01010 | -7.70% | +0.00119 | +0.00218 | 23 |
| ankle 187 | -0.00675 | -5.08% | +0.00380 | +0.00472 | 42 |
| ankle 140 | -0.00639 | -4.79% | +0.00752 | +0.00874 | 79 |

At one seed the change is strictly stricter: 16 clusters move from `1.96` to
`t(15) = 2.13`, an 8.7% wider interval. Those runs used 16 environments, so the
best of them was **seven clusters short** of resolving its own effect. That is a
sampling verdict, not a policy verdict, and the old report could not tell them
apart.

**Re-measure if:** the recovery-DCM effect size, per-environment variance,
episode pairing, or the 5% magnitude bar changes.

**History:**
- 2026-08-27 — replaced the normal interval and seed-mean collapse after the
  branch review found three rejections whose reported cause was confidence.

## run_improvement_screens.py

**Current:** the resumable launcher contains only the supported standard and
ankle tasks. It retains the 188-iteration, seed-42, 128-environment, 30-worker
comparison budget and TensorBoard logging. Completion state and one console log
per arm live under `logs/critique_screens/`.

Launch it inside tmux:

```sh
uv run python scripts/run_improvement_screens.py
```

Qualification accepts `--authority-set`, `--controller-history`,
`--proprio-history`, and `--recurrent` to reconstruct archived actor shapes.
`compare_to_baseline.py --walking-reference` reconstructs the velocity variant.
Historical variants therefore remain evaluable without keeping their task ids in
the default registry.

**Re-measure if:** screen budget, worker capacity, task registrations, or PPO
screen axes change.

**History:**
- 2026-08-27 — removed the eleven failed/non-policy arms after the original
  matrix completed; the launcher now compares only standard and ankle authority.
- 2026-08-24 — added after implementing the critique changes so interrupted
  multi-hour screens resume at arm boundaries instead of silently duplicating
  completed runs.

## Comparing a checkpoint whose config has moved on

Checkpoints die whenever the observation width changes, so scoring an older one
means reconstructing the cfg it trained under. A `git worktree` at the right commit
plus `PYTHONPATH=<worktree>/src:$PYTHONPATH` shadows the installed package without
touching the working tree or any live run. Two things bite:

New residual-balance checkpoints embed their complete base-controller provenance
under `infos/base_controller_provenance`: the project and user mc_rtc YAML, the
installed selected-controller YAML, and the PD gains, each with its source and
SHA-256 digest. Readable copies also live under the run's
`base_controller_config/`. Loading refuses a mismatch, because evaluating the
same residual against a changed base is a different policy. The manual procedure
below remains necessary for checkpoints created before this runner existed.

- **Prepend to `PYTHONPATH`, never replace it.** The mc_rtc bindings arrive on it
  from the sourced workspace; clobbering it fails at `import mc_rbdyn`.
- **Copy `etc/mc_rtc.yaml` into the worktree.** It is untracked, so the worktree
  gets the *committed* `MainRobot`, and a different robot means a different joint
  count and a `size mismatch` that looks exactly like a stale-checkpoint error.

## validate_dcm_objective.py

The gate the DCM objective has to pass **before** a training run, not after:
does the reward score a walking robot and a standing one alike, and does a push
still collapse it? No checkpoint is involved — every arm runs a zero residual.

```sh
uv run python scripts/validate_dcm_objective.py                     # ~1 min/regime
uv run python scripts/validate_dcm_objective.py --residual-level 0  # skip arm 3
```

Three arms, identical but for the controller and the residual:

| arm | config | residual |
| --- | --- | --- |
| `walking` | `etc/mc_rtc.yaml` as it stands | zero |
| `standing` | the same, with `Enabled: Posture` written to a temp copy | zero |
| `walking+residual` | as `walking` | full-scale, alternating signs |

Nothing in the repo is touched to get the standing arm: `posture_config` rewrites
the one `Enabled:` line into `--out-dir`, and `_make_env_cfg(mc_rtc_yaml=...)`
takes it from there.

What each block of the output answers:

- **the quantile table** — the nominal error distribution, which is what sizes
  `DCM_STD`; the printed `std = q / sqrt(-log s)` line inverts the kernel for a
  target score `s` at quantile `q`.
- **corrected against legacy** — the same samples scored with and without the
  commanded offset, so the correction's effect is visible rather than assumed.
- **standing edge over walking** — the defect the correction exists for. Read it
  at the `DCM_STD` actually configured, not at the friendliest column.
- **error by speed deficit** — the decisive one. It bins the *walking* arm by
  `measured - commanded` CoM speed: a term that pays a robot for resisting its
  own gait shows its lowest error in the "slower than commanded" band.
- **error by time since a push** — the disturbance response, and the profile that
  sizes `RECOVERY_WINDOW_S`.
- **residual authority over the target** — the residual must not be able to earn
  reward by moving the *reference* instead of the state.

## probe_residual_authority.py

Answers a different question, and the one to ask first when a reward looks blind:
**can the action reach the objective at all?**

The residual_balance reward proved to be blind to the policy — with episode
length divided out, the per-step `zmp_tracking` rate is the same for a zero
residual and for every trained checkpoint (0.01183 / 0.01187 / 0.01196, SE
0.00015). Two things can cause that: the reward is badly shaped, or the action
cannot move the objective. This settles the second before anyone spends time on
the first, because at `residual_scale = 0.01` rad (~0.57 degrees of joint offset)
it is entirely possible the residual simply cannot shift the centre of pressure.

The probe drives a **constant** residual instead of a policy and measures
`mdp.zmp_error` — the distance in metres between the measured centre of pressure
and the one mc_rtc planned. A constant offset is the bluntest possible input: if
a full-scale one does not move that distance, nothing a policy does will either.

```sh
uv run python scripts/probe_residual_authority.py --level 0    --minutes 10
uv run python scripts/probe_residual_authority.py --level 1.0  --pattern alternating
uv run python scripts/probe_residual_authority.py --level 1.0  --pattern noise
```

Compare runs on `zmp_error mean`. Authority is adequate if a full-scale constant
residual shifts it by >= 0.007 m (20% of the ~0.036 m operating error).

### Binning by time since a push

The probe also bins the error by time since the last push, which gives the
recovery profile — how long after a disturbance the tracking error stays
elevated. That is the window a disturbance-gated reward should pay on, so the
sweep sizes `RECOVERY_WINDOW_S` as well as justifying it.

Getting that binning right took two corrections, both of which had inflated the
early bins — the ones that size the window:

- **Detect the push from the term's own record, not the interval timer.**
  `EventManager` re-samples the countdown whenever the term *fires*, including
  the ticks `push_and_record` suppresses during `PUSH_WARMUP_S`. A timer-based
  detector therefore counts pushes that never landed, and with a 10 s warm-up
  that is the first ~2 ticks of every episode. `mdp.steps_since_push` reads
  `last_push_step`, which is only set when a push actually happens.
- **Do not treat an episode reset as a push.** Setting the age to zero on reset
  binned the following 8 s of a *fresh* episode as post-push recovery — and those
  8 s contain the ~4 s posture settle, which carries the largest ZMP errors in
  the run. `steps_since_push` reports a huge age for a push older than the
  current episode, so reset envs drop out of the filter instead.

The gate is `age >= 1`, matching `recovery_tracking`'s: at age 0 the push has
been applied but no physics has run on it yet, so that sample predates its
effect. Matching the gate keeps the bins aligned with the steps the reward is
actually paid on.

See `RECOVERY_TRACKING_WEIGHT` in [reward-shaping.md](reward-shaping.md) for the
profile itself and the caveat that the recorded numbers predate these fixes.

## calibrate_recovery_detector.py

Fits the deployable residual-authority detector from dedicated zero-disturbance
and fixed-energy disturbed cohorts, with even environments used for fitting and
odd environments held out. A failed held-out gate still writes a temporary JSON
when fitting succeeds, but it must not replace `etc/recovery_detector.json`.

```sh
uv run python scripts/calibrate_recovery_detector.py \
  --output /tmp/recovery_detector.json --trace-out /tmp/recovery_trace.pt
uv run python scripts/calibrate_recovery_detector.py \
  --output /tmp/recovery_detector.json --trace-in /tmp/recovery_trace.pt
uv run python scripts/verify_live_recovery_detector.py
```

The trace replay is the fast path for filter-only changes. Acceptance requires
nominal mean authority at most 5%, at least 80% recovery samples above 5%
authority in the first 2 s, and fewer than 1% above 5% after 2 s. The live check
then drives nonzero action to assert exact inactive residual zeroing.
