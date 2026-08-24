# mc_mjlab residual-balance: full comparison record

Generated 2026-08-19 (updated through `obj-recovery010-full`, 2026-08-24). Every number below is
derived from the per-episode dump CSVs written by
`scripts/compare_to_baseline.py --dump`, re-analysed with a common pipeline, not
transcribed from the script's own printout.

**Untracked on purpose.** This is a data record, not documentation; the
reasoning that survives belongs in `docs/`.

## Method

- **Paired design.** Each comparison runs both arms simultaneously on one env
  set, split in half: the policy drives one half, a zero residual the other.
  Same physics, same wall clock, independent push draws per env.
- **Fixed K per env.** For each env the first `K` finished episodes are taken,
  where `K = min` over all envs, then episodes are pooled. This removes the
  1/duration bias that inflates failure rates when counting episodes inside a
  wall-clock budget.
- **Per-step rates.** Every reward figure is the episode's reward sum divided by
  its length in steps. Episode sums correlate with episode length at r = +0.98
  on this task and are not independent evidence.
- **`tracking sum`** is the per-step sum of that configuration's own tracking
  terms, excluding `residual_magnitude` and `residual_rate`, which are zero for
  the baseline by construction and so are not a fair comparator.
- **Hazard** is falls per second of exposure past a 10 s warm-up, with episodes
  reaching the 4500-step cap censored. It is cap-independent where survival-% is
  not.
- **Summary endpoint p-values** are two-sided Welch tests on each environment's
  mean across its K retained episodes; `envs/arm`, not episode rows, is the
  inferential sample size. No multiple-comparison correction is applied
  anywhere.
- **The raw outputs live in `logs/comparisons/`**, one `.csv` of per-episode
  rows and one `.log` of the script's own printout per checkpoint, named
  `<run directory>_<model>`. `compare_to_baseline.py` writes them there by
  default; before 2026-08-20 they were left in a session scratch directory, so
  the `cmp_*` and `h_*` files there are the earlier runs recovered under their
  original names (`h_*` are the 28-envs/arm re-measurements the summary table is
  built from). Probe outputs sit beside them in `logs/probes/`.
- **2026-08-19 is a boundary.** Everything through `pruned-nogate` predates
  three changes made that day: the DCM objective became command-relative,
  `fell_over` and `collapsed` became mutually exclusive labels (their union, and
  so hazard and survival, are unchanged), and controller-worker deaths became
  truncations rather than penalised terminations. So `dcm_*` magnitudes and
  `fell_over`/`collapsed` shares do not compare across that line; hazard,
  survival and episode length do. `dcm-cmd-ref` is the first post-boundary run
  once its paired outputs arrive.

- **Term sets differ between reward eras.** Old: `zmp_tracking`,
  `com_velocity_tracking`, `recovery_tracking`. New: `dcm_stability`,
  `recovery_dcm`, `zmp_tracking`, `com_velocity_tracking`, `angular_momentum`,
  `foot_slip`. Pruned: the same minus the two agreement terms, plus
  `torque_margin` and `upright`. Magnitudes are therefore **not** comparable
  across eras; signs and per-run deltas are.

## Summary

Post-fix rows are trimmed to a common **K = 4 episodes per env**. The historical
pre-orientation row, the `m998` continuation and `head-authority20` `m960`
reached only K = 3 (the policy completed fewer episodes within the same
wall-clock budget), so they are labelled separately, and `head-authority20`
`m999` (‡) reached only K = 2: at 128 envs on one host each environment finishes
fewer episodes in the budget. Hazard ratio below 1 means the policy falls less
often per second than its own zero-residual arm.

The 2026-08-21--24 rows use 32 environments per arm. `rg-fixedstd010`,
`obj-recovery4`, and full-run models 180/340 retained K = 3; the other new rows
retained K = 4. Their summary p-values still use one mean per environment.

Episode rows are supporting data, not independent statistical evidence: episodes
in an environment share its startup encoder bias. The tracking p-value uses one
K-episode mean per environment; the per-run counts remain in the detailed
sections.

| run                               | train date | ckpt       | tracking sum | hazard   | fell_over  | collapsed  | survival   | ep len     |
| --------------------------------- | ---------- | ---------- | ------------ | -------- | ---------- | ---------- | ---------- | ---------- |
| std-floor (pre-orientation-fix) † | 2026-08-13 | `m1499`    | -8.7%        | **0.76** | **-1.0%**  | **-7.3%**  | **+6.8%**  | **+2.8 s** |
| std-floor                         | 2026-08-14 | `m1050`    | -7.3%        | 1.33     | **-1.8%**  | +6.2%      | -8.0%      | -7.3 s     |
| std-floor                         | 2026-08-14 | `m3050`    | -1.4%        | **0.84** | +3.6%      | **-2.7%**  | **+3.6%**  | **+6.0 s** |
| scale03-g997                      | 2026-08-15 | `m1150`    | -20.6%       | **0.82** | +0.0%      | **-13.4%** | **+10.7%** | **+2.6 s** |
| scale01-g997                      | 2026-08-17 | `m950`     | -6.6%        | **0.80** | **-8.9%**  | +0.9%      | **+7.1%**  | **+5.7 s** |
| zeroinit-4ev                      | 2026-08-17 | `m1000`    | -3.8%        | **0.96** | +7.1%      | **-6.2%**  | **+2.7%**  | **+0.5 s** |
| zeroinit-4ev                      | 2026-08-17 | `m2900`    | -6.1%        | **0.85** | +4.5%      | **-14.3%** | **+8.0%**  | **+2.6 s** |
| dcm-obs                           | 2026-08-18 | `m900`     | **+2.4%**    | 1.00     | **-0.9%**  | +0.0%      | **+0.9%**  | -0.5 s     |
| dcm-obs                           | 2026-08-18 | `m1850`    | **+6.1%**    | **0.95** | +3.6%      | **-4.5%**  | **+0.9%**  | **+2.5 s** |
| dcm-obs-gate                      | 2026-08-18 | `m250`     | **+1.1%**    | **0.88** | **-6.2%**  | +1.8%      | **+4.5%**  | **+3.3 s** |
| dcm-obs-gate                      | 2026-08-18 | `m1500`    | -0.5%        | **0.78** | **-1.8%**  | **-12.5%** | **+8.9%**  | **+5.0 s** |
| gate-torque-phase                 | 2026-08-18 | `m250`     | -4.2%        | 1.23     | **-6.2%**  | +15.2%     | -7.1%      | -4.9 s     |
| gate-torque-phase                 | 2026-08-18 | `m1499`    | -0.4%        | **0.97** | **-6.2%**  | +8.9%      | +0.0%      | **+2.0 s** |
| **pruned-nogate**                 | 2026-08-19 | **`m250`** | **+3.2%**    | **0.81** | **-10.7%** | **-1.8%**  | **+8.9%**  | **+4.2 s** |
| pruned-nogate                     | 2026-08-19 | `m499`     | **+4.5%**    | 1.01     | +2.7%      | **-0.9%**  | **+0.9%**  | -0.8 s     |
| **pruned-nogate +500**            | 2026-08-20 | **`m998`** | **+0.7%**    | **0.95** | **-2.1%**  | **-1.0%**  | **+1.6%**  | **+1.5 s** |
| head-authority20                  | 2026-08-20 | `m500`     | -34.7%       | 1.37     | **-3.1%**  | +5.2%      | -8.3%      | -9.8 s     |
| head-authority20                  | 2026-08-20 | `m960`     | -12.8%       | 1.01     | **-7.3%**  | +3.1%      | **+4.2%**  | -3.1 s     |
| head-authority20 ‡                | 2026-08-20 | `m999`     | -27.3%       | 1.27     | **-1.6%**  | +8.6%      | -7.0%      | -6.4 s     |
| rg-ref01                          | 2026-08-21 | `m187`     | -1.2%        | **0.98** | +3.9%      | **-5.5%**  | +0.0%      | **+0.1 s** |
| rg-entropy0                       | 2026-08-21 | `m187`     | **+1.0%**    | **0.99** | +7.8%      | **-11.7%** | **+0.8%**  | -2.0 s     |
| rg-fixedstd010                    | 2026-08-21 | `m187`     | **+0.7%**    | **0.94** | +3.1%      | **-7.3%**  | **+4.2%**  | -0.1 s     |
| rg-mag03                          | 2026-08-21 | `m187`     | -0.3%        | 1.10     | **-3.9%**  | +6.2%      | -3.1%      | -2.9 s     |
| recovery-only-005                | 2026-08-21 | `m187`     | -2.6%        | **0.95** | +7.3%      | **-8.3%**  | **+1.0%**  | **+1.5 s** |
| recovery-only-010                | 2026-08-22 | `m187`     | **+1.3%**    | **0.84** | **-8.6%**  | +1.6%      | **+6.2%**  | **+3.3 s** |
| recovery010-full                 | 2026-08-24 | `m180`     | -2.6%        | 1.03     | **-1.0%**  | +2.1%      | **+1.0%**  | -0.4 s     |
| recovery010-full                 | 2026-08-24 | `m340`     | **+2.7%**    | **0.69** | +1.0%      | **-14.6%** | **+15.6%** | **+7.9 s** |
| recovery010-full                 | 2026-08-24 | `m499`     | -1.0%        | 1.16     | +8.6%      | +0.0%      | -7.0%      | -1.5 s     |

**Bold** marks a value in the policy's favour: a higher tracking sum, a hazard
ratio below 1, fewer falls or collapses than its own baseline, more survivals,
longer episodes. It marks direction only — for whether a cell is distinguishable
from no effect, read the p-value table below.

A **bold run and checkpoint** means every one of the six endpoints points the
policy's way — `pruned-nogate` `m250` and `pruned-nogate +500` `m998` are the
only two such rows in the record, and they are marked in the p-value table too
so the pair reads together. Neither sweep is significant: `m250`'s best endpoint
is p = 0.052 and `m998`'s is p = 0.60. A clean sweep is a run worth repeating,
not a result.

### Summary endpoint p-values

Every value below is a two-sided, environment-clustered Welch p-value: each
environment contributes one mean across its retained K episodes. `hazard` uses
each environment's adverse-ending rate per second after warm-up; `fell`,
`collapsed`, and `survival` use its episode share; `ep len` uses its mean
episode length. These are unadjusted for the many endpoints and runs, so they
describe evidence for one named comparison, not a family-wide discovery claim.
**Bold** marks the conventional, unadjusted p < 0.05 threshold. A **bold run and
checkpoint** repeats the summary table's clean sweep — every endpoint favoured
the policy there, whatever the p-values in that row say here.

| run                               | train date | ckpt       | tracking    | hazard      | fell    | collapsed   | survival | ep len      |
| --------------------------------- | ---------- | ---------- | ----------- | ----------- | ------- | ----------- | -------- | ----------- |
| std-floor (pre-orientation-fix) † | 2026-08-13 | `m1499`    | **1.2e-07** | 6.4e-01     | 8.3e-01 | 1.9e-01     | 8.1e-02  | 3.9e-01     |
| std-floor                         | 2026-08-14 | `m1050`    | **2.0e-05** | **2.9e-02** | 7.7e-01 | 4.0e-01     | 1.2e-01  | **3.1e-02** |
| std-floor                         | 2026-08-14 | `m3050`    | 3.5e-01     | 6.9e-02     | 5.0e-01 | 6.5e-01     | 4.5e-01  | **4.7e-02** |
| scale03-g997                      | 2026-08-15 | `m1150`    | **1.3e-32** | 7.1e-01     | 1.0e+00 | 6.5e-02     | 6.0e-02  | 4.8e-01     |
| scale01-g997                      | 2026-08-17 | `m950`     | **1.5e-03** | 2.9e-01     | 6.7e-02 | 9.0e-01     | 2.4e-01  | 1.8e-01     |
| zeroinit-4ev                      | 2026-08-17 | `m1000`    | **3.3e-02** | 7.9e-01     | 1.2e-01 | 3.3e-01     | 5.8e-01  | 8.8e-01     |
| zeroinit-4ev                      | 2026-08-17 | `m2900`    | **5.5e-04** | 3.1e-01     | 3.4e-01 | **2.0e-02** | 2.1e-01  | 4.8e-01     |
| dcm-obs                           | 2026-08-18 | `m900`     | 5.2e-01     | 9.3e-01     | 8.7e-01 | 1.0e+00     | 8.9e-01  | 9.0e-01     |
| dcm-obs                           | 2026-08-18 | `m1850`    | 5.3e-02     | 2.4e-01     | 4.7e-01 | 5.0e-01     | 8.6e-01  | 4.4e-01     |
| dcm-obs-gate                      | 2026-08-18 | `m250`     | 7.1e-01     | 2.4e-01     | 2.1e-01 | 7.8e-01     | 4.6e-01  | 4.0e-01     |
| dcm-obs-gate                      | 2026-08-18 | `m1500`    | 8.9e-01     | 2.0e-01     | 6.9e-01 | 6.6e-02     | 1.5e-01  | 2.5e-01     |
| gate-torque-phase                 | 2026-08-18 | `m250`     | 1.1e-01     | 1.1e-01     | 2.3e-01 | **4.2e-02** | 2.3e-01  | 2.0e-01     |
| gate-torque-phase                 | 2026-08-18 | `m1499`    | 8.8e-01     | 8.2e-01     | 2.5e-01 | 1.8e-01     | 1.0e+00  | 6.4e-01     |
| **pruned-nogate**                 | 2026-08-19 | **`m250`** | 3.2e-01     | 6.7e-02     | 5.2e-02 | 7.5e-01     | 6.1e-02  | 1.9e-01     |
| pruned-nogate                     | 2026-08-19 | `m499`     | 1.4e-01     | 6.7e-01     | 5.9e-01 | 8.9e-01     | 8.9e-01  | 8.2e-01     |
| **pruned-nogate +500**            | 2026-08-20 | **`m998`** | 7.3e-01     | 6.7e-01     | 6.1e-01 | 8.3e-01     | 7.0e-01  | 6.0e-01     |
| head-authority20                  | 2026-08-20 | `m500`     | **2.3e-54** | 3.4e-01     | 4.8e-01 | 4.7e-01     | 1.2e-01  | **1.2e-02** |
| head-authority20                  | 2026-08-20 | `m960`     | **6.5e-05** | 7.8e-01     | 5.2e-02 | 6.7e-01     | 5.3e-01  | 4.5e-01     |
| head-authority20 ‡                | 2026-08-20 | `m999`     | **7.3e-21** | 8.6e-02     | 7.3e-01 | 1.8e-01     | 2.0e-01  | 5.9e-02     |
| rg-ref01                          | 2026-08-21 | `m187`     | 6.2e-01     | 8.9e-01     | 5.0e-01 | 4.0e-01     | 1.0e+00  | 9.8e-01     |
| rg-entropy0                       | 2026-08-21 | `m187`     | 6.8e-01     | 7.8e-01     | 8.2e-02 | **2.3e-02** | 8.7e-01  | 5.3e-01     |
| rg-fixedstd010                    | 2026-08-21 | `m187`     | 7.7e-01     | 9.0e-01     | 5.8e-01 | 3.1e-01     | 5.1e-01  | 9.7e-01     |
| rg-mag03                          | 2026-08-21 | `m187`     | 8.9e-01     | 3.4e-01     | 4.3e-01 | 3.5e-01     | 5.4e-01  | 3.6e-01     |
| recovery-only-005                | 2026-08-21 | `m187`     | 3.1e-01     | 4.9e-01     | 1.7e-01 | 1.9e-01     | 8.6e-01  | 7.1e-01     |
| recovery-only-010                | 2026-08-22 | `m187`     | 5.3e-01     | 2.3e-01     | 6.3e-02 | 7.9e-01     | 2.2e-01  | 3.4e-01     |
| recovery010-full                 | 2026-08-24 | `m180`     | 2.7e-01     | 1.8e-01     | 8.3e-01 | 7.7e-01     | 8.8e-01  | 9.2e-01     |
| recovery010-full                 | 2026-08-24 | `m340`     | 2.2e-01     | 2.7e-01     | 8.2e-01 | 5.2e-02     | **3.6e-02** | 6.7e-02  |
| recovery010-full                 | 2026-08-24 | `m499`     | 5.9e-01     | 1.3e-01     | 1.2e-01 | 1.0e+00     | 1.4e-01  | 6.6e-01     |

† Paired in the checkpoint's own code state: `c1def76` plus the archived run's
62-line uncommitted diff, deliberately retaining the pre-fix reset ordering. Raw
rows and the report are
`logs/comparisons/2026-08-13_10-37-48_std-floor_model_1499.{csv,log}`. It is a
valid within-era policy-vs-baseline comparison, but not a comparison to post-fix
rows: the orientation bug changes the task distribution. The detailed context is
in
[Pre-orientation-fix reference](#pre-orientation-fix-reference-2026-08-13_10-37-48_std-floor).

### Reading the Summary columns

- **run** is the training configuration family; rows with the same name are
  checkpoints from one training run, not independent replications.
- **train date** is the date the corresponding training run started, in local
  run directory time; it identifies the code and objective era more reliably
  than the checkpoint iteration alone.
- **ckpt** is the PPO iteration checkpoint evaluated (`m1499` means model 1499).
- **tracking sum** is `(policy - baseline) / baseline` for the mean per-step sum
  of that run's tracking rewards, excluding residual penalties. Positive is
  better for the policy. Compare signs and within-run deltas across reward eras,
  not magnitudes.
- **hazard** is policy / baseline adverse-ending rate per second after the 10 s
  warm-up, with cap-reaching episodes censored. Below 1 favours the policy; it
  is more robust to the episode cap than survival percentage.
- **fell_over** is policy minus baseline share of episodes labelled `fell_over`,
  in percentage points. The label definition changed on 2026-08-19, so do not
  compare this column across that boundary.
- **collapsed** is policy minus baseline share of episodes labelled `collapsed`,
  in percentage points. Before 2026-08-19 it overlaps with `fell_over`;
  afterwards the labels are mutually exclusive. Do not compare this column
  across that boundary or add it to `fell_over`; use hazard for the
  all-adverse-outcomes rate.
- **survival** is policy minus baseline share of episodes that reached the 90 s
  cap (`time_out`), in percentage points. Positive favours the policy.
- **ep len** is policy minus baseline mean episode duration in seconds. Positive
  means the policy stayed active longer, but is not tracking evidence on its
  own.

### What each run changed

- **`std-floor`** — reference config: gamma 0.99, scale 0.01, 5x4 epochs x
  minibatches. First run with `std_range` and `desired_kl = 0.02`.
- **`scale03-g997`** — **gamma 0.99 -> 0.997** (2 s -> 6.7 s credit horizon,
  `num_steps_per_env` 48 -> 96) **and residual_scale 0.01 -> 0.03**, bundled.
- **`scale01-g997`** — **residual_scale reverted 0.03 -> 0.01**, keeping gamma
  0.997. Splits the previous bundle.
- **`zeroinit-4ev`** — **`ZeroInitMLPModel`** (rsl_rl never zeroed the actor's
  mean head: untrained peak 0.43 against `init_std` 0.10) **and 5x4 -> 2x2
  epochs x minibatches** (LR adaptation events 20 -> 4).
- **`dcm-obs`** — **objective swapped to `dcm_stability`** (LIPM divergence
  rate, plan-matching demoted 0.5 -> 0.05), plus `angular_momentum` and
  `foot_slip`; **observations 284 -> 1179** (`q_ref`, position error, gait
  proxies, history 5 -> 20, four critic-only terms).
- **`dcm-obs-gate`** — **directional-coherence gate on** (`GATE_STRENGTH` 1.0,
  `GATE_ALPHA_REF` 0.5): withholds authority from a residual opposing the
  controller's commanded joint velocity. Otherwise identical to `dcm-obs`.
- **`gate-torque-phase`** — gate retained, plus **`residual_rate` clamped** at
  `RAW_CLIP` (the second cause of the iteration-2945 divergence),
  **`executed_action`** replacing `last_action` so the policy observes the gated
  action, **`Episode_Metrics/gate_mean`** logged, **`torque_margin`** guard with
  the task's first weight curriculum, and **`gait_phase`** (obs 1179 -> 1219).
- **`pruned-nogate`** — **agreement rewards deleted** -- `zmp_tracking` and
  `com_velocity_tracking` scored `measured - planned` against a plan the
  stabilizer QP already optimises; both survive as metrics (`zmp_error`,
  `com_velocity_error`). **Gate reverted to 0** after it failed its own
  falsification test. Rewards 11 -> 9. Capped at 500 iterations.
- **`pruned-nogate +500`** — resumed `m499` for 500 updates with source code
  equivalent to reachable `952ced5`, plus the archived HRP5P controller
  selection; evaluated at `m998`.
- **`head-authority20`** — **`residual_scale` 0.01 -> 0.20 for position
  control** (20x the clip, `c386cd3`), on `dcm-cmd-ref`'s configuration and
  objective, run to 1000 iterations instead of 500. Nothing else differs;
  `dcm-cmd-ref` is its natural comparator on the training side.
- **`rg-ref01` / `rg-entropy0` / `rg-fixedstd010` / `rg-mag03`** — four
  188-iteration residual-growth screens at restored position scale 0.01.
  Isolated entropy `0`, fixed std `0.10`, and magnitude weight `-0.3` against
  the current learned-std, entropy-0.0005, magnitude-`-0.1` reference.
- **`recovery-only-005`** — nominal/recovery DCM weights `1/1 -> 0/4`, keeping
  recovery std 0.05. The 4x gated weight approximately preserves average DCM
  reward density while removing payment on ordinary walking.
- **`recovery-only-010`** — the same recovery-only objective with only its std
  widened `0.05 -> 0.10`, centering useful gradient on the measured 0.10 m
  post-push error peak. The first row is the 188-iteration screen.
- **`recovery010-full`** — a fresh 500-iteration run of the successful screen,
  bracketed at models 180, 340 and 499 to test whether the result survives
  continued optimization.

Each row is measured against **its own** zero-residual arm under **its own**
code state, so deltas are comparable across rows even where absolute term values
are not.

⚠ = fewer than 16 envs per arm. See _Statistical power_ below: at 8 envs/arm the
sign of `tracking sum` is wrong 12.7% of the time, at 6 envs/arm 15.3%.

## Residual-growth and recovery-objective screens, 2026-08-21--24

The residual-growth matrix did not identify a PPO-side suppressor. Zero entropy
left the policy-mean tail 1.7% above reference. Fixed std reduced it 7.3% and its
slope 42.8%, short of the registered gate. A threefold magnitude penalty cut the
slope 73.7%, but its deterministic policy had hazard 1.10 and foot-slip cost
+90.5% (environment-clustered p=0.000347). The production reference therefore
remained scale 0.01, entropy 0.0005, learned std and both action penalties -0.1.

Moving all DCM shaping onto the exogenous recovery window located the source of
growth more precisely:

| run | policy-mean tail | slope | recovery DCM | hazard | deterministic magnitude cost/step |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rg-ref01` | 0.04696 | +0.0002289 | +0.33% | 0.98 | -0.0000811 |
| `recovery-only-005` | 0.02847 | +0.0001165 | **-8.14%** | 0.95 | -0.0000263 |
| `recovery-only-010` | 0.02648 | **-0.0001046** | **+1.03%** | **0.84** | -0.0000144 |

The 0.05 recovery kernel suppressed the residual but failed to improve the
quantity it exclusively optimized. Widening it to 0.10 moved gradient onto the
measured push peak and passed the short screen, which justified the full run.

The full run then reproduced the repository's overtraining pattern rather than
holding the screen result:

| model | recovery DCM | hazard | survival | episode duration | magnitude cost/step |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 180 | -3.70% | 1.03 | +1.0 pp | -0.4 s | -0.0000230 |
| 340 | **+7.43%** | **0.69** | **+15.6 pp** (p=0.0359) | **+7.9 s** | -0.0000866 |
| 499 | +1.88% | **1.16** | -7.0 pp | -1.5 s | -0.000161 |

Model 340 proves the broad recovery objective can teach useful recovery, but its
residual magnitude had already exceeded `rg-ref01` by 6.8%. By model 499 it was
98.1% above reference and the hazard gate had reversed. The registered
two-of-three adoption rule failed; neither recovery-only configuration became a
default. Raw rows and reports use the run-directory/model stems listed in the
summary and remain under `logs/comparisons/`.

## std-floor @ `m1050`

`logs/rsl_rl/Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position/2026-08-14_19-14-46_std-floor`

28 envs per arm, K = 5 episodes per env, n = 140 vs 140 after the fixed-K trim
(483 episodes finished in total), reward era **old**. **This section uses every
available episode**, unlike the summary table, which trims all runs to K = 4 so
`n/arm` is uniform. Where the two disagree the section is the better-powered
estimate and the summary is the comparable one.

### Per-step reward rates

| term                    | baseline     | policy       | delta     | p            |
| ----------------------- | ------------ | ------------ | --------- | ------------ |
| `zmp_tracking`          | 0.006518     | 0.006153     | -5.6%     | 8.25e-08     |
| `com_velocity_tracking` | 0.007783     | 0.007271     | -6.6%     | 0.00e+00     |
| `recovery_tracking`     | 0.002437     | 0.001975     | -18.9%    | 2.30e-04     |
| `residual_magnitude`    | 0.000000     | -0.001255    | --        | 0.00e+00     |
| `residual_rate`         | 0.000000     | -0.000166    | --        | 0.00e+00     |
| **tracking sum**        | **0.016738** | **0.015400** | **-8.0%** | **6.36e-12** |

### Terminations

| outcome             | baseline | policy | delta  | p     |
| ------------------- | -------- | ------ | ------ | ----- |
| `time_out`          | 25.0%    | 17.9%  | -7.1pp | 0.145 |
| `fell_over`         | 23.6%    | 22.9%  | -0.7pp | 0.887 |
| `collapsed`         | 59.3%    | 63.6%  | +4.3pp | 0.461 |
| `controller_failed` | 0.7%     | 1.4%   | +0.7pp | 0.562 |

### Episode length and hazard

- baseline mean 54.1 s (median 54.6 s), policy mean 45.7 s (median 37.2 s),
  delta -8.5 s, p = 0.011
- hazard baseline **0.0170/s** (+/-0.0017, 105 falls over 6179 s), policy
  **0.0230/s** (+/-0.0021, 115 falls over 4995 s)
- **hazard ratio 1.355**; mean life 1/h = 59 s vs 43 s

### `zmp_tracking` per-step by episode-length band

| band      | n base | n policy | baseline | policy   | delta | p        |
| --------- | ------ | -------- | -------- | -------- | ----- | -------- |
| < 30 s    | 40     | 56       | 0.006483 | 0.006087 | -6.1% | 8.78e-06 |
| 30-60 s   | 39     | 40       | 0.006414 | 0.006147 | -4.2% | 4.68e-02 |
| 60-90 s   | 26     | 19       | 0.006453 | 0.006124 | -5.1% | 1.86e-02 |
| survivors | 35     | 25       | 0.006723 | 0.006334 | -5.8% | 6.20e-02 |

- `corr(per-step |residual|, episode length)` = **-0.43** (was -0.60 under gamma
  = 0.99, the failure signature)
- per-step residual magnitude 0.00126

## std-floor @ `m3050`

`logs/rsl_rl/Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position/2026-08-14_19-14-46_std-floor`

28 envs per arm, K = 6 episodes per env, n = 168 vs 168 after the fixed-K trim
(471 episodes finished in total), reward era **old**. **This section uses every
available episode**, unlike the summary table, which trims all runs to K = 4 so
`n/arm` is uniform. Where the two disagree the section is the better-powered
estimate and the summary is the comparable one.

### Per-step reward rates

| term                    | baseline     | policy       | delta     | p            |
| ----------------------- | ------------ | ------------ | --------- | ------------ |
| `zmp_tracking`          | 0.006513     | 0.006423     | -1.4%     | 1.53e-01     |
| `com_velocity_tracking` | 0.007738     | 0.007625     | -1.5%     | 1.45e-02     |
| `recovery_tracking`     | 0.002467     | 0.002517     | +2.0%     | 6.29e-01     |
| `residual_magnitude`    | 0.000000     | -0.001606    | --        | 0.00e+00     |
| `residual_rate`         | 0.000000     | -0.000180    | --        | 0.00e+00     |
| **tracking sum**        | **0.016718** | **0.016565** | **-0.9%** | **3.72e-01** |

### Terminations

| outcome             | baseline | policy | delta  | p     |
| ------------------- | -------- | ------ | ------ | ----- |
| `time_out`          | 20.2%    | 25.6%  | +5.4pp | 0.243 |
| `fell_over`         | 13.7%    | 21.4%  | +7.7pp | 0.062 |
| `collapsed`         | 68.5%    | 60.7%  | -7.7pp | 0.138 |
| `controller_failed` | 1.2%     | 0.0%   | -1.2pp | 0.156 |

### Episode length and hazard

- baseline mean 49.7 s (median 43.7 s), policy mean 56.0 s (median 54.1 s),
  delta +6.2 s, p = 0.037
- hazard baseline **0.0199/s** (+/-0.0017, 133 falls over 6679 s), policy
  **0.0162/s** (+/-0.0014, 125 falls over 7720 s)
- **hazard ratio 0.813**; mean life 1/h = 50 s vs 62 s

### `zmp_tracking` per-step by episode-length band

| band      | n base | n policy | baseline | policy   | delta | p        |
| --------- | ------ | -------- | -------- | -------- | ----- | -------- |
| < 30 s    | 52     | 41       | 0.006493 | 0.006506 | +0.2% | 9.00e-01 |
| 30-60 s   | 59     | 50       | 0.006401 | 0.006336 | -1.0% | 5.30e-01 |
| 60-90 s   | 23     | 34       | 0.006906 | 0.006318 | -8.5% | 1.61e-03 |
| survivors | 34     | 43       | 0.006475 | 0.006530 | +0.9% | 7.19e-01 |

- `corr(per-step |residual|, episode length)` = **-0.50** (was -0.60 under gamma
  = 0.99, the failure signature)
- per-step residual magnitude 0.00161

## scale03-g997 @ `m1150`

`logs/rsl_rl/Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position/2026-08-15_07-30-27_scale03-gamma997`

28 envs per arm, K = 4 episodes per env, n = 112 vs 112 after the fixed-K trim
(372 episodes finished in total), reward era **old**. **This section uses every
available episode**, unlike the summary table, which trims all runs to K = 4 so
`n/arm` is uniform. Where the two disagree the section is the better-powered
estimate and the summary is the comparable one.

### Per-step reward rates

| term                    | baseline     | policy       | delta      | p            |
| ----------------------- | ------------ | ------------ | ---------- | ------------ |
| `zmp_tracking`          | 0.006460     | 0.005379     | -16.7%     | 0.00e+00     |
| `com_velocity_tracking` | 0.007696     | 0.005757     | -25.2%     | 0.00e+00     |
| `recovery_tracking`     | 0.002522     | 0.002105     | -16.5%     | 7.09e-04     |
| `residual_magnitude`    | 0.000000     | -0.001250    | --         | 0.00e+00     |
| `residual_rate`         | 0.000000     | -0.000084    | --         | 0.00e+00     |
| **tracking sum**        | **0.016679** | **0.013241** | **-20.6%** | **0.00e+00** |

### Terminations

| outcome             | baseline | policy | delta   | p     |
| ------------------- | -------- | ------ | ------- | ----- |
| `time_out`          | 18.8%    | 29.5%  | +10.7pp | 0.061 |
| `fell_over`         | 19.6%    | 19.6%  | +0.0pp  | 1.000 |
| `collapsed`         | 70.5%    | 57.1%  | -13.4pp | 0.037 |
| `controller_failed` | 0.0%     | 0.0%   | +0.0pp  | nan   |

### Episode length and hazard

- baseline mean 55.1 s (median 56.6 s), policy mean 57.8 s (median 59.1 s),
  delta +2.6 s, p = 0.482
- hazard baseline **0.0180/s** (+/-0.0019, 91 falls over 5056 s), policy
  **0.0148/s** (+/-0.0017, 79 falls over 5349 s)
- **hazard ratio 0.821**; mean life 1/h = 56 s vs 68 s

### `zmp_tracking` per-step by episode-length band

| band      | n base | n policy | baseline | policy   | delta  | p        |
| --------- | ------ | -------- | -------- | -------- | ------ | -------- |
| < 30 s    | 33     | 24       | 0.006272 | 0.005403 | -13.9% | 0.00e+00 |
| 30-60 s   | 26     | 33       | 0.006306 | 0.005196 | -17.6% | 1.71e-14 |
| 60-90 s   | 32     | 22       | 0.006729 | 0.005367 | -20.2% | 1.32e-10 |
| survivors | 21     | 33       | 0.006539 | 0.005553 | -15.1% | 5.55e-08 |

- `corr(per-step |residual|, episode length)` = **-0.49** (was -0.60 under gamma
  = 0.99, the failure signature)
- per-step residual magnitude 0.00125

## scale01-g997 @ `m950`

`logs/rsl_rl/Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position/2026-08-17_12-18-01_scale01-gamma997`

28 envs per arm, K = 5 episodes per env, n = 140 vs 140 after the fixed-K trim
(467 episodes finished in total), reward era **old**. **This section uses every
available episode**, unlike the summary table, which trims all runs to K = 4 so
`n/arm` is uniform. Where the two disagree the section is the better-powered
estimate and the summary is the comparable one.

### Per-step reward rates

| term                    | baseline     | policy       | delta     | p            |
| ----------------------- | ------------ | ------------ | --------- | ------------ |
| `zmp_tracking`          | 0.006524     | 0.006033     | -7.5%     | 1.19e-10     |
| `com_velocity_tracking` | 0.007781     | 0.007070     | -9.1%     | 0.00e+00     |
| `recovery_tracking`     | 0.002504     | 0.002461     | -1.7%     | 7.23e-01     |
| `residual_magnitude`    | 0.000000     | -0.002028    | --        | 0.00e+00     |
| `residual_rate`         | 0.000000     | -0.000199    | --        | 0.00e+00     |
| **tracking sum**        | **0.016809** | **0.015564** | **-7.4%** | **1.22e-08** |

### Terminations

| outcome             | baseline | policy | delta  | p     |
| ------------------- | -------- | ------ | ------ | ----- |
| `time_out`          | 20.7%    | 26.4%  | +5.7pp | 0.260 |
| `fell_over`         | 21.4%    | 14.3%  | -7.1pp | 0.119 |
| `collapsed`         | 67.1%    | 67.9%  | +0.7pp | 0.898 |
| `controller_failed` | 0.0%     | 1.4%   | +1.4pp | 0.156 |

### Episode length and hazard

- baseline mean 51.4 s (median 48.7 s), policy mean 55.8 s (median 55.3 s),
  delta +4.3 s, p = 0.196
- hazard baseline **0.0191/s** (+/-0.0018, 111 falls over 5800 s), policy
  **0.0161/s** (+/-0.0016, 103 falls over 6408 s)
- **hazard ratio 0.840**; mean life 1/h = 52 s vs 62 s

### `zmp_tracking` per-step by episode-length band

| band      | n base | n policy | baseline | policy   | delta | p        |
| --------- | ------ | -------- | -------- | -------- | ----- | -------- |
| < 30 s    | 44     | 34       | 0.006412 | 0.005857 | -8.6% | 1.74e-10 |
| 30-60 s   | 38     | 44       | 0.006444 | 0.005904 | -8.4% | 3.06e-05 |
| 60-90 s   | 29     | 25       | 0.006489 | 0.005966 | -8.1% | 1.96e-03 |
| survivors | 29     | 37       | 0.006834 | 0.006394 | -6.4% | 3.47e-02 |

- `corr(per-step |residual|, episode length)` = **-0.42** (was -0.60 under gamma
  = 0.99, the failure signature)
- per-step residual magnitude 0.00203

## zeroinit-4ev @ `m1000`

`logs/rsl_rl/Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position/2026-08-17_15-38-02_zeroinit-4ev`

28 envs per arm, K = 6 episodes per env, n = 168 vs 168 after the fixed-K trim
(466 episodes finished in total), reward era **old**. **This section uses every
available episode**, unlike the summary table, which trims all runs to K = 4 so
`n/arm` is uniform. Where the two disagree the section is the better-powered
estimate and the summary is the comparable one.

### Per-step reward rates

| term                    | baseline     | policy       | delta     | p            |
| ----------------------- | ------------ | ------------ | --------- | ------------ |
| `zmp_tracking`          | 0.006372     | 0.006336     | -0.6%     | 6.08e-01     |
| `com_velocity_tracking` | 0.007611     | 0.007146     | -6.1%     | 0.00e+00     |
| `recovery_tracking`     | 0.002568     | 0.002464     | -4.1%     | 3.12e-01     |
| `residual_magnitude`    | 0.000000     | -0.001438    | --        | 0.00e+00     |
| `residual_rate`         | 0.000000     | -0.000148    | --        | 2.56e-02     |
| **tracking sum**        | **0.016552** | **0.015946** | **-3.7%** | **1.39e-03** |

### Terminations

| outcome             | baseline | policy | delta  | p     |
| ------------------- | -------- | ------ | ------ | ----- |
| `time_out`          | 22.0%    | 24.4%  | +2.4pp | 0.605 |
| `fell_over`         | 13.7%    | 14.3%  | +0.6pp | 0.875 |
| `collapsed`         | 66.7%    | 64.9%  | -1.8pp | 0.730 |
| `controller_failed` | 0.0%     | 0.0%   | +0.0pp | nan   |

### Episode length and hazard

- baseline mean 53.1 s (median 47.1 s), policy mean 53.6 s (median 50.5 s),
  delta +0.5 s, p = 0.855
- hazard baseline **0.0181/s** (+/-0.0016, 131 falls over 7239 s), policy
  **0.0173/s** (+/-0.0015, 127 falls over 7330 s)
- **hazard ratio 0.957**; mean life 1/h = 55 s vs 58 s

### `zmp_tracking` per-step by episode-length band

| band      | n base | n policy | baseline | policy   | delta | p        |
| --------- | ------ | -------- | -------- | -------- | ----- | -------- |
| < 30 s    | 46     | 45       | 0.006265 | 0.006329 | +1.0% | 4.60e-01 |
| 30-60 s   | 54     | 56       | 0.006345 | 0.006232 | -1.8% | 3.05e-01 |
| 60-90 s   | 31     | 25       | 0.006410 | 0.006407 | -0.0% | 9.91e-01 |
| survivors | 37     | 41       | 0.006514 | 0.006461 | -0.8% | 7.28e-01 |

- `corr(per-step |residual|, episode length)` = **-0.05** (was -0.60 under gamma
  = 0.99, the failure signature)
- per-step residual magnitude 0.00144

## zeroinit-4ev @ `m2900`

`logs/rsl_rl/Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position/2026-08-17_15-38-02_zeroinit-4ev`

28 envs per arm, K = 8 episodes per env, n = 224 vs 224 after the fixed-K trim
(637 episodes finished in total), reward era **old**. **This section uses every
available episode**, unlike the summary table, which trims all runs to K = 4 so
`n/arm` is uniform. Where the two disagree the section is the better-powered
estimate and the summary is the comparable one.

### Per-step reward rates

| term                    | baseline     | policy       | delta     | p            |
| ----------------------- | ------------ | ------------ | --------- | ------------ |
| `zmp_tracking`          | 0.006532     | 0.006157     | -5.7%     | 1.89e-09     |
| `com_velocity_tracking` | 0.007757     | 0.007254     | -6.5%     | 0.00e+00     |
| `recovery_tracking`     | 0.002496     | 0.002354     | -5.7%     | 1.51e-01     |
| `residual_magnitude`    | 0.000000     | -0.001373    | --        | 0.00e+00     |
| `residual_rate`         | 0.000000     | -0.000090    | --        | 0.00e+00     |
| **tracking sum**        | **0.016785** | **0.015766** | **-6.1%** | **3.59e-09** |

### Terminations

| outcome             | baseline | policy | delta   | p     |
| ------------------- | -------- | ------ | ------- | ----- |
| `time_out`          | 21.4%    | 26.8%  | +5.4pp  | 0.185 |
| `fell_over`         | 15.6%    | 21.9%  | +6.2pp  | 0.090 |
| `collapsed`         | 69.2%    | 56.2%  | -12.9pp | 0.005 |
| `controller_failed` | 0.0%     | 0.9%   | +0.9pp  | 0.156 |

### Episode length and hazard

- baseline mean 51.6 s (median 46.3 s), policy mean 54.6 s (median 49.6 s),
  delta +3.0 s, p = 0.263
- hazard baseline **0.0189/s** (+/-0.0014, 176 falls over 9328 s), policy
  **0.0164/s** (+/-0.0013, 164 falls over 9990 s)
- **hazard ratio 0.870**; mean life 1/h = 53 s vs 61 s

### `zmp_tracking` per-step by episode-length band

| band      | n base | n policy | baseline | policy   | delta | p        |
| --------- | ------ | -------- | -------- | -------- | ----- | -------- |
| < 30 s    | 70     | 63       | 0.006460 | 0.006009 | -7.0% | 8.68e-10 |
| 30-60 s   | 69     | 62       | 0.006401 | 0.006238 | -2.5% | 1.49e-01 |
| 60-90 s   | 37     | 39       | 0.006577 | 0.006093 | -7.3% | 5.50e-03 |
| survivors | 48     | 60       | 0.006792 | 0.006272 | -7.7% | 1.11e-03 |

- `corr(per-step |residual|, episode length)` = **-0.27** (was -0.60 under gamma
  = 0.99, the failure signature)
- per-step residual magnitude 0.00137

## dcm-obs @ `m900`

`logs/rsl_rl/Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position/2026-08-18_02-35-58_dcm-obs`

28 envs per arm, K = 7 episodes per env, n = 196 vs 196 after the fixed-K trim
(599 episodes finished in total), reward era **new**. **This section uses every
available episode**, unlike the summary table, which trims all runs to K = 4 so
`n/arm` is uniform. Where the two disagree the section is the better-powered
estimate and the summary is the comparable one.

### Per-step reward rates

| term                    | baseline     | policy       | delta     | p            |
| ----------------------- | ------------ | ------------ | --------- | ------------ |
| `dcm_stability`         | 0.010492     | 0.010967     | +4.5%     | 2.43e-02     |
| `recovery_dcm`          | 0.001889     | 0.002061     | +9.1%     | 8.17e-02     |
| `zmp_tracking`          | 0.000648     | 0.000647     | -0.2%     | 8.97e-01     |
| `com_velocity_tracking` | 0.000770     | 0.000727     | -5.6%     | 2.20e-14     |
| `angular_momentum`      | -0.000950    | -0.000890    | -6.3%     | 2.67e-01     |
| `foot_slip`             | -0.000010    | -0.000009    | -10.7%    | 4.04e-01     |
| `residual_magnitude`    | 0.000000     | -0.001254    | --        | 0.00e+00     |
| `residual_rate`         | 0.000000     | -0.000031    | --        | 0.00e+00     |
| **tracking sum**        | **0.012840** | **0.013502** | **+5.2%** | **3.14e-02** |

### Terminations

| outcome             | baseline | policy | delta  | p     |
| ------------------- | -------- | ------ | ------ | ----- |
| `time_out`          | 28.1%    | 31.6%  | +3.6pp | 0.440 |
| `fell_over`         | 18.4%    | 19.4%  | +1.0pp | 0.796 |
| `collapsed`         | 59.2%    | 56.6%  | -2.6pp | 0.609 |
| `controller_failed` | 0.0%     | 0.0%   | +0.0pp | nan   |

### Episode length and hazard

- baseline mean 55.5 s (median 56.2 s), policy mean 55.6 s (median 52.8 s),
  delta +0.0 s, p = 0.999
- hazard baseline **0.0158/s** (+/-0.0013, 141 falls over 8928 s), policy
  **0.0150/s** (+/-0.0013, 134 falls over 8928 s)
- **hazard ratio 0.950**; mean life 1/h = 63 s vs 67 s

### `dcm_stability` per-step by episode-length band

| band      | n base | n policy | baseline | policy   | delta  | p        |
| --------- | ------ | -------- | -------- | -------- | ------ | -------- |
| < 30 s    | 51     | 56       | 0.011055 | 0.010781 | -2.5%  | 2.99e-01 |
| 30-60 s   | 55     | 51       | 0.010250 | 0.010395 | +1.4%  | 6.83e-01 |
| 60-90 s   | 35     | 27       | 0.010250 | 0.010667 | +4.1%  | 5.19e-01 |
| survivors | 55     | 62       | 0.010367 | 0.011735 | +13.2% | 2.33e-03 |

- `corr(per-step |residual|, episode length)` = **-0.41** (was -0.60 under gamma
  = 0.99, the failure signature)
- per-step residual magnitude 0.00125

## dcm-obs @ `m1850`

`logs/rsl_rl/Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position/2026-08-18_02-35-58_dcm-obs`

28 envs per arm, K = 7 episodes per env, n = 196 vs 196 after the fixed-K trim
(604 episodes finished in total), reward era **new**. **This section uses every
available episode**, unlike the summary table, which trims all runs to K = 4 so
`n/arm` is uniform. Where the two disagree the section is the better-powered
estimate and the summary is the comparable one.

### Per-step reward rates

| term                    | baseline     | policy       | delta     | p            |
| ----------------------- | ------------ | ------------ | --------- | ------------ |
| `dcm_stability`         | 0.010502     | 0.010947     | +4.2%     | 6.88e-03     |
| `recovery_dcm`          | 0.001890     | 0.002066     | +9.3%     | 2.88e-02     |
| `zmp_tracking`          | 0.000649     | 0.000657     | +1.3%     | 1.66e-01     |
| `com_velocity_tracking` | 0.000772     | 0.000717     | -7.2%     | 0.00e+00     |
| `angular_momentum`      | -0.001002    | -0.000830    | -17.2%    | 4.74e-04     |
| `foot_slip`             | -0.000009    | -0.000008    | -19.2%    | 9.03e-02     |
| `residual_magnitude`    | 0.000000     | -0.001155    | --        | 0.00e+00     |
| `residual_rate`         | 0.000000     | -0.000036    | --        | 0.00e+00     |
| **tracking sum**        | **0.012801** | **0.013549** | **+5.8%** | **1.18e-03** |

### Terminations

| outcome             | baseline | policy | delta  | p     |
| ------------------- | -------- | ------ | ------ | ----- |
| `time_out`          | 20.9%    | 28.6%  | +7.7pp | 0.079 |
| `fell_over`         | 19.4%    | 20.4%  | +1.0pp | 0.800 |
| `collapsed`         | 64.3%    | 57.7%  | -6.6pp | 0.178 |
| `controller_failed` | 1.0%     | 1.0%   | +0.0pp | 1.000 |

### Episode length and hazard

- baseline mean 51.8 s (median 49.3 s), policy mean 57.7 s (median 56.2 s),
  delta +5.9 s, p = 0.033
- hazard baseline **0.0188/s** (+/-0.0015, 154 falls over 8191 s), policy
  **0.0150/s** (+/-0.0013, 140 falls over 9350 s)
- **hazard ratio 0.796**; mean life 1/h = 53 s vs 67 s

### `dcm_stability` per-step by episode-length band

| band      | n base | n policy | baseline | policy   | delta | p        |
| --------- | ------ | -------- | -------- | -------- | ----- | -------- |
| < 30 s    | 58     | 41       | 0.010763 | 0.011553 | +7.3% | 4.04e-03 |
| 30-60 s   | 61     | 62       | 0.010254 | 0.010575 | +3.1% | 2.19e-01 |
| 60-90 s   | 36     | 37       | 0.010697 | 0.011121 | +4.0% | 3.68e-01 |
| survivors | 41     | 56       | 0.010332 | 0.010799 | +4.5% | 1.84e-01 |

- `corr(per-step |residual|, episode length)` = **-0.07** (was -0.60 under gamma
  = 0.99, the failure signature)
- per-step residual magnitude 0.00115

## dcm-obs-gate @ `m250`

`logs/rsl_rl/Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position/2026-08-18_13-05-51_dcm-obs-gate`

28 envs per arm, K = 4 episodes per env, n = 112 vs 112 after the fixed-K trim
(328 episodes finished in total), reward era **new**. **This section uses every
available episode**, unlike the summary table, which trims all runs to K = 4 so
`n/arm` is uniform. Where the two disagree the section is the better-powered
estimate and the summary is the comparable one.

### Per-step reward rates

| term                    | baseline     | policy       | delta     | p            |
| ----------------------- | ------------ | ------------ | --------- | ------------ |
| `dcm_stability`         | 0.010649     | 0.010664     | +0.1%     | 9.52e-01     |
| `recovery_dcm`          | 0.002030     | 0.002138     | +5.3%     | 3.32e-01     |
| `zmp_tracking`          | 0.000655     | 0.000656     | +0.3%     | 8.32e-01     |
| `com_velocity_tracking` | 0.000777     | 0.000769     | -1.0%     | 2.16e-01     |
| `angular_momentum`      | -0.000863    | -0.000836    | -3.2%     | 6.55e-01     |
| `foot_slip`             | -0.000007    | -0.000009    | +21.0%    | 2.03e-01     |
| `residual_magnitude`    | 0.000000     | -0.000243    | --        | 0.00e+00     |
| `residual_rate`         | 0.000000     | -0.000010    | --        | 0.00e+00     |
| **tracking sum**        | **0.013240** | **0.013383** | **+1.1%** | **6.91e-01** |

### Terminations

| outcome             | baseline | policy | delta  | p     |
| ------------------- | -------- | ------ | ------ | ----- |
| `time_out`          | 27.7%    | 32.1%  | +4.5pp | 0.466 |
| `fell_over`         | 23.2%    | 17.0%  | -6.2pp | 0.243 |
| `collapsed`         | 56.2%    | 58.0%  | +1.8pp | 0.787 |
| `controller_failed` | 0.0%     | 1.8%   | +1.8pp | 0.155 |

### Episode length and hazard

- baseline mean 56.3 s (median 54.2 s), policy mean 59.7 s (median 63.0 s),
  delta +3.3 s, p = 0.366
- hazard baseline **0.0156/s** (+/-0.0017, 81 falls over 5186 s), policy
  **0.0137/s** (+/-0.0016, 76 falls over 5561 s)
- **hazard ratio 0.875**; mean life 1/h = 64 s vs 73 s

### `dcm_stability` per-step by episode-length band

| band      | n base | n policy | baseline | policy   | delta | p        |
| --------- | ------ | -------- | -------- | -------- | ----- | -------- |
| < 30 s    | 29     | 22       | 0.010535 | 0.010810 | +2.6% | 3.79e-01 |
| 30-60 s   | 30     | 33       | 0.010415 | 0.010522 | +1.0% | 7.87e-01 |
| 60-90 s   | 22     | 21       | 0.010605 | 0.010305 | -2.8% | 5.58e-01 |
| survivors | 31     | 36       | 0.011013 | 0.010912 | -0.9% | 8.63e-01 |

- `corr(per-step |residual|, episode length)` = **-0.28** (was -0.60 under gamma
  = 0.99, the failure signature)
- per-step residual magnitude 0.00024

## dcm-obs-gate @ `m1500`

`logs/rsl_rl/Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position/2026-08-18_13-05-51_dcm-obs-gate`

28 envs per arm, K = 4 episodes per env, n = 112 vs 112 after the fixed-K trim
(344 episodes finished in total), reward era **new**. **This section uses every
available episode**, unlike the summary table, which trims all runs to K = 4 so
`n/arm` is uniform. Where the two disagree the section is the better-powered
estimate and the summary is the comparable one.

### Per-step reward rates

| term                    | baseline     | policy       | delta     | p            |
| ----------------------- | ------------ | ------------ | --------- | ------------ |
| `dcm_stability`         | 0.010651     | 0.010483     | -1.6%     | 4.56e-01     |
| `recovery_dcm`          | 0.001876     | 0.001907     | +1.7%     | 7.96e-01     |
| `zmp_tracking`          | 0.000649     | 0.000639     | -1.5%     | 2.87e-01     |
| `com_velocity_tracking` | 0.000771     | 0.000719     | -6.7%     | 6.22e-15     |
| `angular_momentum`      | -0.001038    | -0.000898    | -13.5%    | 4.29e-02     |
| `foot_slip`             | -0.000008    | -0.000008    | +6.4%     | 7.34e-01     |
| `residual_magnitude`    | 0.000000     | -0.001494    | --        | 0.00e+00     |
| `residual_rate`         | 0.000000     | -0.000038    | --        | 0.00e+00     |
| **tracking sum**        | **0.012900** | **0.012842** | **-0.5%** | **8.61e-01** |

### Terminations

| outcome             | baseline | policy | delta   | p     |
| ------------------- | -------- | ------ | ------- | ----- |
| `time_out`          | 17.9%    | 26.8%  | +8.9pp  | 0.109 |
| `fell_over`         | 14.3%    | 12.5%  | -1.8pp  | 0.695 |
| `collapsed`         | 74.1%    | 61.6%  | -12.5pp | 0.045 |
| `controller_failed` | 1.8%     | 2.7%   | +0.9pp  | 0.651 |

### Episode length and hazard

- baseline mean 50.3 s (median 47.5 s), policy mean 55.3 s (median 55.9 s),
  delta +5.0 s, p = 0.185
- hazard baseline **0.0204/s** (+/-0.0021, 92 falls over 4515 s), policy
  **0.0159/s** (+/-0.0018, 81 falls over 5085 s)
- **hazard ratio 0.782**; mean life 1/h = 49 s vs 63 s

### `dcm_stability` per-step by episode-length band

| band      | n base | n policy | baseline | policy   | delta | p        |
| --------- | ------ | -------- | -------- | -------- | ----- | -------- |
| < 30 s    | 37     | 32       | 0.010888 | 0.010864 | -0.2% | 9.20e-01 |
| 30-60 s   | 33     | 28       | 0.010344 | 0.009913 | -4.2% | 2.38e-01 |
| 60-90 s   | 22     | 22       | 0.010831 | 0.010260 | -5.3% | 3.89e-01 |
| survivors | 20     | 30       | 0.010521 | 0.010773 | +2.4% | 6.85e-01 |

- `corr(per-step |residual|, episode length)` = **-0.25** (was -0.60 under gamma
  = 0.99, the failure signature)
- per-step residual magnitude 0.00149

## gate-torque-phase @ `m250`

`logs/rsl_rl/Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position/2026-08-18_18-53-27_gate-torque-phase`

28 envs per arm, K = 4 episodes per env, n = 112 vs 112 after the fixed-K trim
(417 episodes finished in total), reward era **new**. **This section uses every
available episode**, unlike the summary table, which trims all runs to K = 4 so
`n/arm` is uniform. Where the two disagree the section is the better-powered
estimate and the summary is the comparable one.

### Per-step reward rates

| term                    | baseline     | policy       | delta     | p            |
| ----------------------- | ------------ | ------------ | --------- | ------------ |
| `dcm_stability`         | 0.010817     | 0.010512     | -2.8%     | 1.24e-01     |
| `recovery_dcm`          | 0.002077     | 0.001939     | -6.7%     | 1.85e-01     |
| `zmp_tracking`          | 0.000651     | 0.000646     | -0.9%     | 4.67e-01     |
| `com_velocity_tracking` | 0.000776     | 0.000758     | -2.3%     | 1.16e-03     |
| `angular_momentum`      | -0.000856    | -0.000951    | +11.1%    | 1.56e-01     |
| `foot_slip`             | -0.000009    | -0.000007    | -27.6%    | 1.16e-01     |
| `torque_margin`         | 0.000000     | 0.000000     | --        | nan          |
| `residual_magnitude`    | 0.000000     | -0.000419    | --        | 0.00e+00     |
| `residual_rate`         | 0.000000     | -0.000024    | --        | 0.00e+00     |
| **tracking sum**        | **0.013456** | **0.012897** | **-4.2%** | **6.24e-02** |

### Terminations

| outcome             | baseline | policy | delta   | p     |
| ------------------- | -------- | ------ | ------- | ----- |
| `time_out`          | 33.9%    | 26.8%  | -7.1pp  | 0.245 |
| `fell_over`         | 17.9%    | 11.6%  | -6.2pp  | 0.187 |
| `collapsed`         | 50.9%    | 66.1%  | +15.2pp | 0.021 |
| `controller_failed` | 0.9%     | 1.8%   | +0.9pp  | 0.561 |

### Episode length and hazard

- baseline mean 59.0 s (median 57.7 s), policy mean 54.1 s (median 50.2 s),
  delta -4.9 s, p = 0.192
- hazard baseline **0.0135/s** (+/-0.0016, 74 falls over 5488 s), policy
  **0.0166/s** (+/-0.0018, 82 falls over 4942 s)
- **hazard ratio 1.230**; mean life 1/h = 74 s vs 60 s

### `dcm_stability` per-step by episode-length band

| band      | n base | n policy | baseline | policy   | delta | p        |
| --------- | ------ | -------- | -------- | -------- | ----- | -------- |
| < 30 s    | 22     | 30       | 0.010921 | 0.010884 | -0.3% | 8.91e-01 |
| 30-60 s   | 37     | 37       | 0.010331 | 0.010329 | -0.0% | 9.97e-01 |
| 60-90 s   | 15     | 15       | 0.010090 | 0.010344 | +2.5% | 4.87e-01 |
| survivors | 38     | 30       | 0.011517 | 0.010449 | -9.3% | 3.08e-02 |

- `corr(per-step |residual|, episode length)` = **-0.33** (was -0.60 under gamma
  = 0.99, the failure signature)
- per-step residual magnitude 0.00042

## gate-torque-phase @ `m1499`

`logs/rsl_rl/Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position/2026-08-18_18-53-27_gate-torque-phase`

28 envs per arm, K = 5 episodes per env, n = 140 vs 140 after the fixed-K trim
(445 episodes finished in total), reward era **new**. **This section uses every
available episode**, unlike the summary table, which trims all runs to K = 4 so
`n/arm` is uniform. Where the two disagree the section is the better-powered
estimate and the summary is the comparable one.

### Per-step reward rates

| term                    | baseline     | policy       | delta     | p            |
| ----------------------- | ------------ | ------------ | --------- | ------------ |
| `dcm_stability`         | 0.010677     | 0.010728     | +0.5%     | 7.81e-01     |
| `recovery_dcm`          | 0.001864     | 0.001891     | +1.4%     | 7.97e-01     |
| `zmp_tracking`          | 0.000650     | 0.000641     | -1.4%     | 2.34e-01     |
| `com_velocity_tracking` | 0.000776     | 0.000748     | -3.6%     | 3.71e-07     |
| `angular_momentum`      | -0.000987    | -0.000967    | -2.1%     | 7.38e-01     |
| `foot_slip`             | -0.000010    | -0.000013    | +36.8%    | 6.50e-02     |
| `torque_margin`         | 0.000000     | 0.000000     | --        | nan          |
| `residual_magnitude`    | 0.000000     | -0.001370    | --        | 0.00e+00     |
| `residual_rate`         | 0.000000     | -0.000056    | --        | 0.00e+00     |
| **tracking sum**        | **0.012970** | **0.013028** | **+0.4%** | **8.35e-01** |

### Terminations

| outcome             | baseline | policy | delta  | p     |
| ------------------- | -------- | ------ | ------ | ----- |
| `time_out`          | 22.9%    | 23.6%  | +0.7pp | 0.887 |
| `fell_over`         | 20.0%    | 16.4%  | -3.6pp | 0.439 |
| `collapsed`         | 62.9%    | 67.9%  | +5.0pp | 0.379 |
| `controller_failed` | 1.4%     | 0.0%   | -1.4pp | 0.156 |

### Episode length and hazard

- baseline mean 51.2 s (median 46.3 s), policy mean 53.4 s (median 52.0 s),
  delta +2.2 s, p = 0.512
- hazard baseline **0.0185/s** (+/-0.0018, 107 falls over 5772 s), policy
  **0.0176/s** (+/-0.0017, 107 falls over 6081 s)
- **hazard ratio 0.949**; mean life 1/h = 54 s vs 57 s

### `dcm_stability` per-step by episode-length band

| band      | n base | n policy | baseline | policy   | delta | p        |
| --------- | ------ | -------- | -------- | -------- | ----- | -------- |
| < 30 s    | 39     | 41       | 0.010948 | 0.011051 | +0.9% | 6.53e-01 |
| 30-60 s   | 46     | 38       | 0.010526 | 0.010485 | -0.4% | 9.05e-01 |
| 60-90 s   | 23     | 28       | 0.010569 | 0.010093 | -4.5% | 2.45e-01 |
| survivors | 32     | 33       | 0.010641 | 0.011147 | +4.8% | 3.10e-01 |

- `corr(per-step |residual|, episode length)` = **-0.19** (was -0.60 under gamma
  = 0.99, the failure signature)
- per-step residual magnitude 0.00137

## pruned-nogate @ `m250`

`logs/rsl_rl/Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position/2026-08-19_10-25-48_pruned-nogate`

28 envs per arm, K = 6 episodes per env, n = 168 vs 168 after the fixed-K trim
(438 episodes finished in total), reward era **pruned**. **This section uses
every available episode**, unlike the summary table, which trims all runs to K =
4 so `n/arm` is uniform. Where the two disagree the section is the
better-powered estimate and the summary is the comparable one.

### Per-step reward rates

| term                 | baseline     | policy       | delta     | p            |
| -------------------- | ------------ | ------------ | --------- | ------------ |
| `dcm_stability`      | 0.010873     | 0.010683     | -1.7%     | 3.34e-01     |
| `recovery_dcm`       | 0.001939     | 0.002030     | +4.7%     | 3.94e-01     |
| `angular_momentum`   | -0.000961    | -0.000911    | -5.3%     | 3.27e-01     |
| `foot_slip`          | -0.000010    | -0.000009    | -1.7%     | 9.22e-01     |
| `torque_margin`      | 0.000000     | 0.000000     | --        | nan          |
| `upright`            | -0.000090    | -0.000077    | -13.9%    | 8.86e-02     |
| `residual_magnitude` | 0.000000     | -0.000467    | --        | 0.00e+00     |
| `residual_rate`      | 0.000000     | -0.000019    | --        | 0.00e+00     |
| **tracking sum**     | **0.011751** | **0.011716** | **-0.3%** | **9.04e-01** |

### Terminations

| outcome             | baseline | policy | delta   | p     |
| ------------------- | -------- | ------ | ------- | ----- |
| `time_out`          | 20.8%    | 21.4%  | +0.6pp  | 0.894 |
| `fell_over`         | 25.6%    | 15.5%  | -10.1pp | 0.022 |
| `collapsed`         | 60.1%    | 64.3%  | +4.2pp  | 0.431 |
| `controller_failed` | 1.2%     | 1.2%   | +0.0pp  | 1.000 |

### Episode length and hazard

- baseline mean 51.4 s (median 48.7 s), policy mean 53.8 s (median 51.6 s),
  delta +2.4 s, p = 0.419
- hazard baseline **0.0191/s** (+/-0.0017, 133 falls over 6951 s), policy
  **0.0179/s** (+/-0.0016, 132 falls over 7354 s)
- **hazard ratio 0.938**; mean life 1/h = 52 s vs 56 s

### `dcm_stability` per-step by episode-length band

| band      | n base | n policy | baseline | policy   | delta | p        |
| --------- | ------ | -------- | -------- | -------- | ----- | -------- |
| < 30 s    | 48     | 45       | 0.011017 | 0.010504 | -4.7% | 1.54e-02 |
| 30-60 s   | 55     | 52       | 0.010557 | 0.010365 | -1.8% | 5.16e-01 |
| 60-90 s   | 30     | 35       | 0.011073 | 0.010769 | -2.7% | 5.72e-01 |
| survivors | 35     | 36       | 0.011001 | 0.011282 | +2.6% | 6.30e-01 |

- `corr(per-step |residual|, episode length)` = **-0.33** (was -0.60 under gamma
  = 0.99, the failure signature)
- per-step residual magnitude 0.00047

## pruned-nogate @ `m499`

`logs/rsl_rl/Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position/2026-08-19_10-25-48_pruned-nogate`

28 envs per arm, K = 5 episodes per env, n = 140 vs 140 after the fixed-K trim
(440 episodes finished in total), reward era **pruned**. **This section uses
every available episode**, unlike the summary table, which trims all runs to K =
4 so `n/arm` is uniform. Where the two disagree the section is the
better-powered estimate and the summary is the comparable one.

### Per-step reward rates

| term                 | baseline     | policy       | delta     | p            |
| -------------------- | ------------ | ------------ | --------- | ------------ |
| `dcm_stability`      | 0.010401     | 0.010900     | +4.8%     | 1.32e-02     |
| `recovery_dcm`       | 0.001924     | 0.001999     | +3.9%     | 4.57e-01     |
| `angular_momentum`   | -0.000944    | -0.000910    | -3.6%     | 5.57e-01     |
| `foot_slip`          | -0.000010    | -0.000007    | -24.6%    | 7.24e-02     |
| `torque_margin`      | 0.000000     | 0.000000     | --        | nan          |
| `upright`            | -0.000083    | -0.000085    | +3.0%     | 7.79e-01     |
| `residual_magnitude` | 0.000000     | -0.000697    | --        | 0.00e+00     |
| `residual_rate`      | 0.000000     | -0.000022    | --        | 0.00e+00     |
| **tracking sum**     | **0.011289** | **0.011897** | **+5.4%** | **3.29e-02** |

### Terminations

| outcome             | baseline | policy | delta  | p     |
| ------------------- | -------- | ------ | ------ | ----- |
| `time_out`          | 25.0%    | 27.1%  | +2.1pp | 0.683 |
| `fell_over`         | 15.7%    | 20.7%  | +5.0pp | 0.278 |
| `collapsed`         | 61.4%    | 58.6%  | -2.9pp | 0.626 |
| `controller_failed` | 0.0%     | 1.4%   | +1.4pp | 0.156 |

### Episode length and hazard

- baseline mean 54.9 s (median 52.9 s), policy mean 55.1 s (median 54.5 s),
  delta +0.3 s, p = 0.933
- hazard baseline **0.0167/s** (+/-0.0016, 105 falls over 6280 s), policy
  **0.0161/s** (+/-0.0016, 102 falls over 6318 s)
- **hazard ratio 0.965**; mean life 1/h = 60 s vs 62 s

### `dcm_stability` per-step by episode-length band

| band      | n base | n policy | baseline | policy   | delta | p        |
| --------- | ------ | -------- | -------- | -------- | ----- | -------- |
| < 30 s    | 34     | 37       | 0.010687 | 0.011354 | +6.2% | 3.73e-02 |
| 30-60 s   | 43     | 40       | 0.010296 | 0.010496 | +1.9% | 4.42e-01 |
| 60-90 s   | 28     | 25       | 0.010092 | 0.010695 | +6.0% | 1.91e-01 |
| survivors | 35     | 38       | 0.010500 | 0.011019 | +4.9% | 3.37e-01 |

- `corr(per-step |residual|, episode length)` = **-0.27** (was -0.60 under gamma
  = 0.99, the failure signature)
- per-step residual magnitude 0.00070

## pruned-nogate +500 @ `m998`

`logs/rsl_rl/Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position/2026-08-20_14-39-18_pruned-nogate-resume500`

Resumed `m499` for 500 updates with source code equivalent to reachable
`952ced5`, plus the archived HRP5P controller selection. 64 envs per arm, K = 3
episodes per env, n = 192 vs 192 after the fixed-K trim (687 episodes finished
in total), reward era **pruned**. Raw rows and report:
`logs/comparisons/2026-08-20_14-39-18_pruned-nogate-resume500_model_998.{csv,log}`.

### Per-step reward rates

| term               | baseline     | policy       | delta     | p (env)      |
| ------------------ | ------------ | ------------ | --------- | ------------ |
| `dcm_stability`    | 0.010677     | 0.010722     | +0.4%     | 7.91e-01     |
| `recovery_dcm`     | 0.001907     | 0.001902     | -0.2%     | 9.51e-01     |
| `angular_momentum` | -0.000952    | -0.000923    | -3.0%     | 5.31e-01     |
| `foot_slip`        | -0.000008    | -0.000009    | +9.4%     | 4.38e-01     |
| `torque_margin`    | 0.000000     | 0.000000     | --        | nan          |
| `upright`          | -0.000087    | -0.000079    | -8.5%     | 3.26e-01     |
| **tracking sum**   | **0.011537** | **0.011613** | **+0.7%** | **7.30e-01** |

### Terminations

| outcome             | baseline | policy | delta      | p (env)   |
| ------------------- | -------- | ------ | ---------- | --------- |
| `time_out`          | 20.3%    | 21.9%  | +1.6pp     | 0.705     |
| `fell_over`         | 18.8%    | 16.7%  | -2.1pp     | 0.608     |
| `collapsed`         | 63.0%    | 62.0%  | -1.0pp     | 0.832     |
| `controller_failed` | 1.6%     | 6.8%   | **+5.2pp** | **0.006** |

### Episode length and hazard

- baseline mean 50.7 s (median 48.7 s), policy mean 52.2 s (median 56.0 s),
  delta +1.5 s, p (env) = 0.603.
- hazard baseline **0.0196/s** (153 adverse endings over 7819 s), policy
  **0.0185/s** (150 over 8109 s); **hazard ratio 0.945**, p (env) = 0.673.

**Worker-failure caveat.** Three controller workers died and were respawned
during this comparison. This archived task labels those exogenous worker losses
as `controller_failed`, rather than the truncation introduced on 2026-08-19; the
significant +5.2pp entry above is therefore infrastructure-contaminated and not
evidence about the policy.

## head-authority20 @ `m500`

`logs/rsl_rl/Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position/2026-08-20_17-59-46_head-authority20`

The halfway bracket: the same run at iteration 500, the budget every earlier run
in this record was capped at. 32 envs per arm, K = 3 episodes per env, n = 96 vs
96 after the fixed-K trim (358 episodes finished in total), reward era **cmd**.
Raw rows and report:
`logs/comparisons/2026-08-20_17-59-46_head-authority20_model_500.{csv,log}`.

### Per-step reward rates

| term               | baseline     | policy       | delta      | p (env)      |
| ------------------ | ------------ | ------------ | ---------- | ------------ |
| `dcm_stability`    | 0.012396     | 0.008663     | -30.1%     | 5.63e-64     |
| `recovery_dcm`     | 0.002136     | 0.001405     | -34.2%     | 1.17e-11     |
| `angular_momentum` | -0.000856    | -0.001070    | +25.0%     | 2.18e-03     |
| `foot_slip`        | -0.000008    | -0.000038    | +357.0%    | 1.01e-56     |
| `torque_margin`    | 0.000000     | -0.000000    | --         | 3.17e-01     |
| `upright`          | -0.000071    | -0.000082    | +14.1%     | 2.61e-01     |
| **tracking sum**   | **0.013596** | **0.008878** | **-34.7%** | **2.32e-54** |

### Terminations

| outcome                    | baseline | policy | delta  | p (env) |
| -------------------------- | -------- | ------ | ------ | ------- |
| `time_out`                 | 26.0%    | 17.7%  | -8.3pp | 0.118   |
| `fell_over`                | 16.7%    | 13.5%  | -3.1pp | 0.483   |
| `collapsed`                | 57.3%    | 62.5%  | +5.2pp | 0.468   |
| `controller_worker_failed` | 0.0%     | 6.2%   | +6.2pp | 0.007   |

No controller or worker failure occurred on either arm.

### Episode length and hazard

- baseline mean 58.6 s (median env 63.0 s), policy mean 48.8 s (median env 44.3
  s), delta -9.8 s, p (env) = 0.012.
- hazard baseline **0.0152/s** (71 adverse endings over 4666 s), policy
  **0.0209/s** (78 over 3734 s); **hazard ratio 1.373**, p (env) = 0.341.

### `dcm_stability` per-step by episode-length band

| band      | n base | n policy | baseline | policy   | delta  | p (env)  |
| --------- | ------ | -------- | -------- | -------- | ------ | -------- |
| < 30 s    | 21     | 28       | 0.012649 | 0.009275 | -26.7% | 1.56e-65 |
| 30-60 s   | 24     | 36       | 0.012196 | 0.008581 | -29.6% | 3.85e-29 |
| 60-90 s   | 26     | 14       | 0.012196 | 0.008535 | -30.0% | 1.96e-07 |
| survivors | 25     | 17       | 0.012584 | 0.007992 | -36.5% | 8.75e-12 |

**Worker-failure caveat.** One controller worker died during the policy arm,
taking six episodes across envs 33-41 with it. Censoring those as truncations
rather than falls leaves the reading unchanged: tracking sum **-35.4%** (p =
2e-52, n = 90), hazard ratio **1.285**, episode length **-8.5 s**.

At the halfway point the deficit is already larger than at either late
checkpoint (-34.7% against -12.8% and -27.3%), and every band including the
survivors is 27-37% down. The 500 further updates bought a partial recovery, not
a change of sign: the 0.20 rad range is harmful from early in training and stays
harmful.

## head-authority20 @ `m960`

`logs/rsl_rl/Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position/2026-08-20_17-59-46_head-authority20`

The same run at its `dcm_error` argmin (smoothed peak is iteration 963;
checkpoints save every 20). 39 iterations from `m999` and 0.0002 apart on the
smoothed metric, so this is a replication of the `m999` measurement rather than
a different policy. 32 envs per arm, K = 3 episodes per env, n = 96 vs 96 after
the fixed-K trim (309 episodes finished in total), reward era **cmd**. Raw rows
and report:
`logs/comparisons/2026-08-20_17-59-46_head-authority20_model_960.{csv,log}`.

### Per-step reward rates

| term               | baseline     | policy       | delta      | p (env)      |
| ------------------ | ------------ | ------------ | ---------- | ------------ |
| `dcm_stability`    | 0.012480     | 0.010831     | -13.2%     | 6.07e-09     |
| `recovery_dcm`     | 0.002055     | 0.002093     | +1.8%      | 7.88e-01     |
| `angular_momentum` | -0.000923    | -0.001024    | +10.9%     | 2.37e-01     |
| `foot_slip`        | -0.000009    | -0.000037    | +295.2%    | 2.11e-30     |
| `torque_margin`    | 0.000000     | -0.000001    | --         | 4.02e-04     |
| `upright`          | -0.000078    | -0.000070    | -9.9%      | 3.46e-01     |
| **tracking sum**   | **0.013524** | **0.011791** | **-12.8%** | **6.52e-05** |

### Terminations

| outcome     | baseline | policy | delta  | p (env) |
| ----------- | -------- | ------ | ------ | ------- |
| `time_out`  | 25.0%    | 29.2%  | +4.2pp | 0.531   |
| `fell_over` | 13.5%    | 6.2%   | -7.3pp | 0.052   |
| `collapsed` | 61.5%    | 64.6%  | +3.1pp | 0.670   |

No controller or worker failure occurred on either arm.

### Episode length and hazard

- baseline mean 56.0 s (median env 60.4 s), policy mean 53.0 s (median env 48.5
  s), delta -3.1 s, p (env) = 0.453.
- hazard baseline **0.0163/s** (72 adverse endings over 4421 s), policy
  **0.0165/s** (68 over 4124 s); **hazard ratio 1.012**, p (env) = 0.779.

### `dcm_stability` per-step by episode-length band

| band      | n base | n policy | baseline | policy   | delta  | p (env)  |
| --------- | ------ | -------- | -------- | -------- | ------ | -------- |
| < 30 s    | 29     | 31       | 0.012571 | 0.011003 | -12.5% | 3.88e-05 |
| 30-60 s   | 22     | 25       | 0.012466 | 0.009730 | -21.9% | 3.18e-09 |
| 60-90 s   | 21     | 12       | 0.012207 | 0.010745 | -12.0% | 1.85e-02 |
| survivors | 24     | 28       | 0.012621 | 0.011661 | -7.6%  | 1.39e-01 |

The sign and the significance replicate; the magnitude does not. `m960` loses
-12.8% against `m999`'s -27.3% on a nearly identical policy, with the two
zero-residual arms agreeing to within 0.1% (0.013524 against 0.013516). The
spread is between the two _policy_ arms, on checkpoints 39 iterations apart
whose smoothed `dcm_error` differs by 0.0002, and `m960` also carries half the
environments (32 per arm against 64) — so seed draw and checkpoint noise are not
separable here. Across all three checkpoints the run costs between -13% and
-35%; no single figure is the answer.

## head-authority20 @ `m999`

`logs/rsl_rl/Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position/2026-08-20_17-59-46_head-authority20`

`dcm-cmd-ref`'s configuration with **`residual_scale` 0.01 -> 0.20** and a
1000-iteration budget; nothing else differs. 64 envs per arm, K = 2 episodes per
env, n = 128 vs 128 after the fixed-K trim (594 episodes finished in total),
reward era **cmd**. Raw rows and report:
`logs/comparisons/2026-08-20_17-59-46_head-authority20_model_999.{csv,log}`.

### Per-step reward rates

| term               | baseline     | policy       | delta      | p (env)      |
| ------------------ | ------------ | ------------ | ---------- | ------------ |
| `dcm_stability`    | 0.012343     | 0.009473     | -23.3%     | 1.88e-30     |
| `recovery_dcm`     | 0.002135     | 0.001817     | -14.9%     | 3.39e-03     |
| `angular_momentum` | -0.000882    | -0.001319    | +49.5%     | 3.16e-08     |
| `foot_slip`        | -0.000007    | -0.000051    | +579.8%    | 2.04e-51     |
| `torque_margin`    | 0.000000     | -0.000001    | --         | 1.00e-11     |
| `upright`          | -0.000073    | -0.000098    | +35.5%     | 6.09e-03     |
| **tracking sum**   | **0.013516** | **0.009822** | **-27.3%** | **7.29e-21** |

### Terminations

| outcome     | baseline | policy | delta  | p (env) |
| ----------- | -------- | ------ | ------ | ------- |
| `time_out`  | 26.6%    | 19.5%  | -7.0pp | 0.203   |
| `fell_over` | 14.1%    | 12.5%  | -1.6pp | 0.730   |
| `collapsed` | 59.4%    | 68.0%  | +8.6pp | 0.181   |

No controller or worker failure occurred on either arm.

### Episode length and hazard

- baseline mean 55.8 s (median env 55.3 s), policy mean 49.4 s (median env 46.8
  s), delta -6.4 s, p (env) = 0.059.
- hazard baseline **0.0160/s** (94 adverse endings over 5859 s), policy
  **0.0204/s** (103 over 5039 s); **hazard ratio 1.274**, p (env) = 0.086.

### `dcm_stability` per-step by episode-length band

| band      | n base | n policy | baseline | policy   | delta  | p (env)  |
| --------- | ------ | -------- | -------- | -------- | ------ | -------- |
| < 30 s    | 29     | 36       | 0.012635 | 0.008788 | -30.4% | 6.54e-26 |
| 30-60 s   | 42     | 47       | 0.012030 | 0.008917 | -25.9% | 1.93e-11 |
| 60-90 s   | 23     | 20       | 0.011965 | 0.010038 | -16.1% | 1.81e-02 |
| survivors | 34     | 25       | 0.012737 | 0.011050 | -13.2% | 1.72e-02 |

The deficit is present in every band, including the 90 s survivors, so it is not
an artefact of the policy's shorter episodes: at matched duration the residual
still scores below the controller it is added to.

## Pre-orientation-fix reference: `2026-08-13_10-37-48_std-floor`

`~/Documents/old_runs/2026-08-13_10-37-48_std-floor` (archived, 31 checkpoints)

This checkpoint now has a paired comparison, run in its own code state
(`c1def76` plus the archived 62-line working-tree diff) so it deliberately
retains the pre-fix reset ordering. At 64 envs per arm, K = 3, the policy's
tracking sum is **-8.7%** (p = 4.3e-07), its fall hazard is **0.76x** baseline,
and survival is **+6.8pp**. The rows and script report live in
`logs/comparisons/2026-08-13_10-37-48_std-floor_model_1499.{csv,log}`.

This is valid only within that old task distribution. Before `c704bf2`
(2026-08-14 18:53), `MCGlobalController::reset()` rebuilt the walking plan
against the pose the robot was about to leave. With `pose_range` yaw over full
+/-pi, many episodes began in a rotated frame and fell within seconds, before
any push and regardless of policy. The archive's README records 813/2328 such
failures at `|yaw| >= 1.18` before the fix, versus 0/1920 after (p = 4.6e-121).
Its checkpoint also predates `base_controller_provenance`, and its observation
space was 284 rather than the current 1219. Do not use its absolute rates or its
delta as evidence about post-fix policies.

### Training-side scalars, against its post-fix namesake

Same nominal configuration (`std-floor`: gamma 0.99, scale 0.01, 5x4), 34 hours
and one bug apart:

|                                    | pre-fix `08-13_10-37-48` | post-fix `08-14_19-14-46` |
| ---------------------------------- | ------------------------ | ------------------------- |
| iterations logged                  | 1447                     | 3321                      |
| `Train/mean_episode_length` median | 1753 steps (**35.1 s**)  | 2589 steps (**51.8 s**)   |
| smoothed `zmp_error` peak / final  | 0.0615 @ 1288 / 0.0630   | 0.0523 @ 3036 / 0.0547    |
| `lr` floored                       | 51.1%                    | 54.0%                     |
| `lr` median                        | 1.50e-05                 | 1.00e-05                  |
| `Policy/mean_std` median           | 0.065                    | 0.050                     |
| `Loss/value` median                | 0.028                    | 0.020                     |

**The episode length is the contamination, and it is 48% of the metric.** Both
runs sit on the same pinned learning rate and the same objective, so the gap is
not learning: it is episodes ending early in a rotated frame. Terminations over
the last 50 iterations of the pre-fix run read `collapsed` 0.70, `fell_over`
0.23, `time_out` 0.16 — against a post-fix `collapsed` median of 0.67 with
`fell_over` at 0.

**What it is good for:** calibrating how large the pre-fix distortion was, and
nothing else. An effect of the size this record deals in (+/-5% on a per-step
rate) is far smaller than the 48% episode-length shift this bug produced, which
is why the 33 archived runs cannot be pooled with anything above.

## Statistical power: episodes are not independent within an env

`encoder_bias` is a `startup` event, drawn once per env and fixed for the run,
so episodes sharing an env share a bias. Measured on `dcm-obs` `m1850`, the
between-env SD of per-step `dcm_stability` is **0.84-0.87x the within-env SD** —
env identity carries nearly as much variance as episode-to-episode noise. A
comparison's effective sample size therefore tracks its **env count**, not its
episode count.

Resampling `cmp_dcm_1850.csv` (true delta **+5.8%** at 28 envs/arm) down to
smaller env counts, 3000 draws each:

| envs/arm | median delta | 5th pct | 95th pct | sign wrong |
| -------- | ------------ | ------- | -------- | ---------- |
| 4        | +5.5%        | -4.2%   | +17.7%   | 19.3%      |
| 6        | +5.8%        | -2.3%   | +14.7%   | 12.4%      |
| 8        | +5.5%        | -1.1%   | +12.8%   | 8.9%       |
| 12       | +5.6%        | +0.7%   | +10.9%   | 3.2%       |
| 16       | +5.6%        | +1.8%   | +9.5%    | 0.8%       |
| 28       | +5.6%        | +5.6%   | +5.6%    | 0.0%       |

**Floor: 16 envs per arm** (1.7% sign error). Comparisons below that are marked
⚠ in the summary and should not be used to decide anything on their own.

## Baseline hazard is flat, so the 90 s cap is not a ceiling

Pooled zero-residual episodes from four dumps, n = 1119, 21.7% reached the cap,
mean 52.3 s, median 48.9 s. Hazard per 10 s bin:

| window  | at risk | falls | hazard/s | 95% CI            |
| ------- | ------- | ----- | -------- | ----------------- |
| 0-10 s  | 1119    | 1     | 0.0001   | [-0.0001, 0.0003] |
| 10-20 s | 1118    | 175   | 0.0157   | [0.0133, 0.0180]  |
| 20-30 s | 943     | 146   | 0.0155   | [0.0130, 0.0180]  |
| 30-40 s | 797     | 132   | 0.0166   | [0.0137, 0.0194]  |
| 40-50 s | 665     | 125   | 0.0188   | [0.0155, 0.0221]  |
| 50-60 s | 540     | 99    | 0.0183   | [0.0147, 0.0219]  |
| 60-70 s | 441     | 80    | 0.0181   | [0.0142, 0.0221]  |
| 70-80 s | 361     | 63    | 0.0175   | [0.0131, 0.0218]  |
| 80-90 s | 298     | 55    | 0.0185   | [0.0136, 0.0233]  |

All CIs overlap past the warm-up: the process is memoryless, so a longer episode
shows nothing new. Constant-hazard fit **0.0185/s** (mean life 54 s) predicts
22.8% survival at the 90 s cap, 13.1% at 120 s, 4.3% at 180 s — a longer cap
only makes survival rarer and comparisons noisier.

## Training-side scalars

`peak iter` is the argmin of a 60-iteration smoothed `dcm_error` where logged,
else `zmp_error`. The warm-up skipped before searching scales with run length,
so a short run is not scored on a window that discards a fifth of it.

| run               | iters | lr floored | lr median | mean_std | Loss/value | peak metric        | peak iter | final  |
| ----------------- | ----- | ---------- | --------- | -------- | ---------- | ------------------ | --------- | ------ |
| std-floor         | 3320  | 54.0%      | 1.00e-05  | 0.050    | 0.020      | `zmp_error` 0.0523 | 3036      | 0.0547 |
| scale03-g997      | 1150  | 22.5%      | 3.37e-05  | 0.068    | 0.080      | `zmp_error` 0.0787 | 1089      | 0.0863 |
| scale01-g997      | 1261  | 19.4%      | 3.37e-05  | 0.086    | 0.101      | `zmp_error` 0.0568 | 900       | 0.0582 |
| zeroinit-4ev      | 2999  | 1.4%       | 2.96e-04  | 0.120    | 20.975     | `zmp_error` 0.0607 | 1320      | 0.0625 |
| dcm-obs           | 1999  | 0.0%       | 2.96e-04  | 0.136    | 0.122      | `dcm_error` 0.0529 | 1871      | 0.0542 |
| dcm-obs-gate      | 1506  | 0.0%       | 4.44e-04  | 0.149    | 0.116      | `dcm_error` 0.0527 | 267       | 0.0546 |
| gate-torque-phase | 1499  | 0.0%       | 4.44e-04  | 0.157    | 0.111      | `dcm_error` 0.0540 | 227       | 0.0555 |
| pruned-nogate     | 499   | 0.0%       | 4.44e-04  | 0.127    | 0.129      | `dcm_error` 0.0535 | 300       | 0.0562 |
| dcm-cmd-ref       | 499   | 0.0%       | 1.00e-03  | 0.186    | 0.445      | `dcm_error` 0.0449 | 67        | 0.0474 |
| head-authority20  | 999   | 0.0%       | 6.67e-04  | 0.063    | 0.241      | `dcm_error` 0.1276 | 963       | 0.1278 |

`zeroinit-4ev`'s `Loss/value` tail of ~21 is real: it diverged at iteration
~2945.

`dcm_error` became command-relative on 2026-08-19, so the last two rows'
magnitudes compare with each other and not with the rows above them. Against
that comparator `head-authority20` is **2.8x worse** (0.1276 against
`dcm-cmd-ref`'s 0.0449) and was already at 0.152 by iteration 80 — the deficit
is there from the first updates, not drifted into. Its `mean_std` tail of 0.063
is the lowest of any run: the policy narrowed onto the wide action range rather
than exploring it.

## Provenance

| run               | commit    | scored via   |
| ----------------- | --------- | ------------ |
| std-floor         | `9489dbf` | `wt_9489dbf` |
| scale03-g997      | `21b60de` | (native)     |
| scale01-g997      | `1336974` | `wt_1336974` |
| zeroinit-4ev      | `d144762` | `oldcfg`     |
| dcm-obs           | `02e32c9` | (native)     |
| dcm-obs-gate      | `7d99fe9` | `wt_gate`    |
| gate-torque-phase | `6139099` | (native)     |
| pruned-nogate     | `0c81ab6` | (native)     |
| dcm-cmd-ref       | `7b58590` | (native)     |
| head-authority20  | `c386cd3` | (native)     |
| residual-growth screens | run snapshot | (native) |
| recovery-only-005 | `dd1df68` | (native) |
| recovery-only-010 | `0ec55e9` | (native) |
| recovery010-full | `fa316b6` | (native) |

Checkpoints are scored under the code state they trained in, via a git worktree
with `PYTHONPATH` shadowing. Two traps: **prepend** to `PYTHONPATH` (the mc_rtc
bindings arrive on it), and **copy `etc/mc_rtc.yaml` into the worktree** — it is
untracked, so a worktree otherwise gets the committed `MainRobot: JVRC1` and the
checkpoint fails to load with a shape mismatch that reads like a stale
checkpoint.

Runs are located by directory name across `logs/rsl_rl/*/`, because
`experiment_name` changed from `mc_rtc_residual_balance` to the task id on
2026-08-19. The older runs have since been consolidated under the task-id
parent, which is what the paths above give.

All listed runs postdate `c704bf2` (2026-08-14 18:53), the reset-ordering fix. The 33
runs predating it are archived in `~/Documents/old_runs` with their own README.

## What the record supports

1. **The DCM objective is the only change with a measured positive effect**, and
   it survives pruning. `dcm-obs` `m1850` +6.1% (p = 4e-03); `pruned-nogate`
   `m499` **+5.4% (p = 0.033)** at full K — the same effect from a run **3.7x
   shorter**.

2. **The two agreement rewards were pure cost.** Deleting `zmp_tracking` and
   `com_velocity_tracking` — 8.3% of the dense signal — left the objective delta
   unchanged, and `com_velocity_error` held at 0.037-0.039 against a ~0.035
   baseline with no trend.

3. **The gate failed its own falsification test.** `gate_mean` drifted +0.008
   over 1500 iterations and the tracking aggregate went +6.1% -> -0.5% / -0.4%.

4. **`torque_margin` is free and worth keeping** — exactly 0.000000 everywhere.

5. **Nominal DCM payment drives much of the residual growth.** It climbed in
   every earlier objective, but moving all DCM payment onto recovery windows cut
   deterministic magnitude 67.6--82.3% in the two short screens; the widened
   screen's last-60 policy-mean slope was negative. PPO entropy, learned std and
   a global threefold magnitude penalty did not suppress it safely.

6. **Low env counts produced wrong answers, not just noisy ones.** Re-measuring
   the four sub-16-env comparisons at 28 envs/arm moved `std-floor` `m3050` from
   -10.1% (p = 2e-06) to -0.9% (p = 0.37) and flipped `fell_over` signs on three
   of four.

7. **The training metric picks the wrong checkpoint.** `pruned-nogate` peaked on
   smoothed `dcm_error` near iteration 300, but the deterministic objective win
   is at 499 (+4.8%, p = 0.013) while `m250` is null. Bracket; do not trust the
   curve.

8. **`pruned-nogate` was cut short of its peak** — still improving at its
   500-iteration cap, which was chosen on the old early-peak pattern rather than
   measured.

9. **More residual authority is actively harmful.** `head-authority20` raised
   `residual_scale` 0.01 -> 0.20 with nothing else changed and lost **-27.3% of
   the per-step tracking sum (p = 7e-21)**, with hazard 1.27x its own baseline
   and 6.4 s shorter episodes. The deficit holds in every episode-length band
   including the 90 s survivors, so it is not the shorter episodes doing it.
   This closes the direction 0.03 opened (`scale03-g997`, -20.6%): authority is
   not what the policy lacks. Three checkpoints agree on the sign and disagree
   on the size: `m500` -34.7%, `m960` -12.8%, `m999` -27.3%, all with p < 1e-04.
   Read the cost as a range, and note that the halfway bracket is the worst of
   the three — the damage is there early and training only partly walks it back.

10. **The `torque_margin` curriculum never fired in any completed run.** Its
    stage thresholds were written as
    `iterations * num_steps_per_env * num_envs`, but `common_step_counter`
    increments once per `step()` and is **not** scaled by `num_envs`, so stage 1
    needed 64,000 iterations. `Curriculum/torque_margin_weight` holds -0.2 for
    the whole of `gate-torque-phase` and `pruned-nogate`. Fixed in `65123b4`. No
    result is affected: `torque_margin` measured exactly 0 throughout.

11. **A wider recovery kernel can teach a useful residual.** At model 340 it
    improved recovery DCM 7.43%, cut hazard to 0.69 and raised survival 15.6 pp
    (p=0.0359). The old 0.05 kernel had made the measured 0.10 m recovery peak
    pay only 0.018 and failed its own recovery endpoint.

12. **The remaining failure is overtraining-driven residual regrowth.** The
    full recovery-0.10 run moved from magnitude cost -0.0000230 at model 180 to
    -0.0000866 at 340 and -0.000161 at 499. The useful middle checkpoint had
    already exceeded `rg-ref01`; by the end hazard reversed to 1.16. The
    registered two-of-three gate therefore rejected the configuration despite
    model 340's behavioral result.
