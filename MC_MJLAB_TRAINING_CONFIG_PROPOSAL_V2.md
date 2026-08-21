# mc_mjlab residual-growth training proposal V2

**Status:** closed without promotion, 2026-08-21. The four screens and paired
evaluations are recorded in `docs/residual-growth.md`. This supersedes the
decisions in `MC_MJLAB_TRAINING_CONFIG_PROPOSAL.md` without deleting that
historical proposal.

## Decision

Restore the position residual scale to `0.01` and freeze the rest of the current
command-relative DCM configuration. Diagnose residual growth with single-factor
screens before adopting another PPO or reward change. The selected setting must
reduce the deterministic residual without giving back the tracking improvement;
training curves alone cannot select it.

The `0.20` authority experiment is closed. Against its own zero-residual arm it
lost 13--35% of per-step tracking across three checkpoints. The earlier `0.03`
experiment was harmful too. More physical authority is not the missing input,
so every arm below uses `residual_scale = 0.01`.

## Fixed reference

Every screen keeps the following values. They are not part of this experiment.

| Parameter | Value |
| --- | --- |
| environments | 128 |
| controller workers | default, 30 on this 32-core host |
| rollout | 256 policy steps per environment |
| actor/critic | `(512, 256, 128)`, ELU, normalized observations |
| actor mean initialization | zero |
| initial/ranged standard deviation | `0.1`, clamped to `(0.05, 0.30)` |
| PPO updates | 2 epochs x 2 minibatches |
| learning rate | `1e-3`, adaptive, desired KL `0.02` |
| discount/GAE | `gamma = 0.997`, `lambda = 0.99` |
| position residual scale and clip | `0.01 rad` |
| residual-rate weight | `-0.1` |
| objective | current pruned command-relative DCM reward set |
| observations, pushes and terminations | current values, unchanged |

The gate and the agreement rewards stay pruned. Value clipping, rollout length,
GAE, learning rate, network architecture and observation changes are deferred
until this program closes the residual-growth question. The roots and confidence
for every fixed value are recorded in `docs/training-parameter-provenance.md`.

## Required diagnostics

Training must log the following rollout quantities under `Diagnostics/`:

| Scalar | Question answered |
| --- | --- |
| `policy_mean_rms` | Is the learned deterministic correction growing? |
| `sampled_action_rms` | How large is the action applied during training? |
| `exploration_rms` | How much of that action is sampling noise? |
| `policy_mean_rate_rms` | Is deterministic feedback becoming more abrupt? |
| `sampled_action_rate_rms` | How much total temporal jitter reaches the action term? |
| `policy_mean_saturation` | Is the deterministic policy reaching the raw clip? |
| `action_saturation` | Are stochastic samples reaching the raw clip? |

Rate diagnostics exclude transitions across episode resets. Keep the existing
KL, clip-fraction and explained-variance diagnostics. Read these scalars instead
of `Episode_Reward/residual_magnitude` when diagnosing growth: episode sums move
with episode length and stochastic training actions mix policy mean with noise.

## Screening matrix

Run one seed for each arm. A screen uses the full 128 environments and 30 worker
processes for `48,128` policy steps per environment: 188 iterations at the fixed
256-step rollout. Shortening the per-environment budget must not reduce the
environment or worker count.

| Run name | `entropy_coef` | std learning | magnitude weight | Hypothesis |
| --- | ---: | --- | ---: | --- |
| `rg-ref01` | `0.0005` | enabled | `-0.1` | restored reference |
| `rg-entropy0` | `0.0` | enabled | `-0.1` | explicit entropy pressure grows exploration |
| `rg-fixedstd010` | `0.0` | disabled at `0.1` | `-0.1` | learned variance grows independently of entropy |
| `rg-mag03` | `0.0005` | enabled | `-0.3` | the deterministic action is under-regularized |

The fixed-standard-deviation arm sets entropy to zero because entropy is constant
when the standard deviation cannot learn; retaining a nonzero coefficient would
not change its gradient. The `-0.3` magnitude weight is a threefold causal probe,
not a paper default or a proposed final value. At `pruned-nogate` model 499 the
deterministic tracking gain was about `+0.00061` per step while the existing
magnitude and rate costs totaled about `-0.00072`, so this arm deliberately makes
the trade-off strong enough to move if insufficient regularization is the cause.

## Screen evaluation and selection

Evaluate iteration 187 or the final saved checkpoint from every arm against a
zero residual. Use 64 environments split evenly between policy and baseline, 30
workers, and a wall-clock budget long enough to retain at least four episodes per
environment where practical. Do not run evaluation beside training.

Reanalyse episode rows by environment: each environment contributes one mean
over its retained episodes. Report tracking sum without residual penalties,
residual magnitude and rate per step, fall hazard, episode duration, all
termination labels, DCM error, foot slip, upright cost, and controller/worker
failures.

An arm is eligible for promotion only if:

1. its last-60-iteration mean `policy_mean_rms` is at least 20% below
   `rg-ref01`, or its linear growth slope is at least 50% smaller;
2. its deterministic tracking delta against its own zero-residual arm is not
   negative;
3. its fall-hazard ratio is at most `1.10`; and
4. worker failures are treated as truncations rather than policy failures.

Rank eligible arms by tracking delta, then lower hazard ratio, then lower
deterministic residual magnitude. Promote exactly one. Do not combine entropy
and magnitude changes automatically. If no arm is eligible, retain `rg-ref01`
and record that this program did not identify a safe suppressor.

## Full run and adoption gate

Train the selected arm from a fresh initialization for `128,000` policy steps
per environment: 500 iterations with 128 environments and 30 workers. Evaluate
checkpoints near iterations 180, 340 and 499. Evaluate the matching checkpoints
from `dcm-cmd-ref` under their recorded controller inputs as the reference; never
bypass checkpoint provenance validation.

Adopt the promoted configuration only if at least two of the three checkpoints:

1. improve tracking over their own zero-residual arm;
2. are within two percentage points of, or better than, the reference tracking
   delta;
3. reduce deterministic residual magnitude by at least 25% against the matched
   reference checkpoint;
4. keep fall-hazard ratio at or below `1.10`; and
5. show no statistically significant degradation in foot slip or upright cost.

No evaluated checkpoint may show a statistically significant tracking loss.
Checkpoint choice comes from the bracketed deterministic comparisons, not the
training-side DCM metric.

If the final arm fails, the production choice is `rg-ref01`: scale `0.01`,
entropy `0.0005`, learned standard deviation and both residual penalties at
`-0.1`. A failed arm is evidence about its hypothesis, not permission to tune
another parameter in the same run.

## Interpretation

| Result | Conclusion |
| --- | --- |
| entropy arms reduce sampled RMS but not policy-mean RMS | entropy explains exploration growth only |
| `rg-entropy0` reduces both and passes tracking gates | explicit entropy pressure is the usable lever |
| fixed std succeeds where zero entropy does not | PPO learns variance growth without entropy pressure |
| `rg-mag03` reduces policy-mean RMS and passes | deterministic action was under-regularized |
| no arm changes policy-mean growth safely | growth follows the optimized objective; inspect the objective before more PPO tuning |

Only the gate-selected result becomes a training default. Exact experimental
values introduced here have a local-measured root; no cited paper determines
`entropy_coef = 0`, fixed `std = 0.1`, or magnitude weight `-0.3` for this task.

## Result

No arm passed both the residual-growth and behavioral gates, so no full run was
launched. `rg-mag03` was the only arm to suppress the policy-mean slope enough,
but it produced tracking -0.28%, hazard ratio 1.139 and foot-slip cost +90.5%
(environment-clustered p=0.000347). The production reference remains scale
`0.01`, entropy `0.0005`, learned std and both residual penalties at `-0.1`.
