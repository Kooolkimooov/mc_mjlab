# Proposed changes to the `mc_mjlab` training configuration

Date: 2026-08-19

Target repository: `/home/martin/git/mc_mjlab`

Status: proposal only; no `mc_mjlab` source or configuration files were changed while preparing this document.

## Executive recommendation

Make three correctness changes before further hyperparameter tuning:

1. Replace the current zero-target DCM reward with a command-relative DCM reward.
2. Separate genuine controller/QP failures from controller-worker process failures, treating the latter as bootstrapped truncations rather than penalized terminal states.
3. Make `fell_over` and `collapsed` mutually exclusive diagnostic outcomes while preserving their existing termination union.

After those fixes, use a fixed reference configuration and test PPO changes one at a time. The first PPO ablation should disable value-loss clipping. Rollout length and GAE lambda must then be isolated, followed by an entropy ablation. Do not change the network architecture yet.

## Evidence from the local repository

The current PPO configuration is defined in:

- `/home/martin/git/mc_mjlab/src/mc_mjlab/tasks/residual_balance/residual_balance_ppo_cfg.py`
- `/home/martin/git/mc_mjlab/src/mc_mjlab/tasks/residual_balance/residual_balance_env_cfg.py`

Its material settings are:

| Setting | Current value |
|---|---:|
| actor and critic widths | 512, 256, 128 |
| actor initialization | zero-initialized mean |
| initial Gaussian standard deviation | 0.1 |
| standard-deviation bounds | 0.05 to 0.30 |
| value-loss clipping | enabled |
| policy clip | 0.2 |
| entropy coefficient | 0.0005 |
| epochs x minibatches | 2 x 2 |
| learning rate | 0.001, adaptive |
| discount `gamma` | 0.997 |
| GAE `lambda` | 0.99 |
| rollout length | 256 steps/environment |
| iterations | 500 |

The repository's deterministic evaluations in `/home/martin/git/mc_mjlab/RUN_COMPARISONS.md` show:

- Replacing plan-agreement rewards with the measured DCM objective was the only objective change with a reproducible positive effect.
- The directional-coherence gate failed its falsification test and should remain disabled.
- Removing the plan-agreement rewards was beneficial; they should stay removed.
- The torque-margin term was exactly zero during the evaluated runs. It can remain as a safety guard, but it is not currently a useful learning signal.
- Residual magnitude increased across all eight recorded runs and remains unexplained.
- Training metrics selected the wrong checkpoint for `pruned-nogate`; deterministic checkpoint evaluation is required.
- Evaluations with too few environments produced incorrect effect signs. The established comparison size is 28 environments per arm.
- `fell_over` and `collapsed` are evaluated independently, so a low, tilted robot can be counted in both termination categories even though the terminal penalty is applied only once.

The active `audit-fixes-500` run was at iteration 467/500 when inspected. It changed both rollout length and GAE lambda relative to `pruned-nogate`, so it cannot attribute an outcome to either change independently.

## 1. Correct the DCM objective

### Problem

`mdp.dcm_offset` currently returns

```text
|| (CoM velocity / omega) - (CoP - CoM) ||
```

which is equivalent to `||xi - CoP||` under the LIPM definition

```text
xi = CoM + CoM velocity / omega.
```

That quantity is the DCM divergence-rate magnitude. It is meaningful, but minimizing it toward zero favors zero velocity. A stable commanded walk normally has a nonzero DCM-to-CoP offset.

The local zero-residual audit in `/home/martin/git/mc_mjlab/docs/reward-shaping.md` measured:

| Motion | Current DCM score |
|---|---:|
| walking | 0.5702 |
| standing | 0.8923 |

Standing therefore receives a 56.5% advantage from a term that supplies approximately 95% of the current dense reward.

### Proposed implementation

Score the error relative to the horizontal CoM velocity commanded by the reference controller:

```python
omega = sqrt(gravity / com_height)
capture_offset = com_velocity_xy / omega
measured_offset = cop_xy - com_xy
expected_offset = command_com_velocity_xy / omega
error = norm((capture_offset - measured_offset) - expected_offset)
reward = exp(-square(error / std))
```

Use the corrected error for both `dcm_stability` and `recovery_dcm`.

The command/reference velocity must be independent of the residual action. If the available reference is phase-dependent, prefer that reference over a constant nominal walking speed.

### Validation gate

Before training:

1. Replay zero-residual standing, nominal walking, and pushed walking trajectories.
2. Confirm that nominal standing and walking have similar scores; a suggested acceptance band is within 10%.
3. Confirm that an external push increases corrected DCM error and that recovery decreases it.
4. Confirm that the residual cannot improve the reward merely by changing the target quantity.
5. Re-estimate `DCM_STD` from the nominal corrected-error distribution rather than retaining `0.05` automatically. For the current squared exponential kernel, a desired score `s` at error quantile `q` implies `std = q / sqrt(-log(s))`.

The DCM dynamics underlying this correction are reviewed in [Zhang et al., 2022](2022_Zhang_RobustWalkingDCM/2022_Zhang_RobustWalkingDCM.pdf).

## 2. Correct worker-failure termination semantics

### Problem

When a controller worker dies, `mc_rtc_controller_pool.py` marks all environments assigned to that worker as failed. The task maps that status to `controller_failed`, a normal terminal condition. The generic termination reward then assigns `-200` to every affected environment.

These failures are correlated across environments and unrelated to the policy's action. Training on them as genuine terminal transitions gives the learner a false penalty and prevents value bootstrapping.

### Proposed implementation

Expose distinct status values for:

- `controller_qp_failed`: a real controller failure; terminal and penalized.
- `controller_worker_failed`: an infrastructure failure; reset as a timeout/truncation, bootstrap the value target, and do not apply the termination penalty.

Log both counts separately. Until the distinction exists, controller-worker crash counts must accompany every training comparison.

This treatment follows the exogenous time-limit/truncation analysis in [Pardo et al., 2018](2018_Pardo_TimeLimitsRL/2018_Pardo_TimeLimitsRL.pdf).

## 3. Make balance-failure labels mutually exclusive

### Problem

`fell_over` currently checks whether trunk tilt exceeds 45 degrees, while `collapsed` independently checks whether root height is below 70% of nominal height. A toppled robot is often both tilted and low, so one episode can increment both diagnostic counters.

The termination manager ORs all non-timeout predicates, and `termination_penalty` reads that aggregate. The overlap therefore does not double the `-200` penalty or change reset timing, but it makes the per-cause percentages non-additive and obscures whether a policy trades crouch-collapse for toppling.

### Proposed implementation

Give `fell_over` priority and reserve `collapsed` for the upright crouch-collapse that the height predicate was introduced to detect:

```python
fell_over = bad_orientation(env, limit_angle)
collapsed_raw = root_height_below_minimum(env, minimum_height)
collapsed = collapsed_raw & ~fell_over
```

This is a diagnostic reclassification rather than a change to the learned task:

```text
old termination = fell_over OR collapsed_raw
new termination = fell_over OR (collapsed_raw AND NOT fell_over)
                = fell_over OR collapsed_raw
```

Consequently, episode reset timing, value targets, the generic terminal penalty, fall hazard, and survival should remain unchanged.

### Validation gate

1. Assert that `fell_over & collapsed` is always false.
2. Assert on recorded or replayed states that the old and new termination unions are identical.
3. Continue computing fall hazard from the union of balance-failure predicates, not by summing category counts.
4. Annotate historical `fell_over` and `collapsed` results as non-exclusive; do not compare their absolute shares directly with the new exclusive labels without re-evaluation.
5. Keep infrastructure and QP failures separate from this balance-failure taxonomy. If a single categorical outcome is later required, define explicit precedence across controller failure, `fell_over`, `collapsed`, and timeout.

## 4. PPO ablation sequence

After the correctness fixes, create a fresh reference run using the otherwise-current PPO configuration. Every subsequent comparison should change only one factor.

### 4.1 Disable value-loss clipping first

Proposed change:

```python
use_clipped_value_loss = False
```

Keep policy clipping at `0.2`. Value-loss clipping is not required by PPO's policy objective, operates in task-specific value units, and was harmful across all five continuous-control domains in the large implementation study by [Andrychowicz et al., 2021](2021_Andrychowicz_WhatMattersOnPolicyRL/2021_Andrychowicz_WhatMattersOnPolicyRL.pdf).

Promote this change only if deterministic evaluation improves the corrected DCM objective without worsening fall hazard or episode length.

### 4.2 Isolate GAE lambda and rollout length

The previous configuration used `lambda=0.95` and a 96-step rollout. The current configuration uses `lambda=0.99` and a 256-step rollout. Because both changed together, retain neither conclusion without a controlled comparison.

Use the following factorial screen if the current and previous corners do not provide a decisive result:

| Run | GAE lambda | Rollout steps |
|---|---:|---:|
| A | 0.95 | 96 |
| B | 0.99 | 96 |
| C | 0.95 | 256 |
| D | 0.99 | 256 |

Keep `gamma=0.997`. GAE lambda controls a bias-variance tradeoff, so it should be selected empirically for this task rather than by increasing it automatically; see [Schulman et al., 2016](2016_Schulman_GeneralizedAdvantageEstimation/2016_Schulman_GeneralizedAdvantageEstimation.pdf).

### 4.3 Remove explicit entropy pressure

Next proposed one-factor test:

```python
entropy_coef = 0.0
```

Retain the learnable Gaussian standard deviation, initial value `0.1`, and bounds `0.05` to `0.30` in this test. The current standard deviation rose to approximately 0.18--0.19 and residual magnitude increased, so additional entropy pressure may be unnecessary for a residual policy whose exploration is itself a physical disturbance.

If standard deviation still rises without deterministic benefit, run a second, separate test with fixed `std=0.1`. Do not combine the zero-entropy and fixed-standard-deviation changes in the first comparison.

### 4.4 Test a fixed learning rate only after adding diagnostics

Add the following training diagnostics first:

- mean approximate KL divergence;
- policy clip fraction;
- explained variance;
- action saturation fraction;
- controller QP failures and worker failures as separate counters.

Then test:

```python
schedule = "fixed"
learning_rate = 3.0e-4
```

The current adaptive schedule can update the learning rate at every minibatch, but the run does not log the KL statistic driving those changes. Without that measurement, learning-rate movement is difficult to diagnose. Treat `3e-4` as an ablation, not an assumed final value.

## 5. Define budgets in transitions, not iterations

Iteration counts are not comparable when rollout length changes:

| Configuration | Calculation | Steps/environment |
|---|---:|---:|
| previous | 96 x 500 | 48,000 |
| current | 256 x 500 | 128,000 |
| matched short run | 256 x 188 | 48,128 |

For screening, use approximately 48,000 policy steps per environment: 500 iterations with rollout 96 or 188 iterations with rollout 256. Promote only promising settings to the 128,000-step budget.

Store both policy steps per environment and total transitions in run metadata. Derive `max_iterations` from the chosen budget.

For the already-running `audit-fixes-500`, deterministically evaluate checkpoints `m180`, `m200`, `m300`, `m440`, and the final checkpoint. `m180` and `m200` bracket the previous 48,000-step budget.

## 6. Evaluation protocol

For each comparison:

1. Use deterministic evaluation with the established 28 environments per arm.
2. Report corrected DCM error and recovery DCM as primary task metrics.
3. Also report fall hazard, episode length, velocity error, residual magnitude, residual rate, and controller failure counts.
4. Do not select a checkpoint using stochastic training reward alone.
5. For a final candidate, train at least seeds 42, 43, and 44.
6. Treat training seeds, not episodes from one trained policy, as independent training replicates.

The exact number of seeds needed depends on observed between-seed variance and the effect size. Use the power-analysis procedure described by [Colas et al., 2018](2018_Colas_HowManyRandomSeeds/2018_Colas_HowManyRandomSeeds.pdf) before making a final performance claim.

## 7. Settings to keep for now

Retain until contrary task-specific evidence is obtained:

- zero-initialized actor mean;
- actor and critic widths `(512, 256, 128)` with ELU;
- observation normalization;
- initial Gaussian standard deviation `0.1` and current bounds;
- policy clip `0.2`;
- `gamma=0.997`;
- two learning epochs and two minibatches;
- residual action scale and existing action magnitude/rate penalties;
- removed plan-agreement rewards;
- directional-coherence gate disabled;
- torque-margin safety guard and its fixed curriculum implementation.

Do not add symmetry augmentation until the DCM objective and failure semantics are corrected. It requires a verified left-right mapping for every actor observation, critic observation, history channel, and residual action, and the reference controller may contain real phase asymmetries.

## 8. Recommended run order

1. Implement and validate command-relative DCM scoring without training.
2. Split QP failure from worker failure and verify timeout bootstrapping.
3. Make `fell_over` and `collapsed` mutually exclusive and verify that their union is unchanged.
4. Train the corrected reference configuration.
5. Run the no-value-clipping comparison.
6. Screen the missing lambda/rollout corners at a matched transition budget if needed.
7. Run the zero-entropy comparison on the best preceding configuration.
8. Add PPO diagnostics and, only then, test fixed learning rate `3e-4`.
9. Retrain the final candidate with multiple seeds and conduct the full deterministic evaluation.

## Sources

| Topic | Source | DOI |
|---|---|---|
| DCM dynamics and walking | [Zhang et al., *Robust Walking for Humanoid Robot Based on Divergent Component of Motion*](2022_Zhang_RobustWalkingDCM/2022_Zhang_RobustWalkingDCM.pdf) | [10.3390/mi13071095](https://doi.org/10.3390/mi13071095) |
| Exogenous truncations and bootstrapping | [Pardo et al., *Time Limits in Reinforcement Learning*](2018_Pardo_TimeLimitsRL/2018_Pardo_TimeLimitsRL.pdf) | [10.48550/arXiv.1712.00378](https://doi.org/10.48550/arXiv.1712.00378) |
| PPO implementation choices | [Andrychowicz et al., *What Matters In On-Policy Reinforcement Learning?*](2021_Andrychowicz_WhatMattersOnPolicyRL/2021_Andrychowicz_WhatMattersOnPolicyRL.pdf) | [10.48550/arXiv.2006.05990](https://doi.org/10.48550/arXiv.2006.05990) |
| GAE bias-variance tradeoff | [Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*](2016_Schulman_GeneralizedAdvantageEstimation/2016_Schulman_GeneralizedAdvantageEstimation.pdf) | [10.48550/arXiv.1506.02438](https://doi.org/10.48550/arXiv.1506.02438) |
| Training-seed power analysis | [Colas et al., *How Many Random Seeds?*](2018_Colas_HowManyRandomSeeds/2018_Colas_HowManyRandomSeeds.pdf) | [10.48550/arXiv.1806.08295](https://doi.org/10.48550/arXiv.1806.08295) |
