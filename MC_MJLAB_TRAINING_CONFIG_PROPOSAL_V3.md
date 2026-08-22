# mc_mjlab recovery-objective proposal V3

**Status:** closed without promotion, 2026-08-22.

## Diagnosis

The residual-growth screen rules out entropy pressure, learned variance and a
threefold magnitude penalty as safe explanations. Its deterministic comparisons
also show that the reference residual earns no objective improvement: relative
to its own zero-residual arm, `dcm_stability` was -1.3%, `recovery_dcm` was +0.3%
and total reward per step was -1.7%.

The objective is dominated by nominal walking even though the task asks the
residual to recover from pushes. `dcm_stability` is active on every grounded step;
`recovery_dcm` is active for only 2 s after pushes spaced 5--7 s apart. The former
therefore supplies roughly three quarters of their combined DCM payment. A
residual can keep changing ordinary gait to pursue small nominal DCM differences
while the magnitude and rate terms are the only rewards that prefer zero action
there.

The `rg-mag03` comparison shows the resulting proxy conflict most clearly. It
improved nominal `dcm_stability` by 1.3%, but recovery payment fell 5.2%, fall
hazard rose to 1.139 and foot-slip cost rose 90.5% (p = 0.000352 in the original
episode-level report; p = 0.000347 after clustering by environment). A nominal
DCM gain is therefore not a sufficient balance objective.

## Screen

Run one causal screen, `obj-recovery4`, for the same 48,128 policy steps per
environment as `rg-ref01`:

| Parameter | Reference | Screen |
| --- | ---: | ---: |
| `dcm_stability` weight | `1.0` | `0.0` |
| `recovery_dcm` weight | `1.0` | `4.0` |
| all PPO, action and other reward settings | current | unchanged |

Four is a scale-preserving probe, not an adopted constant. A 2 s window on a
mean 6 s push cadence has about one-third duty, so the current average DCM
coefficient is `1 + 1/3 = 4/3`; the screen's is `4 * 1/3 = 4/3`. It moves the
same approximate payment onto exogenous recovery windows instead of making a
simultaneous reward-scale change.

Outside a recovery window the only action-dependent dense terms that remain are
penalties and physical guards. The expected optimum there is therefore a zero
residual. The push schedule, last-push velocity and recovery age are already in
the observations, so the policy has the information needed to separate the two
regimes.

## Decision gate

Evaluate model 187 deterministically against its own zero-residual arm with the
same 64 environments, 30 workers and fixed-K environment-clustered analysis as
the residual-growth screen. Promote to 500 iterations only if all of these hold:

1. last-60 `policy_mean_rms` is at least 20% below `rg-ref01`, or its slope is at
   least 50% smaller;
2. `recovery_dcm` per step is non-negative against baseline;
3. fall-hazard ratio is at most 1.10;
4. neither foot slip nor upright cost degrades significantly; and
5. controller-worker failures remain truncations.

Report nominal `dcm_stability` as a diagnostic even though its weight is zero.
If the reward manager omits zero-weight terms from episode output, use
`dcm_error / zmp_grounded` instead. Failure retains the current objective and
closes recovery-only shaping rather than triggering another unregistered tune.

## Result

`obj-recovery4` ran for 188 iterations with seed 42, 128 environments and 30
workers. It passed the training-side action gate: last-60 `policy_mean_rms` was
`0.02847`, 39.4% below `rg-ref01`, and its slope was `0.0001165`, 49.1% below
reference. `policy_mean_rate_rms` was 28.8% lower. The tail episode length was
effectively unchanged (+0.5%).

Model 187 was then compared deterministically for 12 minutes with 64 environments
and 30 workers. Fixed-K analysis retained three episodes for each of 32
environments per arm and reduced each environment to one mean. The evaluation
task reports the diagnostic DCM terms at the repository's default weights, not
the training override; relative policy/baseline changes remain the same.

| Quantity | Policy against zero residual | Environment-clustered p |
| --- | ---: | ---: |
| recovery DCM per step | **-8.14%** | 0.201 |
| nominal DCM per step | -1.26% | 0.501 |
| fall-hazard ratio | 0.952 | -- |
| episode duration | +1.54 s | 0.706 |
| foot-slip cost | +30.3% | 0.298 |
| upright cost | +7.36% | 0.571 |
| controller-worker truncations | 1 vs 1 episode | 1.00 |

The deterministic magnitude cost was `-0.0000263` per step at weight `-0.1`,
67.6% smaller than `rg-ref01`'s `-0.0000811`. Recovery-only shaping therefore
did what it was designed to do to the residual size and did not harm fall hazard,
but it failed the decisive gate: the quantity it exclusively optimized became
worse, not better. No 500-iteration run is warranted. Keep the current
`dcm_stability = 1.0`, `recovery_dcm = 1.0` objective and treat `0.0/4.0` as
pruned.
