# mc_mjlab recovery-width proposal V4

**Status:** closed without adoption, 2026-08-24.

## Diagnosis

The recovery-only `obj-recovery4` screen reduced deterministic residual magnitude
67.6%, confirming that nominal DCM payment drove much of the residual growth. It
nevertheless made recovery DCM 8.14% worse. The next isolated question is whether
the recovery kernel supplies a useful gradient over the errors caused by a push.

The current Gaussian score is `exp(-(error / std)^2)` with `std = 0.05 m`. That
width was selected from nominal walking and deliberately makes a disturbed state
pay almost nothing. The corrected recovery profile later measured mean errors of
0.068--0.100 m during its main peaks and 0.033 m after settling. This has the
wrong learning emphasis for a recovery-only objective:

| Error | Current score, std 0.05 | Current gradient magnitude | Score, std 0.10 | Gradient magnitude |
| ---: | ---: | ---: | ---: | ---: |
| 0.033 m, settled | 0.647 | 17.1/m | 0.897 | 5.92/m |
| 0.100 m, peak | 0.018 | 1.47/m | 0.368 | 7.36/m |
| 0.200 m, severe | 1.13e-7 | 1.80e-5/m | 0.018 | 0.366/m |

Doubling only the recovery width moves gradient away from already-settled steps
and gives a measured 0.10 m peak five times more gradient. It retains the same
bounded Gaussian form and does not broaden nominal `dcm_stability`.

## Screen

Run `obj-recovery010` for 188 iterations, seed 42, 128 environments and 30
workers. It differs from `obj-recovery4` in exactly one value:

| Parameter | `obj-recovery4` | `obj-recovery010` |
| --- | ---: | ---: |
| recovery DCM std | `0.05 m` | `0.10 m` |

Both use nominal/recovery DCM weights `0.0/4.0`; all other reward, action and PPO
settings remain current. The value `0.10` is a measured causal probe centered on
the observed recovery peak, not a proposed production default.

## Decision gate

Apply the same training gate as V3: last-60 `policy_mean_rms` must be at least
20% below `rg-ref01`, or its slope at least 50% smaller. If it passes, evaluate
model 187 deterministically with 64 environments, 30 workers, fixed K and one
mean per environment. Report recovery DCM using `std = 0.10`, not the repository
default used to train the reference.

Promote to 500 iterations only if recovery DCM is non-negative against zero
residual, fall-hazard ratio is at most 1.10, foot slip and upright cost do not
degrade significantly, and worker failures remain truncations. Failure keeps the
current objective and prunes recovery std `0.10`; it does not authorize another
width or kernel search.

## Screen result

`obj-recovery010` completed 188 iterations. Its last-60 `policy_mean_rms` was
`0.02648`, 43.6% below `rg-ref01`, and its slope was `-0.0001046`: deterministic
residual growth reversed over the selection window. Policy-mean rate RMS was
35.5% lower than reference.

The model-187 comparison retained four episodes for each of 32 environments per
arm. Fixed-K, environment-clustered results were:

| Quantity | Policy against zero residual | Environment-clustered p |
| --- | ---: | ---: |
| recovery DCM, std 0.10 | **+1.03%** | 0.807 |
| nominal DCM, std 0.05 | +0.81% | 0.664 |
| fall-hazard ratio | **0.845** | -- |
| episode duration | +3.30 s | 0.344 |
| survival | +6.25 pp | 0.223 |
| foot-slip cost | +23.6% | 0.273 |
| upright cost | -7.21% | 0.495 |
| controller-worker truncations | 0 vs 1 episode | 0.325 |

Deterministic magnitude cost was `-0.0000144` per step at weight `-0.1`, 82.3%
smaller than `rg-ref01`. The recovery score is only a small, unresolved gain,
but non-negative was the registered screen gate; hazard and residual size both
move strongly in the desired direction. The configuration therefore qualifies
for the full run.

## Full-run gate

Train `obj-recovery010-full` from a fresh initialization for 500 iterations with
the same seed, 128 environments and 30 workers. Evaluate models 180, 340 and 499
with the same width-matched deterministic comparison.

Adopt the configuration only if at least two checkpoints have non-negative
recovery DCM, hazard ratio at most 1.10, and no significant foot-slip or upright
degradation. No checkpoint may have a significant recovery loss or hazard above
1.10. Select the checkpoint by recovery DCM, then hazard, while requiring its
deterministic magnitude cost to remain at least 50% below `rg-ref01` model 187.
If the gate fails, keep the current objective and prune recovery std `0.10`.

## Full-run result

`obj-recovery010-full` completed 500 iterations from a fresh initialization.
Fixed-K evaluation retained three episodes per environment for models 180 and
340 and four for model 499, with 32 environments per arm throughout.

| Model | Recovery DCM | Hazard ratio | Foot-slip cost | Upright cost | Magnitude cost/step | Decision |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 180 | -3.70% (p=0.428) | 1.026 | -0.91% (p=0.976) | +7.89% (p=0.594) | -0.0000230 | recovery gate failed |
| 340 | +7.43% (p=0.134) | **0.691** | -16.2% (p=0.323) | -5.94% (p=0.659) | -0.0000866 | residual-size gate failed |
| 499 | +1.88% (p=0.653) | **1.160** | -21.3% (p=0.136) | +9.56% (p=0.456) | -0.000161 | hazard gate failed |

Only model 340 passes the behavioral gates. It also increased survival 15.6
percentage points (environment-clustered p=0.0359) and episode duration 7.94 s
(p=0.0666), so the wider recovery objective can teach a useful correction. But
its deterministic magnitude cost is 6.8% larger than `rg-ref01`, not at least
50% smaller as required. By model 499 the cost is 98.1% larger than reference
and hazard has crossed the hard 1.10 limit.

The two-of-three adoption rule therefore fails, and model 499 independently
violates the rule that no checkpoint may exceed hazard 1.10. Do not adopt the
`0.0/4.0`, recovery-std-`0.10` configuration. The result localizes the remaining
problem: broad recovery shaping can improve recovery, but continued optimization
again grows the deterministic residual until the physical benefit reverses.
Keep the current objective and treat this configuration as pruned.
