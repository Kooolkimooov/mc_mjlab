# mc_mjlab recovery-width proposal V4

**Status:** screening.

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

