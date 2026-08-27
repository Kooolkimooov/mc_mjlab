# Critique improvement screens

## CRITIQUE_SCREEN_2026_08_25

**Current:** The fixed-budget matrix completed all 13 arms at seed `42`, `128`
environments, `30` workers, and `188` PPO iterations. Every arm reached
`model_187.pt` with exit code zero and no traceback. Values below are
60-iteration moving-average minima and late-window values from TensorBoard;
`executed_residual_l2` is the mean over the final 20 iterations.

| arm | best ZMP (iteration) | late ZMP | best recovery DCM (iteration) | late recovery DCM | executed residual L2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| standard | 0.035675 (173) | 0.035680 | 0.013182 (142) | 0.013216 | 0.000460 |
| authority-ankle | **0.035626 (138)** | 0.035714 | **0.013121 (89)** | 0.013260 | **0.000154** |
| authority-sagittal | 0.035678 (170) | 0.035694 | 0.013174 (125) | 0.013235 | 0.000250 |
| authority-hardware | 0.035665 (87) | 0.035687 | 0.013172 (107) | 0.013245 | 0.000484 |
| history-10 | 0.035675 (140) | 0.035773 | 0.013161 (139) | 0.013251 | 0.000602 |
| history-5 | 0.035654 (104) | 0.035727 | 0.013156 (164) | 0.013188 | 0.000564 |
| history-gru256 | 0.035708 (87) | 0.035789 | 0.013134 (95) | 0.013290 | 0.000632 |
| PPO fixed 1e-4 | 0.035685 (83) | 0.035866 | 0.013154 (89) | 0.013206 | 0.000798 |
| PPO fixed 3e-4 | 0.035739 (137) | 0.035798 | 0.013135 (107) | 0.013268 | 0.000465 |
| PPO fixed 1e-3 | 0.035666 (153) | 0.035793 | 0.013171 (183) | 0.013219 | 0.000374 |
| PPO epochs 5x4 | 0.035735 (87) | 0.035813 | 0.013199 (89) | 0.013276 | 0.000970 |
| PPO unclipped | 0.035697 (154) | 0.035791 | 0.013169 (159) | 0.013264 | 0.003230 |
| position-velocity | 0.035733 (174) | 0.035815 | 0.013187 (178) | 0.013226 | 0.000654 |

These are stochastic training diagnostics, not policy-versus-controller
evidence. They only shortlist paired deterministic evaluation. The PPO,
history, and GRU variants do not advance: none improves both tracking measures,
several peak early and decay, and unclipped PPO uses seven times the standard
residual. The ankle authority arm advances because it has the best smoothed ZMP
and recovery DCM values with one-third of the standard residual. Qualify
`authority-ankle/model_140` plus `model_187` as the trajectory's second read,
and `standard/model_180` as the full-authority control. Do not start the
500-iteration confirmation until paired qualification shows an advantage.

Paired deterministic qualification completed with `16` environments, two
episodes per arm per environment, seed `42`, and all four scenarios. Negative
recovery delta is better. The interval is the environment-clustered 95% interval
on the absolute recovery DCM difference in metres.

| checkpoint | finite-impulse recovery | 95% interval | hazard ratio | max effort | projection / near-bound | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| standard 180 | -7.70% | [-0.02140, +0.00119] | 1.000 | 0.572 | 0 / 0 | fail: recovery confidence, nominal foot-slip CI, hazard |
| ankle 140 | -4.79% | [-0.02030, +0.00752] | 1.011 | 0.580 | 0 / 0 | fail: recovery magnitude/confidence and hazard |
| ankle 187 | -5.08% | [-0.01731, +0.00380] | 1.011 | 0.554 | 0 / 0 | fail: recovery confidence and hazard |

No checkpoint is promoted and the 500-iteration confirmation does not run. The
safety and measurement infrastructure remains, but this screen provides no
evidence for changing the policy defaults.

The later `position-velocity` arm also fails promotion. Its best saved
checkpoint, `model_140`, matched baseline survival at 91.7% and changed recovery
and total reward per step by only +0.2% (p=0.887 and p=0.915). Its required late
read, `model_187`, reduced survival from 79.2% to 54.2% and total reward per step
by 5.2% (p=0.437). One policy-side worker failure affected that late read, but
the best read already establishes only parity. Exact results and the fixed-law
precursor are in `walking-reference.md`.

The robust scenario in that original qualification was invalid: baseline and
policy episodes fell at about 6.0 s, before the first 10 s disturbance, because
the inertial-field randomizer changed HRP5P dynamics even at zero perturbation.
Its measurements are not evidence about recovery. This does not reverse the
decision above: the finite-impulse recovery and nominal-gait gates independently
reject all three checkpoints. After removing the incompatible inertial and COM
fields, an 8-environment, one-pair, 20 s robust smoke test had zero pre-push
hazard and no worker failures. Its post-push hazard was 0.875 for the baseline
and 1.000 for standard checkpoint 180, so it still supplied no promotion signal.

Quarantined workers died once in standard, once in fixed-1e-4, twice in
fixed-1e-3, once in epochs-5x4, twice in unclipped PPO, and twice in GRU. The
position-velocity arm also lost one worker; the remaining six arms had no worker
death. Late-window
`controller_worker_failed` was zero in every arm, so no screen was discarded;
the paired qualifier remains stricter and invalidates any affected run.

**Re-measure if:** the task, controller, disturbance curriculum, authority
scales, observation contract, or training budget changes.

**History:**
- 2026-08-25 — the position-velocity extension produced parity at its best
  checkpoint and degradation at its late read, so it did not advance.
- 2026-08-25 — paired qualification selected no checkpoint, ending the
  experiment before the long-run stage.
- 2026-08-25 — selected the standard and ankle checkpoints for deterministic
  paired qualification; no policy is promoted from training curves.

## actor_update_fraction

**Current:** the active-authority correction completed its isolated seed-42
comparison in 67 minutes: 188 PPO iterations, 128 environments, 30 controller
workers, 48,128 policy steps per environment, and 6,160,384 total transitions.
It reached `model_187.pt` without an OOM, controller failure, or worker failure.
The actor surrogate used 12.553% of rollout samples on average and 12.296% over
the final 60 iterations, so active samples were no longer diluted by the roughly
87.4% inactive cohort.

The comparison below uses the original seed-42 ankle screen as its control.
Best and late tracking values are 60-iteration means; residual is the final
20-iteration mean. Hazard is the mutually exclusive `fell_over + collapsed`
union, averaged over the final 60 training iterations.

| diagnostic | old ankle | active-authority correction | change |
| --- | ---: | ---: | ---: |
| best ZMP error | 0.035626 at 138 | 0.035713 at 84 | +0.25% |
| late ZMP error | 0.035714 | 0.035775 | +0.17% |
| best recovery DCM error | 0.013121 at 89 | 0.013196 at 107 | +0.58% |
| late recovery DCM error | 0.013260 | 0.013289 | +0.22% |
| executed residual L2 | 0.000154 | 0.000254 | +65.5% |
| late hazard | 0.003598 | 0.004167 | +15.8% |
| late foot slip | 0.0001013 | 0.0000948 | -6.3% |
| policy-mean RMS | 0.05473 | 0.05825 | +6.4% |
| policy-mean rate RMS | 0.02044 | 0.03184 | +55.8% |
| action standard deviation | 0.06566 | 0.08058 | +22.7% |

The correction passed its mechanical acceptance test: the surrogate now learns
from effective actions and the resulting policy differs materially. It did not
produce favorable training evidence. It used more residual, changed faster, and
was slightly worse on both stability errors and training hazard; only foot slip
improved. These curves still cannot establish controller-relative performance.
`model_100.pt` brackets the best smoothed ZMP and recovery readings at iterations
84 and 107 and predates the observed collapse samples, so it is the single
checkpoint worth a paired deterministic triage. Do not spend a second training
seed unless that comparison beats the controller baseline.

**Re-measure if:** actor masking, requested-action pricing, recovery authority,
rollout length, or the ankle action set changes.

**History:**
- 2026-08-27 — completed the isolated active-authority seed; objective plumbing
  passed, while training diagnostics did not support promotion.
