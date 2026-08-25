# Critique improvement screens

## CRITIQUE_SCREEN_2026_08_25

**Current:** The fixed-budget matrix completed all 12 arms at seed `42`, `128`
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

These are stochastic training diagnostics, not policy-versus-controller
evidence. They only shortlist paired deterministic evaluation. The PPO,
history, and GRU variants do not advance: none improves both tracking measures,
several peak early and decay, and unclipped PPO uses seven times the standard
residual. The ankle authority arm advances because it has the best smoothed ZMP
and recovery DCM values with one-third of the standard residual. Qualify
`authority-ankle/model_140` plus `model_187` as the trajectory's second read,
and `standard/model_180` as the full-authority control. Do not start the
500-iteration confirmation until paired qualification shows an advantage.

Quarantined workers died once in standard, once in fixed-1e-4, twice in
fixed-1e-3, once in epochs-5x4, twice in unclipped PPO, and twice in GRU. The
remaining seven arms had no worker death. Late-window
`controller_worker_failed` was zero in every arm, so no screen was discarded;
the paired qualifier remains stricter and invalidates any affected run.

**Re-measure if:** the task, controller, disturbance curriculum, authority
scales, observation contract, or training budget changes.

**History:**
- 2026-08-25 — selected the standard and ankle checkpoints for deterministic
  paired qualification; no policy is promoted from training curves.
