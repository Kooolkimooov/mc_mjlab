# Walking-reference modulation

## WALKING_REFERENCE_SCALE

**Current:** `(0.20, 0.15, 0.30)` bounds the recovery-only `vx`, `vy`, and yaw
rate offsets. The command reaches these bounds in at least `0.10 s`, is multiplied
by the calibrated recovery authority, and becomes exactly zero when authority
does. The live nominal reference is restored rather than assuming the installed
controller still commands `(0.1, 0, 0)`.

The first learned screen does not clear promotion. At seed 42, 128 environments,
30 workers, and 188 PPO iterations, the 60-iteration means bottomed at
`0.035733 m` ZMP error (window ending at iteration 174) and `0.013187 m`
recovery DCM error (ending at 178). The final-20-iteration means were
`0.000654` executed joint residual L2 and `0.000152` walking-reference L2.

Paired deterministic reads used 16 environments for eight minutes and kept the
first three episodes per environment and arm:

| checkpoint | baseline / policy survival | recovery reward per-step delta | total reward per-step delta | worker failures |
| --- | ---: | ---: | ---: | ---: |
| `model_140` | 91.7% / 91.7% | +0.2%, p=0.887 | +0.2%, p=0.915 | 0 / 0 |
| `model_187` | 79.2% / 54.2% | -2.2%, p=0.596 | -5.2%, p=0.437 | 0 / 1 |

`model_140` is the saved checkpoint nearest both smoothed minima; `model_187`
is the required later read. The best checkpoint is parity and the later one is
worse, so neither advances to qualification or longer training. One worker death
in the later policy arm does not explain its five additional physical falls.

**Re-measure if:** controller gait speed, control period, detector attack, robot,
or disturbance profile changes.

**History:**
- 2026-08-25 — the first learned screen found parity at its best checkpoint and
  degradation at its final checkpoint; no walking-reference policy was
  promoted.
- 2026-08-25 — chosen as a deliberately broad first screen. The deterministic
  probe exercised mean planar deltas up to `0.092209 m/s` without a worker or
  controller failure; these are exploration bounds, not promoted hardware
  limits.

## GatedWalkingReferenceDeltaActionCfg

**Current:** the recovery-gated delta channel is its own action term in
`actions/walking_reference_action.py`, not a pair of optional fields on the
shared mc_rtc residual action. Its `__post_init__` reads `Enabled` out of the
configured mc_rtc yaml and refuses to build unless the walking controller named
by `walking_controller` is the one enabled, because
`ismpc_walking::set_ref_vel` exists only there. `AbsoluteWalkingReferenceMixin`
is the second drive mode (`residual_mpc`): a command-manager term sent as an
absolute target, adding no action dimensions.

**Re-measure if:** n/a — structural.

**History:**
- 2026-09-15 — split out of `McRtcResidualActionCfg`. The two modes had been
  mutually exclusive fields guarded by a runtime `ValueError`; they are now two
  classes, and the base action carries three generic extension hooks
  (`_setup_datastore_vector_input_commands`, `_setup_action_extensions`,
  `_process_action_extensions`, `_reset_action_extensions`) instead of any
  walking state.

## walking_reference_velocity_slew_rate

**Current:** `(2.0, 1.5, 3.0)` per second reaches any action bound in at least
`0.10 s`. Slew applies while authority is nonzero; zero authority bypasses the
ramp to guarantee exact zero executed offset and immediate nominal restoration.

**Re-measure if:** action bounds, policy period, controller period, or detector
attack changes.

**History:**
- 2026-08-25 — added before the final gain map so a policy cannot jump a walking
  reference faster than the recovery gate's own attack.

## probe_walking_reference

**Current:** run all gain cohorts concurrently with identical reset state,
encoder bias, and a fixed sagittal finite impulse. Feedback is the signed
deployable command-relative DCM error rotated into the robot frame. The joint
residual remains zero, so the screen isolates the walking-reference channel.

At seed 42, a `0.35 m/s` equivalent impulse at `0.20 m` height for `0.10 s`
produced:

| gain (1/s) | envs | hazards | worker failures | 2 s DCM error (m) | mean command delta (m/s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -2 | 2 | 0 | 0 | 0.050517 | 0.037240 |
| 0 | 2 | 0 | 0 | 0.049570 | 0.000000 |
| +2 | 2 | 0 | 0 | 0.063163 | 0.044402 |
| +4 | 2 | 0 | 0 | 0.077178 | 0.092209 |

A narrower 14-second run measured `0.054583`, `0.048402`, `0.049461`, and
`0.049334 m` for gains `-0.25`, `-0.5`, `-1`, and zero respectively, again with
two worlds per gain and no failures. The best fixed law improved error only
`1.9%`; positive feedback clearly harms, but no fixed law clears the 5% policy
promotion gate. The result justifies a bounded nonlinear short screen, not a
claim that the new task already improves the controller.

**Re-measure if:** detector calibration, DCM construction, reference bounds,
impulse, or walking controller changes.

**History:**
- 2026-08-25 — replaced a random-impulse draft whose cohorts received different
  directions and whose `+4` cohort never activated. Identical impulses make gain
  the controlled difference.
- 2026-08-25 — a final one-env `-0.5 s^-1` run measured
  `max_restore_error = 0.000000000` after the gate closed.
