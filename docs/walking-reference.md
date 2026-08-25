# Walking-reference modulation

## WALKING_REFERENCE_SCALE

**Current:** `(0.20, 0.15, 0.30)` bounds the recovery-only `vx`, `vy`, and yaw
rate offsets. The command reaches these bounds in at least `0.10 s`, is multiplied
by the calibrated recovery authority, and becomes exactly zero when authority
does. The live nominal reference is restored rather than assuming the installed
controller still commands `(0.1, 0, 0)`.

**Re-measure if:** controller gait speed, control period, detector attack, robot,
or disturbance profile changes.

**History:**
- 2026-08-25 — chosen as a deliberately broad first screen. The deterministic
  probe exercised mean planar deltas up to `0.092209 m/s` without a worker or
  controller failure; these are exploration bounds, not promoted hardware
  limits.

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
