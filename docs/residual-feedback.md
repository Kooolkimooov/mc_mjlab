# Residual feedback learning

## SCOPE

**Current:** `Mc-Mjlab-Residual-Feedback-...-Joint-Torque` reproduces the
residual-feedback formulation of
[Ranjbar 2021](https://arxiv.org/abs/2106.04306) on the ResidualMPC task. The
policy adds a bounded offset to the joint positions mc_rtc reads, alongside the
existing torque residual, so the controller replans *with* the correction
instead of resisting it.

Everything except the action term is shared with `## residual-mpc.md#SCOPE`:
same rewards, observations, events, disturbance, command box and PPO settings.
The only difference is where the residual enters, which is the comparison the
paper makes.

**Re-measure if:** the ResidualMPC task's rewards or observations change — the
two tasks are only comparable while they share them.

**History:**
- 2026-09-01 — added after the ResidualMPC residual was measured spending more
  torque for no tracking gain, the signature Ranjbar describes.

## WHY_FEEDBACK

**Current:** a residual added to a controller's *output* fights that
controller's own feedback loop. Ranjbar states it directly: the residual "causes
a feedback distribution shift that the controller sees as external perturbation
which it tries to resist," so the two "compete".

**Our prior does exactly this.** mc_rtc receives live encoder, IMU and wrench
feedback, and the ISMPC stabilizer is an explicit force-feedback loop, so an
injected torque is observed and opposed. The measurements match: across every
ResidualMPC comparison the residual spent *more* torque
(`+5.2%`, p `0.010` on `ent001`) while gaining nothing on the objective, and
under domain randomization it was significantly worse than the bare prior
(`-4.2%`, p `0.009`). `docs/residual-mpc.md#POWERED_RESULT`

**The formulation.** Where residual *action* learning computes
`a = f(o) + a_rl`, residual *feedback* learning computes `a = f(o + o_rl)`, and
the hybrid — Ranjbar's best variant, and the default here — computes
`a = f(o + o_rl) + a_rl`. Their result is that feedback residuals handle
position uncertainty best, action residuals handle orientation uncertainty best,
and the hybrid handles both.

**Re-measure if:** the controller stops receiving live state, which is what makes
the competition possible in the first place.

**History:**
- 2026-09-01 — recorded with the ResidualMPC torque evidence that motivated it.

## feedback_scale

**Current:** `0.02` rad (about `1.15` degrees) at a saturated action, applied per
residual joint to the encoder vector in `ControllerIoBinding._fill_joint_columns`.

**It is a lie told to a balance controller, so it is bounded.** The offset makes
mc_rtc believe a joint is somewhere it is not; too large a value and the
stabilizer solves for a state the robot is not in. `0.02` rad is double the
encoder-bias randomization the controller already tolerates
(`bias_range` `+-0.01`), so it is inside the range the controller is known to
absorb while still being large enough to steer a solve.

**Gated with the torque residual.** The offset is multiplied by the same
`last_gate` the torque residual uses, so a suppressed residual cannot keep
misreporting joint positions.

**Hardware caution:** this channel falsifies proprioception. In simulation that
is a research knob; on a real robot a false state estimate reaching a stabilizer
is a fall, and the scale is not a safety bound in any certified sense.

**Re-measure if:** the encoder bias randomization changes, or a robot other than
HRP5P is used — the tolerable offset is a property of the stabilizer's gains.

**History:**
- 2026-09-01 — set at twice the encoder-bias range; not yet swept.

## torque_channel

**Current:** `True`, giving Ranjbar's hybrid. Setting it `False` zeroes the
torque residual and leaves the feedback channel alone, which is the paper's
feedback-only variant.

**The action layout is `[residual, walking_reference, feedback]`**, with the
feedback block appended last so the base class keeps owning the columns it
already slices. `process_actions` removes the trailing block before delegating.

**Re-measure if:** another action channel is appended — it must go before the
feedback block or after it consistently, since both ends are now sliced.

**History:**
- 2026-09-01 — hybrid chosen as the default because it is the paper's best
  variant on combined uncertainty.
