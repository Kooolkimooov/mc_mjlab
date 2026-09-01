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

## RESULT

**Current:** residual feedback roughly halves survival on this prior. The
formulation transfers structurally but not behaviourally: what a compliant
impedance controller absorbs, a balance-critical QP stabilizer does not.

**Matched-iteration episode length**, 128 environments, identical env config
(measured: zero differences outside the action term):

| run | 50 | 100 | 150 | 200 | 245 |
| --- | ---: | ---: | ---: | ---: | ---: |
| feedback, hybrid | 754 | 945 | 1004 | 946 | 960 |
| feedback, `joint_position` only | 741 | 1032 | 991 | 984 | 909 |
| **ResidualMPC + curriculum** | 478 | 1224 | 2172 | 2296 | **2318** |
| ResidualMPC, no curriculum (`std015`) | 534 | 1318 | 2011 | 2207 | 2079 |

**The curriculum is not the cause.** It was added at the same time as the
feedback channel, so both were confounded until this control ran. ResidualMPC
with it tracks ResidualMPC without it, and `kick_scale` stayed at `1.0` because
survival never fell below the regress threshold — the term behaves as designed.

**Neither is the channel count or the root-pose modality.** Dropping the torque
residual and every modality but `joint_position` changed nothing: `909` against
`960` at iteration 245. The remaining cause is the joint-position feedback
residual itself.

**And it is worse than it looks.** Both feedback runs drove `kick_scale` to its
`0.4` floor within the first `~100` iterations, so across the window compared
above they survived half as long against a disturbance `2.5x` milder than the
`1.0` ResidualMPC was holding at iteration `245`.

ResidualMPC reaches the same floor eventually — `kick_scale` was `0.4` by
iteration `474` — so the difficulty gap is a property of the comparison window,
not of the whole run. It does not weaken the result: by then both arms sit at the
same easiest setting and ResidualMPC still survives about twice as long
(`1948` at `474`).

**The mechanism is the one `## feedback_scale` warns about.** A virtual joint
offset is a lie told to a stabilizer that closes a state and force feedback loop.
Ranjbar's prior is a Cartesian impedance controller on a 7-DoF arm, where a
false joint reading perturbs a spring-damper; here it perturbs what keeps the
robot upright. `## SCOPE` records that difference; this is it measured.

**What is not claimed.** No deterministic checkpoint comparison was run — these
are training curves, and this repository has been wrong reading those before
(`docs/residual-mpc.md#forward_speed`). The claim is about survival during
training under matched conditions, which is what the four runs share. A paired
`compare_to_baseline` at 64 environments would be needed to say anything about
tracking quality.

**Re-measure if:** `feedback_scale` is swept — `0.02 rad` is twice the encoder
bias and remains unswept, so a far smaller offset might be tolerable even if this
one is not.

**History:**
- 2026-09-01 — three runs plus a control; the feedback channel halves survival
  and the curriculum is exonerated.

## feedback_modalities

**Current:** `("joint_position",)` by default. Each entry adds a block to the end
of the action vector, in the order listed, so the layout is
`[torque residual | walking reference | joint_position | root_pose]`.

| modality | width | offsets | scale |
| --- | ---: | --- | --- |
| `joint_position` | one per residual joint | encoder columns, `in_np[:, 0:T]` | `feedback_scale`, rad |
| `root_pose` | 6 | root block, `ro:ro+3` and `ro+3:ro+7` | `root_translation_scale` m, `root_rotation_scale` rad |

**Why a second space.** Ranjbar's argument is that the residual should enter a
space *relevant to the task*, and the paper evaluates an end-effector-pose
variant beside the joint-position one. For a biped the analogue is the root pose:
the ISMPC stabilizer keys off base state, so a correction there has more leverage
than one distributed across leg encoders.

**Rotation is composed, not added.** The root block carries a `wxyz` quaternion;
adding to it yields something that is not a rotation.
`ControllerIoBinding._compose_small_rotation` builds a unit quaternion from the
residual's rotation vector, multiplies in the body frame, and renormalises.

**Applied after both branches.** `_fill_root_and_sensor_columns` writes the root
block in either the named-routing or the fallback path, and the IMU columns are
overwritten afterwards, so the offset is applied last rather than inside a branch.

**The root channel reaches the controller, hard.** Probed in-process with the
torque residual zeroed, one channel saturated at a time, reading the ISMPC's own
plan rather than inferring from motion:

| condition | `planned_step_dx` | `qp_objective` | `vx` |
| --- | ---: | ---: | ---: |
| zero | +0.1435 | -5,407 | +0.1290 |
| root pitch `+0.02 rad` | **+0.2438** | -29,215 | +0.2283 |
| root pitch `-0.02 rad` | +0.2152 | -91,875 | +0.1969 |
| root x `+0.01 m` | +0.1888 | **-171,929** | +0.1686 |

A `0.02 rad` tilt nearly doubles the planned step and adds `77%` to speed; a
`1 cm` position offset multiplies the QP cost by `32`. The controller believes it
is falling and lunges. **That is far more authority than intended**, which is why
`root_rotation_scale` and `root_translation_scale` were set roughly `4x` below
the values probed, at `0.005 rad` and `0.0025 m`.

This probe existed to catch the opposite failure — a knob that sets a value and
drives nothing, as `footsteps_planner::set_mean_speed` did
(`docs/residual-mpc.md#mean_speed`). It found the reverse, and the scales are
sized from it rather than guessed.

**Re-measure if:** a modality is added — the widths feed `action_dim`, and every
recorded checkpoint is tied to that width.

**History:**
- 2026-09-01 — `root_pose` added beside `joint_position` and both enabled by
  default; scales set from the probe above.

## survival_kick_curriculum

**Current:** `initial_velocity_kick.scale` moves on the smoothed fraction of
episodes that end in `time_out`. Above `0.7` survival the kick grows by `0.05`,
below `0.6` it shrinks, clamped to `[0.4, 2.0]`. Wired into **both**
`residual_mpc_env_cfg` and `residual_feedback_env_cfg`, and omitted when
`pushes=False`, since there is no kick to scale.

**It mirrors the paper's table.** Ranjbar raises task uncertainty when the
success rate exceeds `0.7` and lowers it below `0.6`, by a fixed increment. Our
uncertainty is the reset kick rather than hole pose, and survival stands in for
insertion success.

**Structure borrowed, not reinvented.** The smoothing and the advance/regress
deadband follow `episode_length_impulse_curriculum` (`tasks/mdp.py`), which could
not be reused directly: it requires a `stratified_finite_impulse_curriculum` push
term, and both tasks now use `initial_velocity_kick`. Like that term, it reads
the termination buffers during `curriculum_manager.compute`, which runs before
`_reset_idx` clears them.

**It invalidates the recorded ResidualMPC comparisons.** Every figure under
`docs/residual-mpc.md#POWERED_RESULT` and `#tuning_plateau` was trained at fixed
difficulty. Re-baseline before comparing anything against them, at `--num-envs 64`
with the clustered test — never at 16
(`docs/evaluation.md#episodes-are-not-independent-samples`).

**It regulates to its setpoint.** Over the full `mpc-curriculum-control` run:

| quantity | min | max | mean | final |
| --- | ---: | ---: | ---: | ---: |
| `kick_scale` | 0.400 | 2.000 | 1.093 | 1.875 |
| smoothed survival | 0.020 | 0.868 | **0.633** | 0.799 |

Mean survival lands inside the `[0.6, 0.7]` deadband, which is what the term is
for. `kick_scale` exercised both clamps and reversed at each — `1.0` to the `0.4`
floor while the policy struggled, up to the `2.0` ceiling once it did not, then
back. Wide swings are the adjuster hunting around its setpoint with a fixed
increment, not instability.

**It costs nothing on ResidualMPC.** Smoothed reward per step peaks at `0.07431`
against `std015`'s `0.07475` without a curriculum, and ends higher (`0.07034`
against `0.06957`). Episode length tracks the no-curriculum run throughout
(`## RESULT`). This run is the re-baseline that adding the curriculum required.

**Re-measure if:** the episode length or termination set changes — both move what
"survived" means.

**History:**
- 2026-09-01 — added to both tasks so the residual-feedback comparison stays
  like-for-like; validated over a full run, and neutral on ResidualMPC.

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
