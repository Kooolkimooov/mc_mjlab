# Residual authority

How much the policy is allowed to add on top of mc_rtc, in the control channel's
own unit (rad for position, Nm for torque — one number cannot serve both).

`scale` maps the policy's ~unit output into that authority and `clip` makes it a
hard bound. A residual able to outvote the controller is how the policy learns to
freeze the gait instead of stabilizing it: a swing trajectory is ~0.5 rad, and
rejecting a push needs far less.

Set in `tasks/residual_balance/residual_balance_env_cfg.py`; applied in
`actions/mc_rtc_residual_action.py`.

## residual_scale

**Current:** `0.01` rad for position control, `10.0` Nm for torque.

The bound is set in position units, but the actuators are unlimited on purpose
(mc_mujoco parity, see `pd_actuator_configuration`), so what it really buys is
torque: **0.01 rad through the real `PDgains_sim.dat` gains is 22-27% of every leg
joint's hardware limit.** Measured over 64 s x 16 envs that is affordable —
mc_rtc alone asks ~5% of the budget, and a saturated residual takes the worst
joint (ankle pitch) to 0.64 of its limit without adding a single over-limit step.

**Re-measure before raising this.** Nothing in the sim clamps, so a residual that
outgrows the hardware is invisible here and divergent on the robot.

**History:**
- That warning was then ignored. A run at `0.1` (20899 iterations, 2026-07-31)
  put a saturated residual at **220-270% of the hardware limit**, and — because
  `scale` multiplies the *exploration* noise too — left the policy's own dither at
  115-140% of it once `mean_std` had grown to 0.52. Both tracking terms read
  ~0.008 of a possible 1.0 for the whole run, against 0.68/0.80 for the
  zero-residual baseline, from the first iteration onward and with a near-zero
  mean residual: **the environment was broken before the policy did anything.**
- 2026-08-03 — back to 0.01.

## residual_scales

**Current:** `{".*": residual_scale}` — uniform, which is exactly the previous
scalar behaviour written in the form that lets the ankles (which is what actually
moves the centre of pressure) be raised without also loosening the hips, once the
authority probe says which joints are worth it.

Per-joint authority is expressed once and used for both the scale and the clip so
the two cannot disagree: `processed = raw * scale` is clipped afterwards, so a
clip left at the old scalar would silently cap any joint given a larger scale.

**The footgun:** the patterns must partition **every actuator**, and only half of
that is enforced. mjlab's `resolve_matching_names_values` raises if a joint
matches two keys and if a key matches nothing, so a specific entry cannot sit
alongside a `".*"` catch-all — write it as
`{"[LR]_ANKLE_.*": 0.03, "^(?![LR]_ANKLE_).*": 0.005}`. What it does **not** raise
on is a joint matching no key at all: `BaseAction.__init__` seeds `scale` to ones
and `clip` to +/-inf, so a residual joint left out silently gets scale 1.0 — 100x
the intended authority — with no clip. That is the 220-270% failure above,
reachable by omission.

Note the resolution is against every actuator matched by `actuator_names`, not
just `residual_joints`; the residual subset is sliced out afterwards in
`_setup_residual`.

## residual_joints

**Current:** the legs only. Balancing is what this task rewards, and the upper
body does not hold the robot up.

This is the task's opinion, so it lives in the env cfg — a locomotion or
manipulation task would want the arms. The robot's own `get_residual_joints`
stays the place for exclusions that hold whatever the task is: a joint mc_rtc
models as fixed can carry no residual anywhere, and is dropped there. Filtering
that set rather than taking the legs directly keeps the robot's carve-outs and
refJointOrder ordering.

## Does the residual have enough authority?

`scripts/probe_residual_authority.py` answers this directly: it drives a
**constant** residual instead of a policy and measures `mdp.zmp_error`. A constant
offset is the bluntest possible input — if a full-scale one does not move the
centre of pressure, nothing a policy does will either, and the reward is not the
binding constraint.

Authority is adequate if a full-scale constant residual shifts the error by
>= 0.007 m (20% of the ~0.036 m operating error).

The question arose because the reward proved blind to the policy — see
`RECOVERY_TRACKING_WEIGHT` in [reward-shaping.md](reward-shaping.md).
