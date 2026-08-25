# Improvement roadmap

This roadmap turns the findings in `MC_MJLAB_CRITIQUE.md` into an ordered
implementation and qualification program. Safety and measurement changes land
before experiments; no training default changes unless the final gates pass.

## BOUNDED_POLICY

**Current:** Replace the unbounded Gaussian action with a local tanh-squashed
Gaussian. Preserve the zero latent-mean initialization, learn one scalar
standard deviation constrained to `[0.05, 0.30]`, include the transform
Jacobian in log probability, use the latent Normal for KL, estimate transformed
entropy, and use `tanh(mean)` for deterministic output. Old Gaussian
checkpoints remain legacy inputs and are never silently loaded into the new
policy.

**Re-measure if:** the action parameterization, residual scale, or RSL-RL
distribution contract changes.

**History:**
- 2026-08-24 — selected as the first implementation milestone because the
  current Gaussian can request actions outside the declared normalized range.

## EXECUTED_RESIDUAL

**Current:** Rewards and diagnostics use the residual actually delivered after
squashing, scaling, authority gating, and feasibility projection. The action
term exposes requested normalized, requested physical, executed physical,
authority gate, and projection masks. Inactive authority must produce exactly
zero executed residual.

**Re-measure if:** another transform is added between the policy and actuator.

**History:**
- 2026-08-24 — raw policy penalties were rejected because they charge commands
  the robot never receives and hide commands removed by safety projection.

## FEASIBILITY_PROJECTION

**Current:** Read position, velocity, and effort bounds from the mc_rtc
`RobotModule`. Position and torque residuals are projected as
`clamp(nominal + residual) - clamp(nominal)`. Install hard per-joint effort
limits in the ideal-PD actuators and explicitly clamp torque commands. Retain a
soft torque guard at `0.8` of the hardware limit and a hard guard at `1.0`.
Log settled projection frequency and near-bound activity per joint.

**Re-measure if:** robot modules, actuator models, or PD gains change.

**History:**
- 2026-08-24 — the settled baseline reaches median/p90/p99/max effort ratios
  `0.133/0.203/0.229/0.40`; the reset transient is excluded from settled gates.

## CHECKPOINT_QUALIFICATION

**Current:** Evaluate every saved validation checkpoint from a run directory or
glob using fixed completed-episode counts and paired environment schedules.
Scenarios cover nominal walking, the current velocity kick, a finite impulse,
and robust held-out conditions. Emit CSV and JSON. Apply safety and nominal
gates first, then rank lexicographically by recovery, hazard, and residual use.
Only the selected checkpoint enters the test set. Inference uses clustered or
hierarchical uncertainty by seed and Holm correction for multiple comparisons.

**Re-measure if:** episode allocation, reset behavior, or promotion gates
change.

**History:**
- 2026-08-24 — selected to prevent post-hoc choice from a single checkpoint or
  treating correlated environment episodes as independent samples.

## RECOVERY_DETECTOR

**Current:** Authority is a transparent monotonic score over command-relative
DCM, base angular velocity, tilt, and total foot-load deviation. A calibrated
base-angular-speed rise starts a bounded burst, and a nominal-score dwell rearms
it. Dedicated zero-residual nominal and fixed-energy disturbance cohorts supply
separate train and held-out traces. The gate has a smooth attack and exponential
decay with `0.5 s` time constant. Push schedule, runtime since push, and
critic-only observations are forbidden inputs.

Acceptance requires nominal duty at most `5%`, at least `80%` recall within
`2 s` after disturbance, and fewer than `1%` activations persisting more than
`5%` authority after `2 s`. Outside activation the executed residual is exactly
zero. The prior useful recovery objective is the starting point: nominal DCM
weight `0`, recovery weight `4`, and standard deviation `0.10`, with gait
quality enforced lexicographically rather than folded into the score.

**Re-measure if:** the controller gait, command speed, sensors, or disturbance
distribution changes.

**History:**
- 2026-08-24 — DCM alone overlaps nominal walking; the multi-feature detector
  replaces the whole-vector coherence gate.
- 2026-08-24 — accepted held-out calibration at 0.000% nominal duty, 96.825%
  two-second recovery recall, and 0.000% late activation (n=4620/882).

## SELECTIVE_AUTHORITY

**Current:** Probe two-second DCM, centre-of-pressure, and effort responses to
per-joint and grouped residual pulses. Position authority per joint is
`min(0.01 rad, 0.20 * effort_limit / kp)` and torque authority is
`min(10 Nm, 0.20 * effort_limit)`. Screen ankle-only, sagittal-leg, full
hardware-normalized, and current uniform authority, promoting the smallest
passing set.

**Re-measure if:** hardware limits, PD gains, controller, or robot changes.

**History:**
- 2026-08-24 — chosen to remove joints whose authority adds exploration cost
  without measurable recovery leverage.
- 2026-08-24 — implemented derived ankle/sagittal sets, hardware-normalized
  scales, registered screen tasks, and a paired per-joint/group probe. All three
  groups produced measurable two-second DCM and centre-of-pressure responses;
  promotion remains gated on training and checkpoint qualification.

## FINITE_DISTURBANCE

**Current:** Keep velocity teleport only as a compatibility evaluation. Train
with a finite torso impulse: uniform planar direction, duration
`[0.08, 0.20] s`, force derived from robot mass and equivalent delta velocity,
and contact point up to `0.25 m` above the nominal root. Curriculum delta
velocity ranges are `[0.10, 0.25]`, `[0.10, 0.40]`, and `[0.10, 0.50] m/s` at
iterations `0`, `48,000`, and `96,000` environment steps.

After the standard task is beaten, add staged randomization. Stage one uses
mass/inertia `+-5%`, centre of mass `+-5 mm`, friction `+-10%`, gains and
strength `+-5%`, and actor/controller delays of `0-1` steps. Stage two doubles
those magnitudes. Sensor noise is added only from measured hardware data.
Held-out tests include compass directions, `0.50-0.60 m/s`, higher contact
points, and combinations.

**Re-measure if:** body mass, control period, contact geometry, or target
hardware changes.

**History:**
- 2026-08-24 — finite impulses were selected to make disturbance energy and
  sim-to-real relevance explicit.
- 2026-08-24 — made finite impulses the training default, verified their wrench
  integral live, retained the velocity-kick compatibility path, and registered
  both robustness stages without enabling them in the standard task.

## ACTOR_OBSERVATIONS

**Current:** Remove sole velocities from the actor; keep them for the critic
and reward computation. Screen history lengths `20`, `10`, and `5`, then a
one-frame GRU with hidden size `256`. Initialize both feed-forward and recurrent
policy mean heads to zero.

**Re-measure if:** deployable sensor availability or controller latency
changes.

**History:**
- 2026-08-24 — actor inputs are restricted to quantities available at runtime
  without simulator-only velocimeters.
- 2026-08-24 — removed both sole velocimeters from the actor, registered 10- and
  5-frame feed-forward screens plus a one-frame GRU-256 screen, and preserved
  the velocimeters for critic value estimation and slip measurement.

## PPO_KL_SCHEDULE

**Current:** A local PPO subclass keeps the learning rate fixed within an
update, computes full-rollout KL after the update, and changes the learning
rate once for the following rollout using the existing `desired_kl / 1.5`
rule. Log both schedule KL and diagnostic KL.

Screen adaptive scheduling against fixed learning rates `1e-4`, `3e-4`, and
`1e-3` with `2 x 2` epochs/minibatches. Compare the winner at `2 x 2` with
`5 x 4`, then compare clipped and unclipped objectives. Retain
`gamma=0.997`, `lambda=0.99`, rollout length `256`, and entropy coefficient
`0.0005` unless a gated screen changes them.

**Re-measure if:** the RSL-RL PPO update contract or rollout budget changes.

**History:**
- 2026-08-24 — per-minibatch adaptive learning-rate changes confound a single
  update and make the recorded KL difficult to interpret.
- 2026-08-24 — implemented a local full-rollout scheduler, retained selectable
  legacy adaptive and fixed-rate arms, and logged its independent schedule KL.

## EXTERNAL_CONTROLLER_API

**Current:** Document, but do not implement here, the controller-side parameter
API needed for deployment. The boundary includes recovery state, authority,
per-joint residual requests, projected residuals, and relevant controller
references. Repository work remains compatible with the current mc_rtc Python
bindings.

**Re-measure if:** the external controller exposes datastore bindings or a
versioned residual interface.

**History:**
- 2026-08-24 — controller-library changes are intentionally outside this
  repository's authorization boundary.
- 2026-08-24 — documented the versioned recovery, residual, reference, and
  compatibility contracts, including fail-closed behavior and sequence pairing.

## EXPERIMENT_PROGRAM

**Current:** Run all training in tmux and obtain task identifiers from
`uv run list-envs`. Screens use seed `42`, `128` environments, `30` workers,
and `188` iterations (`48,128` steps per environment). Passing arms restart
fresh for `500` iterations (`128,000` steps per environment), save every `20`
iterations, and qualify every checkpoint. Final runs use seeds `42`, `43`, and
`44`; add `45` and `46` if uncertainty crosses zero at a meaningful effect.

Promotion requires: no hard-limit violation; settled projection below `0.1%`;
nominal detector duty at most `5%`; upper confidence bound on no-push CoM
velocity regression below `+5%`; foot-slip and controller-tracking regression
below `max(5%, 2 * SEM)`; recovery DCM improvement at least `5%` with a
seed-level confidence interval above zero; hazard ratio at most `0.90` overall
and no profile above `1.10`; no regression on the historical velocity kick;
inactive residual exactly zero; and near-bound activity below `1%`.

Rank passing candidates lexicographically by recovery, hazard, then residual
use. If none pass, retain the safety and measurement infrastructure and make no
claim that training defaults improved.

**Re-measure if:** hardware capacity, baseline controller, or evaluation
profiles change.

**History:**
- 2026-08-25 — all 12 short-screen arms completed; `standard/model_180`,
  `authority-ankle/model_140`, and the latter's `model_187` second read advance
  to paired qualification. See `docs/improvement-screens.md`.
- 2026-08-24 — multi-seed promotion replaces the current single-seed evidence.
- 2026-08-24 — added a resumable 12-arm short-screen runner with derived task
  IDs, fixed budgets, local TensorBoard logging, and no pre-gate robust arm.

## VERIFICATION

**Current:** Add assertion scripts for distribution math, zero initialization,
bounds, executed-residual accounting, detector replay, and projection. Every
implementation commit runs Ruff format/check, the type checker with only the
known binding/stub diagnostics, and the prose checker. The demo must retain
HRP5P root height near `0.79 m` and walking controller joint-reference median
near `0.4 rad/s`.

**Re-measure if:** robot, task, controller, or static-analysis baseline changes.

**History:**
- 2026-08-24 — verification is staged before expensive training so failed
  invariants do not consume the experiment budget.
