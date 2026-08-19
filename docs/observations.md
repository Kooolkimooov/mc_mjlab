# Observations

What the policy sees, and why. Terms in `tasks/mdp.py`, wiring in
`tasks/residual_balance/residual_balance_env_cfg.py`.

## Noise levels

**Current:** `base_lin_vel` +/-0.02, `base_ang_vel` +/-0.03, `projected_gravity`
+/-0.05, `joint_pos` +/-0.01, `joint_vel` +/-0.05.

Scaled to what this robot's signals actually are, not to mjlab's locomotion
defaults: those assume ~1 m/s travel, and this base controller walks at 0.09 m/s,
so the same absolute corruption buries the signal.

Measured over a 32 s x 16 env zero-residual walk, RMS per channel was
`base_lin_vel` 0.110, `base_ang_vel` 0.121, `joint_pos` 0.085, `joint_vel` 0.125,
`projected_gravity` 0.577. The levels above keep noise near a tenth to a quarter
of that. **`joint_vel` in particular had been carrying noise 12x its own signal.**

## joint_pos and encoder_bias

`joint_pos` uses `params={"biased": True}`, which is what makes the
`encoder_bias` startup event take effect. Without it the event samples a bias
nothing ever reads.

## The actor/critic split

The critic sees the same signals without observation noise. The terms are
**copied** rather than rebuilt from `func`/`params`: a rebuild silently leaves
behind every other field, which is how the critic came to lose the actor's
history. The group's `enable_corruption=False` is what drops the noise.

The encoder bias goes with the noise: a privileged critic should value states
from the true joint angles, not from the miscalibrated reading the actor has to
live with. mjlab's own tracking task splits the two groups the same way.

**Until 2026-08-17 that was the critic's *only* advantage** — noise-free
`joint_pos` and nothing else. It now also gets four terms the actor cannot have,
which matters more at `gamma = 0.997` than it did at 0.99, because a longer
horizon leans harder on the critic:

| critic-only term | why the actor cannot have it |
| --- | --- |
| `push_recency` | the push is exogenous and unobservable, and is the largest single source of return variance |
| `last_push_velocity` | same, and it is the magnitude the critic needs to value the state |
| `encoder_bias` | the bias *value*, rather than merely its absence |
| `measured_zmp_offset` | the true CoP, without the `KinematicInertial` drift the actor lives with |

`push_recency` is `exp(-age * step_dt / tau)`, **not** raw `mdp.steps_since_push`.
That function reports `NEVER_AGE = 1 << 30` for an env not yet pushed in its
episode; feeding 1e9 into `EmpiricalNormalization` would destroy it, and silently.
Any future term built on `steps_since_push` needs the same treatment.

## history

**Current:** `20` (`CONTROLLER_HISTORY`) on the controller and gait channels, `5`
on everything else.

Five frames at 50 Hz is 0.1 s against a gait cycle near 1 s, which left the policy
close to phase-blind — the same push at mid-single-support and at touchdown call
for different corrections, often opposite ones. Twenty frames is 0.4 s.

Only the controller channels were raised. The width cost is multiplicative, and
the robot-state channels are the ones a longer window helps least: base velocity
and joint state are near-Markov, whereas the plan's recent history is what encodes
where in the stride the robot is.

Actor width goes 284 to 1219 (1179 before `gait_phase`), which grows the first
layer from 145k to 624k parameters. Affordable because collection dominates
completely — 6.94 s against 0.014 s of learning per iteration at 2x2 epochs x
minibatches, so a wider first layer costs no wall clock.

## Gait phase proxies

**Current:** `foot_load_share`, the two sole velocimeters (`left_foot_lin_vel`,
`right_foot_lin_vel`), and `gait_phase` below — all at `CONTROLLER_HISTORY`.

These are **sim-side proxies, and deliberately so.** mc_rtc's walking plan — the
FSM's phase, the next planned footstep, time to touchdown — lives in the
datastore, which the Python bindings do not expose (see `CLAUDE.md`). It is out of
reach without patching them.

What is reachable: `foot_load_share` gives each foot's share of the vertical
contact force, which is support state (double support, left single, right single)
in two numbers, and it reuses `_ZmpSensors` so it costs no new plumbing. The
velocimeters separate swing from stance and give swing speed. Both sensor sets
were already being added to every robot MJCF by
`robots/additional_sensors_configuration.py` and had never been read by anything.

## gait_phase

**Current:** `(cos, sin)` of an inferred gait phase, `PHASE_RATE_REF = 7.1`, at
`CONTROLLER_HISTORY`. Added **alongside** `foot_load_share`, not instead of it —
the load share carries support-state magnitude the angle throws away.

mc_rtc owns the gait and its plan is unreachable through the bindings, so the phase
is inferred from the foot-load phase plane: with
`d = (F_left - F_right) / (F_left + F_right)`, emit
`normalise((d, d_dot / PHASE_RATE_REF))`. That normalised 2-vector **is**
`(cos, sin)` of the phase — no `atan2`, and no 0->1 wraparound discontinuity, which
is why leo_mjlab emits sin/cos rather than a raw scalar.

**Measured before wiring, 6 envs x 1200 steps of zero residual:** the estimate winds
once per gait cycle in **6/6 envs** — 70-89% of steps advance in a consistent
direction, turn ratio 0.65-1.10 against the 1.22 s gait period read off `d`'s zero
crossings. `PHASE_RATE_REF = 7.1` is the measured rms of `|d_dot|`; it sets the
aspect ratio of the phase plane and nothing else.

**Do not filter `d`.** An EMA was tried at four strengths and made it strictly
worse — the turn ratio fell 0.80 -> 0.47 from alpha 1.0 to 0.05 as lag ate the
winding, while consistent-direction stayed flat at ~83%. The derivative is noisy
(`|d_dot|` rms 7.1/s against `|d|` mean 0.695) but the noise does not accumulate.

**The phase runs retrograde and that is fine.** For `d = sin(wt)`,
`atan2(d_dot, d)` *decreases* with time, so a first reading of this measurement
looked like a 13% forward rate and a failure. Direction convention carries no
information for a network reading a 2-vector; what matters is that the winding is
consistent and one turn per cycle, which it is.

## controller_planned_zmp and controller_planned_com_vel

**Current:** both present, `history_length=CONTROLLER_HISTORY`, no noise.

They close a gap the task ran into: it *pays* for tracking `planned_zmp` and
`control_com_vel` while showing the actor neither, so the policy was being scored
against a plan it could not see. What it could see of the controller was
`controller_reference_velocity`, which is joint-level and one integration removed
from the centroidal quantities the reward is written on.

A policy with no way to know when intervening helps has one safe strategy left —
intervene less — and that is what the measurements showed it converging to: **mean
action falling 31% -> 14% of its clip while performance approached the
zero-residual baseline from below rather than passing it.**

Controller-internal and exact, so they carry no observation noise, the same as
`controller_reference_velocity`. Given the same history because the plan steps
foot to foot and the phase is the point.

They are *plans*, not errors. The measured side of `com_velocity_tracking` is
largely inferable from `base_lin_vel`; the measured side — the centre of pressure
under the feet — went to the **critic** as `measured_zmp_offset` on 2026-08-17.
The actor still does not get it directly, but `foot_load_share` is derived from
the same wrenches and does reach the actor.

## Why the controller channels exist at all

Without them the policy cannot phase its residual with the plan it is meant to
protect. `controller_position_error` sees encoder-level noise; the reference
velocity is controller-internal and known exactly.
