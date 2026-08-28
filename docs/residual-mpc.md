# ResidualMPC reproduction

## SCOPE

**Current:** `Mc-Mjlab-Residual-Mpc-Logisticcontroller-Ismpc-Hrp5P-Joint-Torque`
reproduces the selected joint-action/torque-blending architecture from
[ResidualMPC](https://arxiv.org/html/2510.12717#S4) at 100 Hz. The existing
kinematic ISMPC controller still runs at 500 Hz and replans at 20 Hz.

**Re-measure if:** the controller becomes a torque-producing kinodynamic MPC.

**History:**

- 2026-08-28 — first forward-only HRP5P task; lateral/yaw commands, terrain,
  end-to-end comparison, and full-seed studies remain out of scope.

## ACTION_SCALE_BLEND_FACTOR

**Current:** normalized leg actions use
`min(0.01 rad, 0.20 * effort_limit / Kp) / 0.1`, then the resulting posture
torque is blended with lambda `0.1`. This keeps the maximum action-dependent
blended torque within 20% of each active randomized effort limit.

**Re-measure if:** lambda, reference PD gains, or hardware limits change.

**History:**

- 2026-08-28 — adopted from the task implementation plan; the action refreshes
  the scale after startup gain and effort randomization.

## QP_OBJECTIVE_CALLBACK

**Current:** `ismpc_walking::qp_objective` reports `0.5 x^T Q x + p^T x` for
the last successful finite ISMPC QP solution. It starts at zero and retains the
last valid value after a failed solve. Constant terms omitted from the QP are
not reconstructed.

**Re-measure if:** the ISMPC cost or solver convention changes.

**History:**

- 2026-08-28 — introduced as an observation-compatible surrogate for the
  paper's nonlinear MPC value; online observation normalization absorbs scale.

## JOINT_REGULARIZATION

**Current:** raw mean squared deviation from the nominal joint stance has weight
`-1.0`. [ResidualMPC Table I](https://arxiv.org/html/2510.12717#S4.SS1), row
“Joint regularization,” prints this positive raw-MSE function with weight
`+1.0`; under return maximization that rewards deviation, so the implementation
uses the likely corrected sign.

**Re-measure if:** the authors publish an erratum or release reference code.

**History:**

- 2026-08-28 — documented the exact likely sign error rather than silently
  changing the printed objective.

## PPO_SAFETY_CHOICE

**Current:** PPO follows the cited three-layer, 24-step, five-epoch/four-batch
configuration, while retaining this repository's tanh-bounded Gaussian and
initial standard deviation `0.1`. The bounded distribution is a deliberate
hardware-safety choice, not a claim about the paper's unpublished output head.

**Re-measure if:** a reference implementation specifies the action distribution.

**History:**

- 2026-08-28 — initial task configuration.

## FIDELITY_LIMITS

**Current:** the installed ISMPC is a predictive walking planner with a
kinematic whole-body controller. `jointTorque` is used when available; otherwise
its tracked position/velocity reference supplies the nominal PD torque. The QP
objective is ISMPC's quadratic objective, not ResidualMPC's nonlinear value.

**Re-measure if:** controller dynamics constraints populate `jointTorque` or a
whole-body MPC replaces ISMPC.

**History:**

- 2026-08-28 — explicitly bounded the interpretation of the reproduction.

## residual_mpc_ppo_cfg

**Current:** `max_iterations = 1000`, `num_steps_per_env = 24`. The seed-42
baseline says both are larger than the task can use.

**The base controller already saturates the objective.** Rewards are per-second
weights scaled by `step_dt = 0.01` over a 3,000-step episode, so each term has a
hard ceiling. Scored against its own zero-action arm over 56 paired episodes:

| term | ceiling | zero-action baseline | headroom |
| --- | ---: | ---: | ---: |
| `linear_tracking` | 300.0 | 293.565 | **2.1%** |
| `angular_tracking` | 150.0 | 149.980 | 0.01% |
| `orientation` | 30.0 | 29.990 | 0.03% |
| `height` | 30.0 | 29.968 | 0.11% |

The baseline also survives `100%` of episodes to the cap. Total available gain is
about `6.4` points on `380.5`, or `1.7%`, and `model_975` captured `+0.383`
(`p = 0.51`) — indistinguishable from doing nothing. The measurement is not the
limit here: the paired standard error is about `0.58`, so the full headroom would
have read as roughly eleven sigma.

**Training-time falls are exploration, not behaviour.** The run ended with `86%`
of episodes terminating on `height`, yet neither arm falls once under evaluation.
Training samples from a Gaussian with `std ~0.24` across twelve torque channels
while `get_inference_policy` returns the deterministic mean. Read the training
termination curves as a measure of exploration noise, not of the policy.

**The budget is mis-sized.** On the two length-independent measures the run
peaked early and then decayed: reward per step was highest at iteration `0`
(`0.1473`, ending `0.1161`) and smoothed episode length peaked at iteration `143`
(`1096.8`, ending `921.4`, a `16%` regression). About `85%` of a 1,000-iteration
budget was spent past the peak.

**This is the residual-balance calibration failure mirrored.** There the
qualifier's baseline fell in `96.9-100%` of disturbed episodes, so no policy
could show a hazard gain; here it succeeds almost perfectly, so no policy can
show a tracking gain. A residual task is only measurable when its base controller
sits somewhere a residual can move it.

**Re-measure if:** the velocity command range, push band, reward weights, or
episode length change — all four set where the baseline sits against the ceiling.

**History:**
- 2026-08-28 — first seed-42 baseline comparison; no measurable gain over the
  zero-action arm, and the objective is `97.9-99.99%` satisfied without a policy.

## SETTLE_S

**Current:** `15 s` discarded before scoring, against a `30 s` default episode.
The first sweep used `3 s` inside a `12 s` episode and was wrong.

**The ISMPC realises a reference over many gait cycles, not immediately.** Reading
`ismpc_walking::get_ref_vel` back through the `walking_ref_vel` vector output,
with `+0.600 m/s` commanded:

| t | `get_ref_vel` | measured `vx` |
| ---: | ---: | ---: |
| 2.0 s | `+0.600` | `-0.003` |
| 4.0 s | `+0.600` | `-0.002` |
| 6.0 s | `+0.600` | `+0.076` |
| 8.0 s | `+0.600` | `+0.107` |

Still accelerating at eight seconds. A short settle therefore measures the
transient and reports the prior as unable to track anything, which is what the
first sweep concluded: its errors fit `|command - 0.055|` across the whole grid
with zero terminations.

**`set_ref_vel` is not the problem, and neither is the FSM yaml.** The readback
shows the commanded value arriving intact and surviving, so
`Walking::WalkCmdVelImpl`'s `targetCmdVel: [0.1, 0, 0]` in the installed
`LogisticController_ismpc.yaml` does *not* clobber it. That was the working
hypothesis before the datastore was inspected, and acting on it would have meant
editing shared workspace configuration to fix a bug that does not exist.

Whatever bounds the achievable speed is downstream of the reference — footstep
geometry and the step timing behind `set_ts` / `get_ts_target` — which is exactly
the kind of limit the paper's residual is supposed to extend.

**Re-measure if:** the gait period, step length, or controller changes.

**History:**
- 2026-08-28 — raised from `3 s` after the readback showed the reference is
  accepted immediately and the velocity follows over tens of seconds.

## COMMAND_RANGES

**Current:** `vx` in `(0.0, 0.60)`, `vy` and `wz` held at zero. Chosen to straddle
the prior's measured boundary rather than sit inside it.

**Measured envelope of the ISMPC prior**, eight environments per command, 40 s
episodes with the first 20 s discarded, no policy:

| commanded `vx` | 0.00 | 0.20 | 0.40 | 0.60 | 0.80 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tracking error | 0.071 | 0.136 | 0.319 | 0.515 | 0.713 |
| implied measured `vx` | ~0.07 | ~0.064 | ~0.081 | ~0.085 | ~0.087 |
| achieved | 100% | 100% | 0% | 0% | 0% |

Zero terminations anywhere: the prior never falls, it simply never speeds up.
Measured velocity saturates near `0.085 m/s` across a fourfold command range, so
the `achieved` boundary at roughly `vx = 0.30` is earned by the command being
close to a fixed gait speed, not by tracking it.

**This is a weaker prior than the paper's, and the difference matters.** In
[Jeon 2025](https://arxiv.org/abs/2510.12717) the residual extends an already
capable MPC by `78%` in `vx`; the prior does the bulk of the work and the residual
corrects it. Here the prior delivers `0.085 m/s` against a `0.60` command, so a
policy that reached even the old `0.25` box would be contributing several times
the prior's own output. That is closer to end-to-end control with an MPC-shaped
bias than to the paper's residual regime, and any reproduction claim has to say
so.

**`vy` and `wz` stay at zero until measured.** Nothing yet shows the ISMPC moves
laterally or turns on command, and enabling a command the controller ignores
would only add reward the policy cannot earn.

**Re-measure if:** the gait period, step length, controller, or `SETTLE_S`
changes.

**History:**
- 2026-08-28 — widened from `(0.0, 0.25)` after the prior's boundary was measured
  at about `0.30`; the old box sat entirely inside it, which is why the baseline
  scored `97.9-99.99%` of ceiling and left nothing to learn.

## ismpc_walking

**Current:** the prior's speed is capped by footstep geometry, not by the
velocity reference, and only one of the three governing parameters is reachable
from Python.

**What caps it.** From the installed
`LogisticController_ismpc.yaml`: `footsteps_planner.mean_speed: 0.1` (line 156),
`FootManager.deltaTransLimit: [0.1 m, 0.08 m, 5 deg]` per step (line 225), and
`ismpc.ts: 1.3` with `ts_range: [0.7, 2]`. Dividing the per-step translation
limit by the step duration predicts `0.077 m/s` forward and `0.062 m/s` lateral;
measured values are `0.085` and `0.071`. The reference is accepted in full —
`get_ref_vel` reads back a commanded `0.600` unchanged — so nothing between the
command and the planner is at fault.

All three axes respond and all three are slow. Lateral is symmetric: `vy = -0.30`
and `+0.30` give errors `0.230` and `0.227`, so about `+/-0.071 m/s`, against a
`0.072` noise floor at `vy = 0`.

**What is reachable at runtime.** `set_ts`, `set_tds`, `set_com_height`,
`set_torso_pitch`, `set_ref_vel` and `set_disturbance` all marshal through the
patched bindings. `set_ts` works and is clamped to the config's own range:
`get_ts_target` starts at `1.2`, `set_ts(0.7)` takes effect, and `set_ts(0.4)`
silently clamps back to `0.7`. That is a `1.71x` step-rate increase, predicting
roughly `0.146 m/s`.

**What is not.** `ismpc_walking::get_config` and `configure` carry
`ControllerConfiguration&`, which the bindings refuse:
`unsupported signature std::function<ControllerConfiguration& ()>`. So
`mean_speed` and `deltaTransLimit` — the two parameters that would actually open
the envelope — cannot be changed from Python. This repo's `etc/mc_rtc.yaml`
holds only the global keys (`MainRobot`, `Timestep`, `Enabled`), so there is no
repo-local override either; they live in the installed workspace file that
CLAUDE.md already warns a rebuild reverts.

**Consequence for the reproduction.** Without touching workspace configuration
the prior tops out near `0.146 m/s`, so the command box should be about
`(0.0, 0.30)` rather than the `(0.0, 0.60)` currently set, and the experiment is
the architecture on a deliberately slow prior rather than the paper's regime.

**Re-measure if:** the installed controller yaml, the binding's supported types,
or `ts_range` changes.

**History:**
- 2026-08-28 — mapped the datastore surface after `set_ref_vel` was cleared of
  suspicion; the cap is `deltaTransLimit / ts`, and only `ts` is reachable.

## LINEAR_TRACKING_SIGMA

**Current:** `0.06`, down from `0.5`. Sized off the measured prior rather than
picked, following [reward-shaping.md](reward-shaping.md#ZMP_TRACKING_STD).

**At `0.5` the reward could not tell a policy from the bare controller.** The
ISMPC walks at a settled `0.085 m/s` whatever it is told, and
`exp(-((c - v)/(1 + |c|))^2 / sigma)` scored it:

| commanded | 0.00 | 0.10 | 0.20 | 0.30 | 0.60 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prior at `sigma = 0.5` | 0.986 | 1.000 | 0.982 | 0.947 | **0.813** |
| prior at `sigma = 0.06` | 0.887 | 0.997 | 0.858 | **0.634** | 0.178 |

Commanded at `0.6 m/s` while walking at `0.085`, the old term still paid `81%`
of its maximum. That is why the first seed-42 run scored `97.9%` of the
`linear_tracking` ceiling with no policy, and why widening the command box alone
would not have helped: the box was never the binding constraint, the kernel
width was.

**The paper does not fix `sigma`**, so this is a free parameter being sized, not
a fidelity break. Table I gives the functional form and the weight (`10.0`); the
scale is ours to choose, and it has to be chosen against the controller we
actually have.

At `0.06` the prior scores `0.634` at the top of the box, leaving about a third
of the term for a residual to earn — the same target
`ZMP_TRACKING_STD` was sized to. `verify_tracking_reward_discriminates` asserts
the prior's score stays inside `(0.5, 0.8)`, so a later change to either the box
or the prior's speed cannot silently restore a reward nothing can win.

**Only the linear term is retuned.** `angular_tracking` keeps `0.5` because `wz`
is held at zero and the yaw envelope is unmeasured; `orientation` and `height`
keep it because the prior already scores `29.99/30` and `29.97/30` on them,
which is correct behaviour rather than a blunt kernel.

**Re-measure if:** the prior's settled speed, the command box, or the controller
changes — all three move where the kernel should sit.

**History:**
- 2026-08-28 — tightened from `0.5` after the bare prior was measured at `81-100%`
  of the tracking ceiling across the whole command range.

## deltaTransLimit

**Current:** with the objective calibrated so the prior leaves a fifth of the
tracking term unearned, the residual still captures none of it and costs torque.
Paired against its own zero-action arm, 88 episodes per arm, `model_100` of the
recalibrated seed:

| term | prior | policy | delta | p |
| --- | ---: | ---: | ---: | ---: |
| `linear_tracking` | 242.631 | 242.013 | `-0.618` | `0.79` |
| `torque_l2` | -117.894 | -124.449 | `-6.555` | `8.4e-05` |
| **total** | **334.457** | **327.265** | **`-7.192`** | **`7.9e-03`** |

The measurement is now capable: the prior scores `80.9%` of the
`linear_tracking` ceiling rather than the `97.9%` it scored at `sigma = 0.5`, so
roughly `57` points were available. The policy took none of them and lost `7.2`
overall, significantly.

**Why, and it is structural.** The ISMPC's speed is bounded by
`deltaTransLimit / ts` — where the footstep planner puts the feet, and how long
it takes to get there. A torque residual can change how the robot executes a
planned footstep; it cannot make the planner take a longer one. The limitation
and the residual live in different spaces.

In [Jeon 2025](https://arxiv.org/abs/2510.12717) they live in the same space: the
prior is a kinodynamic whole-body MPC emitting torques at 100 Hz, so a torque
residual can push against exactly what limits it. Ours is a kinematic planner
followed by tracking, and the binding constraint sits upstream of anything the
residual touches. That is why the architecture reproduces faithfully — blending,
weights, network, initialisation all match — while the *result* does not.

**What follows.** Raising `footsteps_planner.mean_speed` and
`FootManager.deltaTransLimit` in the installed controller yaml would move the
limit into a region where a residual has something to push against. Nothing on
the RL side reaches it: not the command box, not `sigma`, not the budget, and
not more seeds.

**Re-measure if:** the prior becomes torque-emitting, or the footstep limits
move.

**History:**
- 2026-08-29 — recalibrated objective, 300-iteration seed, paired comparison;
  the residual is significantly worse than the prior it is meant to improve.
