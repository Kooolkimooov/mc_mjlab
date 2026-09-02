# ResidualMPC reproduction

## SCOPE

**Current:** `Mc-Mjlab-Residual-Mpc-Logisticcontroller-Ismpc-Hrp5P-Joint-Torque`
reproduces the selected joint-action/torque-blending architecture from
[ResidualMPC](https://arxiv.org/html/2510.12717#S4) at 100 Hz. The existing
kinematic ISMPC controller still runs at 500 Hz and replans at 20 Hz.

**Re-measure if:** the controller becomes a torque-producing kinodynamic MPC.

**History:**

- 2026-08-31 — disturbance brought in line with the paper: a single base-velocity
  kick replaces a recurring impulse train. `## INITIAL_VELOCITY_RANGE`
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

**Current:** `vx` in `(0.15, 0.40)`, `vy` and `wz` held at zero. The floor is the
load-bearing half: near-zero commands make standing still nearly correct, which
is what let a policy freeze. The top leaves the prior at `0.533` of the tracking
term with `sigma = 0.02` — see `## LINEAR_TRACKING_SIGMA` for the arithmetic.

**Measured prior envelope** under the paper's kick, 16 environments each pinned
to one command for `90 s`:

| commanded `vx` | 0.05 | 0.15 | 0.25 | 0.30 | 0.40 | 0.50 | 0.60 | 0.80 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| achieved | 0.038 | 0.148 | 0.241 | 0.243 | 0.236 | 0.243 | 0.213 | 0.199 |
| error | 0.012 | **0.002** | 0.009 | 0.057 | 0.164 | 0.257 | 0.387 | 0.601 |

Tracking is near-exact to `0.30` and saturates around `0.24 m/s`, so `0.40` sits
just past the prior's reach without being hopeless. An earlier revision of this
line said `(0.0, 0.60)` while the constant read `(0.0, 0.30)`.

**Sized against the prior's measured speed.** `linear_velocity_tracking` is
`exp(-sum(((cmd - v) / (1 + |cmd|))^2) / sigma)` — the error is *normalized* by
the command and divided by `sigma`, not by `sigma^2`. With the prior at
`0.269 m/s` and `LINEAR_TRACKING_SIGMA = 0.06`:

| box top | 0.30 | 0.40 | 0.45 | **0.50** | 0.55 | 0.60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prior's score at the top | 0.991 | 0.864 | 0.771 | **0.674** | 0.578 | 0.490 |

`0.50` puts the prior at `0.674` at the hardest command — clearly short of the
ceiling, with a smooth gradient the whole way rather than a flat region.

An earlier revision of this section printed `exp(-(err/sigma)^2)` on unnormalized
error and concluded the term went numerically dead above `0.20`. That was the
wrong functional form: at the old `0.085 m/s` prior the real scores across
`0.10-0.30` were `0.997 / 0.948 / 0.858 / 0.748 / 0.634`, a gentle gradient
throughout, and the "usable band `vx <= 0.18`" that followed from it was wrong.

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

**Current:** `0.02`, with `COMMAND_RANGES` floored at `0.15`. Together these stop
standing still from outscoring walking.

**The freeze was an incentive, not a bug.** At `sigma = 0.06` over a
`(0.0, 0.50)` box, a motionless robot still scored `0.551` of the tracking term —
near-zero commands are common and standing is nearly correct for them — while
also collecting angular, orientation and height reward in full, since none of
those depend on moving. Only `linear_tracking` separates the two behaviours:

| box | sigma | `E[stand]` | `E[walk]` | falls/episode to tie | prior at top |
| --- | ---: | ---: | ---: | ---: | ---: |
| (0.00, 0.50) | 0.06 | 0.551 | 0.920 | **1.11** | 0.613 |
| (0.15, 0.40) | 0.03 | 0.248 | 0.917 | 2.01 | 0.658 |
| **(0.15, 0.40)** | **0.02** | **0.139** | **0.883** | **2.23** | **0.533** |

"Falls to tie" is how many terminations per episode a walking policy can absorb
before freezing pays the same, at `-100` per termination over a `3000`-step
episode. The old setting tied at `1.11`. The bare prior only falls about `0.5`
times per episode, but an *exploring* policy falls far more — at iteration `66`
mean episode length was `923` steps, about `2.3` falls per nominal episode. So
the trap is specific to early training: exploration noise pushes the fall rate
past the tie point, freezing becomes optimal, and the policy stops walking before
it ever learns to.

Raising the floor to `0.15` does most of the work by removing the commands where
standing is genuinely correct; `rel_standing_envs = 0.1` still commands a tenth
of environments to stand, which remains right.

**Observed, `paperkick-cmd050`:** `forward_speed` collapsed `0.062 -> 0.025 ->
0.011` over the first `209` iterations while episode length doubled, then
oscillated near zero — briefly negative at `-0.004` — for the rest of the run.
Reward and episode length both looked healthy throughout, which is exactly why
`## forward_speed` exists.

**THE FREEZE WAS AN ARTIFACT. Everything in this section rests on a mistake.**
`Episode_Metrics/forward_speed` is logged from *training* rollouts, which sample
with `std ~0.24` across twelve torque channels, and that noise breaks the gait.
The deterministic policy walks. `model_999` of `sigma02-cmd015-040`, scored
through `get_inference_policy`, 8 environments, `40 s`, `12 s` settle:

| | mean `vx` | mean command | falls |
| --- | ---: | ---: | ---: |
| kick on | `+0.1972` | 0.279 | 5 |
| kick off | `+0.1857` | 0.246 | 0 |

This repository already documents the same trap for terminations — "training-time
falls are exploration, not behaviour" under `## residual_mpc_ppo_cfg` — and the
lesson was not applied to a newly added metric. **Read `forward_speed` as a
measure of exploration noise, exactly like the termination curves.** The
deterministic speed is the one that means anything, and it needs a checkpoint
evaluation, not a curve.

The `sigma` and `COMMAND_RANGES` changes below were therefore made against a
problem that does not exist. They are not obviously harmful — a tighter `sigma`
and a floored box are defensible on their own terms — but they were not
motivated by evidence, and the run comparison in `## kincstr06-cmd050` used the
older values, so the two are no longer like for like.

**The superseded reasoning follows, kept because the arithmetic is sound and only
its premise was wrong.**

**It did not work, and it failed against its own prediction.** `sigma02-cmd015-040`
froze exactly as its predecessor did: `forward_speed` reached `0.094` by iteration
`136`, collapsed to `0.001` by `229`, and averaged `0.0239` over the last `200`
iterations against the prior's `0.243`. Its maximum, `0.186`, was at iteration
`17` — the near-untrained policy was the fastest the run ever got — and only
`7.5%` of iterations exceeded `0.10 m/s`.

The tie point was `2.23` falls per episode while the observed rate was about
`0.6` (episode length `1859` of `3000`), so the rebalanced reward favoured
walking by roughly `474` to `252` and the policy froze anyway. **The reward
balance is therefore not what holds the freeze in place**, and the arithmetic
above, while correct, is not the explanation.

**What the evidence points to instead.** Freezing is a low-variance local optimum
that is easy to find, while walking through an `8 s` kick is not. The policy must
*actively* suppress the gait to stand still — a zero residual walks at `0.243` —
and it pays `torque_l2` to do so, which reads as escaping instability rather than
chasing reward. Early episodes also end well before the kick, so the only lesson
available is that whatever precedes it is bad. A disturbance curriculum, as
`tasks/residual_balance` already uses via `stratified_finite_impulse_curriculum`,
is the untested lever.

**Re-measure if:** the termination weight, episode length, or the prior's fall
rate under the kick changes — all three set where the tie point falls.

**History:**
- 2026-08-31 — `0.06 -> 0.02` and the box floored at `0.15` after a run froze;
  the next run froze too, refuting the incentive explanation.
- 2026-08-29 — `0.5 -> 0.06`, when the prior scored `97.9%` of the ceiling.

## linear_tracking

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

**Why, and it is structural.** The ISMPC's speed is bounded by its footstep
plan — where the planner puts the feet, and how long it takes to get there. A
torque residual can change how the robot executes a planned footstep; it cannot
make the planner take a longer one. The limitation and the residual live in
different spaces.

The measured `0.085 m/s` fits `footsteps_planner.mean_speed: 0.1` (line 156 of
the installed yaml) and also `FootManager.deltaTransLimit / ts` = `0.1 / 1.3`.
An earlier revision of this section named `deltaTransLimit`; with
`LogisticController_ismpc` enabled and **no `FootManager::` datastore entries
present at all**, `mean_speed` is the likelier owner. Neither is confirmed, and
the argument below does not depend on which.

In [Jeon 2025](https://arxiv.org/abs/2510.12717) they live in the same space: the
prior is a kinodynamic whole-body MPC emitting torques at 100 Hz, so a torque
residual can push against exactly what limits it. Ours is a kinematic planner
followed by tracking, and the binding constraint sits upstream of anything the
residual touches. That is why the architecture reproduces faithfully — blending,
weights, network, initialisation all match — while the *result* does not.

**The datastore cannot reach it.** Every route is blocked at the binding layer:
`ismpc_walking::get_config` and `configure` carry `ControllerConfiguration&`, the
`WalkingInterface` entry holds
`std::shared_ptr<mc_walking::WalkingInterface>`, and `datastore.get` refuses both
with *"which the Python bindings cannot handle"*. A full `--all` listing shows
only three non-`ismpc_walking::` keys and no `FootManager::` namespace. The
writable surface is the scalar and vector setters alone — `set_ts`, `set_tds`,
`set_com_height`, `set_torso_pitch`, `set_ref_vel`, `set_ref_pose`, `set_n_step`
— of which only `set_ts` touches speed, and it clamps at `0.7` for at most
`1.71x`.

**What follows.** Raising the planner's speed limit in the installed controller
yaml would move the constraint into a region where a residual has something to
push against, and extending the patched bindings to marshal
`ControllerConfiguration` would make it settable at runtime instead. Nothing on
the RL side reaches it: not the command box, not `sigma`, not the budget, and
not more seeds.

**Re-measure if:** the prior becomes torque-emitting, or the footstep limits
move.

**History:**
- 2026-08-29 — recalibrated objective, 300-iteration seed, paired comparison;
  the residual is significantly worse than the prior it is meant to improve.

## set_ts

**Current:** the ISMPC's speed ceiling is a ceiling on **step length**, not on
velocity. Step length is invariant at `~0.109 m` while speed scales as `1 / ts`.
Commanding `vx = 0.6 m/s` and varying step duration through
`ismpc_walking::set_ts`, in-process, pushes off, 35 s per point settled over the
last 17 s:

| `ts` | 1.30 s | 1.00 s | 0.80 s |
| --- | ---: | ---: | ---: |
| mean `vx` (`m/s`) | 0.0848 | 0.1087 | 0.1370 |
| implied step length (`m`) | 0.1102 | 0.1087 | 0.1096 |

Step length holds to `1.4%` across a `1.6x` change in cadence, and speed tracks
`0.109 / ts` to within `1%`. This also proves `set_ts` is honoured, which no
getter could confirm: `ismpc_walking` exposes `get_tds` but no `get_ts`.

**The command reaches the solver, which is on a constraint.** Sweeping commanded
`vx` with `ts` at its installed `1.3`, `qp_objective` rises monotonically while
the command is tracked and then pins at the same point speed does:

| commanded `vx` | 0.03 | 0.05 | 0.08 | 0.10 | 0.12 | 0.30 | 0.60 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| achieved `vx` | 0.0333 | 0.0551 | 0.0875 | 0.0911 | 0.0918 | 0.0912 | 0.0916 |
| `qp_objective` | -260 | -672 | -1660 | -1800 | -1800 | -1800 | -1803 |

Tracking is faithful to `0.08 m/s`; above it both quantities are flat to `0.2%`
across a sixfold command increase. A solver merely under-asked would show a flat
cost throughout, so the plan handed to the QP stops changing above `~0.09 m/s`.

**`zmp_cstr_square` is not the cap.** The installed `[0.14,0.08]` carries a
`# hrp4` comment while this runs HRP5P, whose feet are larger, so a mis-sized
centre-of-pressure box was the leading candidate. Doubled to `[0.28,0.12]` with
`zmp_cstr_square_sg_supp` `[0.14,0.05] -> [0.28,0.10]`, commanding `0.6`:
`vx = 0.0916` against a baseline `0.0916`, `qp_objective -1799.8` against
`-1803.0`. The installed file has been restored and verified byte-identical.

**What is left.** Every named candidate is now refuted: `mean_speed` (dead code),
`next_stp_cstr_ratio`, `StepRecoveryState`, the FSM overwriting the command
(`WalkCmdVel::run` never touches velocity; only `start` and a GUI button do), and
`zmp_cstr_square`. The footstep QP does not fail either: with `console_output`
set to `all`, `[Footsteps planner] Step QP failed` appears zero times at both
`0.05` and `0.60`, so there is no fallback path being taken.

The cap is `kinematics_cstr`, measured directly — see `## kinematics_cstr`.

**Neither yaml reaches the planner; it runs on the header default.** The plugin
loads its own `mc_plugins/etc/footsteps_planner_plugin.yaml`
(`kinematics_cstr: [0.3,0.05]`) and then overlays the controller's own
`footsteps_planner` block (`[0.6,0.08]`) at `plugin.cpp:49-56`. Measured live,
`d_h_x` is neither: it is `0.2`, the header default at `footsteps_planner.h:474`.
Editing either file is therefore inert, which is why two config experiments here
returned no change and one of them was wrongly recorded as a refutation.

`Walking_controller.cpp:77` does read the block into `planner_config_`, which is
referenced nowhere else — but that is a separate unused copy, not the path the
plugin uses. An earlier revision of this section called the block dead on that
basis; it is not dead, it is simply not winning.

**Re-measure if:** `ts`, `kinematics_cstr` or the planner's QP weights change.

**History:**
- 2026-08-31 — cap identified as a fixed `0.109 m` step; `zmp_cstr_square`
  refuted; `set_ts` confirmed as the only working speed lever.

## kinematics_cstr

**Current:** the ISMPC's speed ceiling is the footstep planner's kinematic step
box. Planned step length is exactly `d_h_x / 2`, and `d_h_x` is `0.2` — the
header default at `footsteps_planner.h:474`, not either configured value.
Commanding `vx = 0.6` and setting `d_h_x` live through
`footsteps_planner::set_kin_cstr_x`:

| `d_h_x` | planned step `dx` | measured `vx` |
| --- | ---: | ---: |
| `0.2` (in force) | 0.1000 | 0.0914 |
| `1.2` | 0.6000 | **0.2714** |

Step length tracks `d_h_x / 2` to four decimals, and a sixfold box gives `2.97x`
the speed at an unchanged command. This is the constraint every other candidate
was mistaken for.

**The planner clamps; the ISMPC does not.** Reading the planner's request against
what the controller's own footstep QP commits to, both relative to the support
foot:

| commanded `vx` | planned `dx` | optimal `dx` | measured `vx` |
| --- | ---: | ---: | ---: |
| 0.05 | 0.0600 | 0.0598 | 0.0550 |
| 0.60 | 0.1000 | 0.0997 | 0.0918 |

At `0.05` the reference (`v * ts`, about `0.065`) passes through unclamped; at
`0.60` a reference of `0.78 m` is cut to `0.1000`. The ISMPC tracks whatever it
is given to `0.3%`, so `beta_stab`, `zmp_cstr_square` and `next_stp_cstr_ratio`
were never candidates — the step was already short before the controller saw it.

**Neither yaml is in force**, so editing them does nothing; see the provenance
note under `## set_ts`. The runtime setter is the working knob; it is a source
edit, so a rebuild keeps it and only a `git restore` removes it.

**Swept for stability; `0.6` is the setting to use.** Four environments, pushes
off, `25 s` scored after a `15 s` settle at each value, `vx = 0.6` commanded:

| `d_h_x` | planned step | `vx` | `z` mean | `z` min | terminations |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.20 | 0.1000 | 0.0907 | 0.7660 | 0.7571 | 0 |
| 0.40 | 0.2000 | 0.1802 | 0.7660 | 0.7545 | 0 |
| **0.60** | **0.3000** | **0.2690** | 0.7660 | 0.7508 | 0 |
| 0.80 | 0.4000 | 0.2720 | 0.7661 | 0.7499 | 0 |
| 1.00 | 0.5000 | 0.2715 | 0.7661 | 0.7508 | 0 |
| 1.20 | 0.6000 | 0.2712 | 0.7661 | 0.7505 | 0 |

No terminations anywhere and root height is flat to `0.01%`, so nothing here
destabilises the gait. Speed is linear in `d_h_x` up to `0.6` and then flat:
`0.6` captures `0.269` of the `0.272` available, so a larger box only makes the
planner request steps the controller will not execute — `0.6 m` planned against
`0.35 m` realised at `1.2`.

**A second cap sits at `0.272 m/s`, confirmed ISMPC-side.** Reading the planner's
request against the controller's commitment:

| `d_h_x` | planned | optimal | `vx` |
| --- | ---: | ---: | ---: |
| 0.6 | 0.3000 | 0.2976 | 0.2688 |
| 1.2 | 0.6000 | **0.2998** | 0.2713 |

The ISMPC clips at `0.30 m`, which is `foosteps_kin_cstr: [0.6, 0.25]` — a
`0.6 m` full width, `+-0.3 m` per step. At `d_h_x = 0.6` the planner asks `0.3000`
and the controller commits `0.2976`, so the two limits meet exactly there: below
it the planner binds, above it the controller does and the extra request is
discarded. That is the argument for `0.6` and not more.

**Where the value actually comes from.** Neither the plugin yaml nor the
controller block wins: mc_rtc merges the *robot* file last, and
`mc_logistic_controller/src/controller/etc/robots/hrp5_p.yaml` sets
`kinematics_cstr: [0.2, 0.2]`. That is the file to edit, and doing so needs no
runtime setter — verified through worker processes, the configuration training
uses: `planned 0.3000, vx +0.2694, z_min 0.7508, 0 terminations`, matching the
in-process result. One caution: that file also feeds `LogisticController_bwc`
and `_none` for HRP5P.

**This is a screen, not a safety case.** Pushes were off, `zmp_cstr_square` is
still sized for HRP4, and `25 s` across four environments is a thin sample for
termination counts; the flat `z_min` is the stronger evidence.

**Training cannot use this yet.** The setter reaches in-process controllers only,
and the worker-process `configure` path is still broken, so a training run gets
the unmodified prior. Making the planner actually read its configuration would
remove the need for the runtime setter entirely.

**Re-measure if:** the plugin is rebuilt from clean, or the planner starts
reading its configuration.

**History:**
- 2026-08-31 — identified as the cap after `mean_speed`, `next_stp_cstr_ratio`,
  `StepRecoveryState`, `zmp_cstr_square` and footstep-QP failure were each
  refuted; two earlier config experiments were inert because the yaml is unread.

## kincstr06-cmd050

**Current:** with the objective finally calibrated so the prior leaves real
headroom, the residual still does not beat it — and the reason is now visible
rather than inferred. 1000 iterations, seed 42, `model_999`, paired against its
own zero-action arm at 16 environments, per step so episode length divides out:

| term | prior | policy | delta | p |
| --- | ---: | ---: | ---: | ---: |
| `linear_tracking` | 0.07756 | 0.07425 | `-4.3%` | `2.2e-02` |
| `torque_l2` | -0.04401 | -0.04154 | `+5.6%` | `1.4e-03` |
| **TOTAL** | **0.10342** | **0.10253** | **`-0.9%`** | **`0.50`** |

**The calibration worked.** `linear_tracking` has a per-step ceiling of `0.1`
(weight `10` times `step_dt`), so the prior sits at `77.6%` of it against the
`97.9%` it scored before `kinematics_cstr` and `COMMAND_RANGES` moved. Headroom
is `22.4%`, not `2.1%`, and the measurement resolves effects far smaller than
that — both individual terms come back significant.

**The residual optimised what it can reach.** Tracking got significantly *worse*
and torque significantly *better*, netting a wash. That is the structural
limitation as data: a torque residual cannot lengthen a footstep plan, so the
tracking headroom stays out of reach, while torque economy is directly in its
control. The training curves agree — `Metrics/twist/error_vel_xy` ends at
`0.459` against `0.104` for the zero-initialised policy, and smoothed
reward-per-step peaks at iteration `0`.

**Episode length is not the story.** It grew `117 -> ~1900` and oscillated in a
`1700-1980` band from iteration `178` on, while reward-per-step stayed flat
within `+-2%` for `370` iterations. Two separate reads of "the peak" from that
curve were both wrong, once in each direction; the length-independent measure
was flat the whole time.

**Re-measure if:** the prior becomes torque-emitting, or `foosteps_kin_cstr`
opens the way `kinematics_cstr` did.

**History:**
- 2026-08-31 — first run on the recalibrated task; headroom is real and the
  residual captures none of it, trading tracking for torque.

## INITIAL_VELOCITY_RANGE

**Current:** one base-velocity kick per episode, `+-0.5 m/s` planar and
`+-0.5 rad/s` yaw, withheld until `KICK_WARMUP_S`. This replaces a recurring
impulse train that fired every `5-7 s` indefinitely.

**The paper never pushes the robot.** Its only disturbance is a randomized
initial base velocity, `||v_xy|| <= 0.5 m/s` and `||w|| <= 0.5 rad/s`, with
survival scored over the following five seconds
([Jeon 2025](https://arxiv.org/abs/2510.12717) Fig. 4). Robustness is otherwise
shown on unseen terrain and gaits; training is on flat ground with no terrain
randomization. The recurring impulses here were a local invention and a harder
problem than the paper's, consuming reward its policy never has to defend.

**One deviation:** the paper bounds the *norm*, while
`push_by_setting_velocity` samples per axis, so a corner sample reaches `0.707`.

**This is the first headroom a residual can actually reach.** Recovering from a
kick is a torque-level problem; the tracking headroom is not, being fixed by the
footstep plan upstream — see `## kincstr06-cmd050`. Measured on the bare prior,
`28` terminations across `16` environments in `90 s`, spread evenly across
commands rather than concentrated at speed, so it is the kick and not the
command that fells it.

**Re-measure if:** `KICK_WARMUP_S`, the episode length, or the gait changes.

**History:**
- 2026-08-31 — replaced the recurring impulse train after checking the paper.

## KICK_WARMUP_S

**Current:** `8.0 s`. The kick is suppressed until then and fires once per
episode, leaving `22 s` of a `30 s` episode to recover.

**A kick at reset lands on a robot that is not walking.** The FSM stands through
the first seconds after a controller reset: measured `vx` is `-0.003` at `2 s`,
`-0.002` at `4 s`, `0.076` at `6 s` and still rising at `8 s`
(`## SETTLE_S`). Disturbing at `t = 0` would test the controller's startup
transient rather than its gait, and the controller reset may simply absorb it.
`8 s` is the first point the gait is reliably established.

**Once per episode, not per interval.** `initial_velocity_kick` compares the
global `last_push_step` against the per-episode `episode_length_buf`, so the
event cannot fire twice in one episode however often the manager calls it.

**Known limitation:** the timing is deterministic, so a policy can in principle
anticipate it. The paper's `t = 0` kick is equally deterministic, so this
matches rather than worsens it, but a randomized window would be stronger.

**Re-measure if:** the FSM's startup sequence or `episode_length_s` changes.

**History:**
- 2026-08-31 — introduced with the paper-faithful disturbance; a first
  implementation fired at reset and was measuring the standing transient.

## forward_speed

**Current:** `forward_speed` and `commanded_speed` log every run as
`Episode_Metrics`, giving base speed beside the command that asked for it.

**These are exploration-noise measurements, not behaviour.** Like every training
curve here they come from stochastic rollouts, so a policy whose deterministic
mean walks at `0.19 m/s` can log `0.002`. Two full runs were read as "the policy
froze" on exactly that mistake. To learn what a policy actually does, load the
checkpoint and score `get_inference_policy`; the curve only says how much the
sampled noise disturbs the gait.

**`error_vel_xy` alone cannot diagnose a gait.** An error norm reads the same
whether the robot is walking too slowly or tracking badly at speed, which is
exactly the ambiguity that hid the `0.09 m/s` cap for as long as it did. The
pair separates them.

**Re-measure if:** never — these are diagnostics, not tuned values.

**History:**
- 2026-08-31 — added so the speed cap would have been visible from the training
  curves alone.

## sigma02-cmd015-040

**Current:** the first run where the residual beats its prior on the objective.
`model_999`, paired against its own zero-action arm at 16 environments, per step:

| term | prior | policy | delta | p |
| --- | ---: | ---: | ---: | ---: |
| `linear_tracking` | 0.04497 | 0.05092 | **`+13.2%`** | **`2.6e-02`** |
| `torque_l2` | -0.04088 | -0.04299 | `-5.1%` | `1.0e-02` |
| `joint_regularization` | -0.00008 | -0.00008 | `-4.1%` | `1.7e-02` |
| **TOTAL** | **0.07361** | **0.07745** | **`+5.2%`** | **`8.2e-02`** |

**It spends torque to buy tracking**, which is the trade a residual is supposed
to make and the mirror of `## kincstr06-cmd050`, where it spent tracking to save
torque. Total is positive but not significant at `0.05`, so the defensible claim
is "significantly better on the objective, directionally better overall" — one
seed, one checkpoint.

**This weakens the structural argument recorded elsewhere here.** That argument
said a torque residual cannot reach tracking headroom because step length is set
by the footstep planner upstream. It gained `13.2%` regardless, so the claim was
too strong: executing a planned step better, and recovering from a kick faster,
both improve tracking without changing the plan. What remains true is that the
residual cannot make the planner take a *longer* step, so the envelope itself
still ends where `## kinematics_cstr` puts it.

**Deterministic speed is below the prior's** — `0.186-0.197 m/s` against `0.243`
— while tracking reward is better. Both arms in the comparison start each episode
with the FSM standing for several seconds and absorb the same kicks, so the
per-step average is far below the settled envelope figures in
`## COMMAND_RANGES`; the two are not comparable, and only the paired delta is.

**The gain is the disturbance's, not the reward tuning's.** `paperkick-cmd050`,
trained before either reward change, scored on the *same* current objective:

| per step | prior | `paperkick` | `sigma02` |
| --- | ---: | ---: | ---: |
| `linear_tracking` | ~0.0455 | 0.04947 `+7.9%` (p `0.17`) | **0.05092 `+13.2%`** (p `0.026`) |
| `torque_l2` | ~-0.0416 | -0.04220 `-0.4%` (p `0.73`) | -0.04299 `-5.1%` (p `0.010`) |
| **TOTAL** | ~0.0732 | **0.07668 `+5.3%`** (p `0.11`) | **0.07745 `+5.2%`** (p `0.08`) |

The two totals are indistinguishable, so `sigma = 0.02` and the `(0.15, 0.40)`
floor — both chosen against the freeze artifact — did not improve the result.
What they changed is the *composition*: `sigma02` tracks better on an identical
yardstick (`0.05092` against `0.04947`, a real difference rather than a
sensitivity artifact, since both were measured under the same config) and pays
more torque for it. A better trade, not a better outcome.

Neither total is significant at `0.05`. The one significant result on the
objective is `sigma02`'s tracking gain.

**Re-measure if:** the disturbance, `sigma`, or the command box changes — this
run used the kick, `sigma = 0.02` and `(0.15, 0.40)` together.

**History:**
- 2026-08-31 — first measurable win for the residual, after the disturbance was
  brought in line with the paper.

## entropy_coef

**Current:** `0.01`. At the previous `0.1` the entropy bonus was the dominant
term in the loss, and the policy optimised noise rather than reward.

**Measured on `sigma02-cmd015-040`:**

| term | value at iteration 999 | coefficient | contribution |
| --- | ---: | ---: | ---: |
| entropy | `1.44` | `0.1` | **`0.144`** |
| surrogate | `-0.0169` | `1.0` | `0.017` |

The entropy term outweighed the surrogate objective by `8.5x`. `Policy/mean_std`
rose from its `0.1` initialisation to the `0.30` ceiling of `std_range` by
iteration `100` and stayed there for the remaining `900`, which is what a
dominant entropy bonus buys: maximum permitted noise, permanently.

**This is what made the training curves unreadable.** `Metrics/twist/error_vel_xy`
jumps `0.264 -> 0.613` exactly as std reaches `0.30`, and
`Episode_Metrics/forward_speed` collapses at the same point, while the
deterministic policy walks at `0.19 m/s`. Two runs were misread as frozen on
those curves; see `## forward_speed`.

**`0.01` is the value the paper inherits.** ResidualMPC states it uses "the same
PPO hyperparameters from [52]" — Jeon et al. 2023, *Benchmarking potential based
rewards for learning humanoid locomotion* — which uses the stock rsl_rl settings,
where `entropy_coef` is `0.01`. The `0.1` here was a local deviation, not a
paper-faithful choice.

**Not the authority limit.** `projection_fraction` averages `0.001` across the
run and `maximum_effort_ratio` about `0.52`, so the residual's blended torque is
essentially never clipped and it sits at half its effort limit. Raising
`ACTION_SCALE_BLEND_FACTOR` or lambda would change nothing; the policy is not
asking for more than it is allowed.

**Learning rate is not implicated.** The adaptive schedule held it between
`1.0e-05` and `2.25e-05` all run, so KL is near target and the schedule is
working as intended.

**It halved the problem rather than removing it.** At `0.01` the std still climbs
to the same `0.30` ceiling, just more slowly — `0.100` at init, `0.148` by
iteration `100`, `0.231` by `304`, saturated by roughly `500`. Any positive
entropy bonus pushes std up until it hits the bound. Everything degrades with it,
monotonically:

| iteration | 100 | **150** | 200 | 250 | 999 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Policy/mean_std` | 0.148 | 0.169 | 0.192 | 0.207 | 0.300 |
| smoothed reward/step | 0.0732 | **0.0746** | 0.0704 | 0.0645 | 0.0529 |

Reward per step peaks at iteration `140` and falls `29%` by the end of the run.
The next lever is therefore `std_range`'s ceiling — a bound cannot creep the way
a pressure does — and `0.15` is where this run measurably performed best. That
bound is ours, not the paper's: the reference uses an unbounded Gaussian with
`init_noise_std = 1.0`.

**Measured result, `model_150` against its own zero-action arm:**

| term | prior | policy | delta | p |
| --- | ---: | ---: | ---: | ---: |
| `linear_tracking` | 0.04401 | 0.05165 | **`+17.3%`** | **`6.7e-03`** |
| `termination` | -0.00035 | -0.00019 | `-44.0%` | `5.9e-02` |
| `torque_l2` | -0.04011 | -0.04221 | `-5.2%` | `1.1e-02` |
| **TOTAL** | **0.07331** | **0.07902** | **`+7.8%`** | **`1.9e-02`** |

Against every earlier run on the same yardstick:

| run | tracking | p | total | p |
| --- | ---: | ---: | ---: | ---: |
| `kincstr06-cmd050` | `-4.3%` | `0.022` | `-0.9%` | `0.50` |
| `paperkick-cmd050` | `+7.9%` | `0.17` | `+5.3%` | `0.11` |
| `sigma02-cmd015-040` | `+13.2%` | `0.026` | `+5.2%` | `0.082` |
| **`ent001` `model_150`** | **`+17.3%`** | **`0.0067`** | **`+7.8%`** | **`0.0185`** |

The first significant total, and the `-44%` termination delta is the
kick-recovery gain that `## INITIAL_VELOCITY_RANGE` predicted would be the only
headroom a torque residual could reach.

**Two limits on that claim.** `linear_tracking` at `p = 0.0067` does not clear a
Bonferroni threshold of `0.005` across ten terms, though TOTAL is a single
primary endpoint and is not subject to that correction. And this is `model_150`
where every other row is `model_999`, so part of the gain may be the checkpoint
rather than the coefficient; capping `std_range` and comparing a final checkpoint
would separate them.

**Re-measure if:** `std_range`, the reward scale, or the action space changes —
all three move where the entropy term sits relative to the surrogate.

**History:**
- 2026-08-31 — `0.1 -> 0.01` after the entropy term was found to outweigh the
  surrogate by `8.5x`; changed alone, so the effect is attributable.

## std_range

**Current:** `(0.05, 0.15)`. The ceiling is the operative half: it is a bound
where `entropy_coef` was only a pressure.

**Why a bound.** Any positive entropy bonus walks the std to whatever ceiling it
is given. At `entropy_coef = 0.1` it reached the old `0.30` cap by iteration
`100`; at `0.01` it still reached it, around `500`. Lowering the coefficient
changes the speed of the climb, not its destination, so the cap is what actually
decides where the policy explores.

**`0.15` is where the previous run measured best**, not a guess — smoothed reward
per step peaked at iteration `140` with `Policy/mean_std` at about `0.16`, and
fell monotonically as std rose past it (`0.0746` at `150` against `0.0529` at
`999`, a `29%` loss). See the table under `## entropy_coef`.

**This bound is ours, not the paper's.** ResidualMPC inherits stock rsl_rl
settings, which use an unbounded Gaussian with `init_noise_std = 1.0`. The
bounded squashed Gaussian here is the deliberate hardware-safety choice recorded
under `## PPO_SAFETY_CHOICE`, and the cap is part of it.

**It stopped the decay without raising the peak.** Std pins at the new `0.15`
ceiling — any entropy bonus still walks it there, the ceiling is simply lower —
and smoothed reward per step holds flat across the run:

| iteration | 150 | 300 | 500 | 700 | 900 | 999 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reward/step | 0.0737 | 0.0692 | 0.0704 | 0.0711 | 0.0687 | **0.0696** |

The final value is `7%` below the peak against `29%` for `entropy_coef = 0.01`,
and `32%` above that run's final. The peak itself is unchanged (`0.0748` at
iteration `127` here, `0.0748` at `140` there), so the cap prevents loss rather
than producing gain — which was its purpose, since it makes `model_999` scoreable.

**Scored, and the tuning has plateaued.** `model_999` returns `+5.1%` total
(p `0.0997`), tracking `+9.1%` (p `0.081`) — the same band as every run before
`ent001`. See `## tuning_plateau`.

**Re-measure if:** `entropy_coef` or the action scale changes.

**History:**
- 2026-09-01 — `0.30 -> 0.15`, set from the iteration-140 peak of the
  `entropy_coef = 0.01` run.

## paper_torque_blend

**Current:** eq (23) references the **default joint posture**; the PD fallback
references the **controller target**. They are separate arguments, `default_q`
and `controller_q`, because a single tensor cannot express the difference and
cannot be tested.

**This was wrong until 2026-09-02, and it changed the architecture.** One `q_hat`
served both, filled from `interpolated_control["q"]` — the controller's moving
target. So the implemented residual was

`Kp(a + q_cmd - q) - Kd qd` instead of `Kp(a + q_default - q) - Kd qd`,

differing by `Kp(q_cmd - q_default)`, scaled by lambda in the blend and
substantial while walking.

**The paper is explicit**, immediately after eq (25): "*a* is the action output
from the residual policy, **q_hat are default joint positions**". It also
contrasts the two action spaces directly — eq (22) is "relative to `q_cmd` which
is non-stationary" while eq (23) "outputs joint setpoints **relative to default
joint positions**". The implementation had case 1's reference inside case 2's
equation.

**Why it matters more than a scale error.** With a zero action the buggy residual
reduces to `Kp(q_cmd - q) - Kd qd`, a second copy of the prior's own tracking
torque. The paper's version gives a pull toward the default posture, which is
precisely why lambda exists: "the residual policy will have non-zero torques even
if the network is zero-initialised. Consequently, we introduce a weighting
parameter." The bug deleted the effect lambda was introduced to control, and it
is a plausible cause of a residual that preserves or amplifies its prior rather
than correcting it.

**Every checkpoint before this is incompatible** with the corrected semantics.
The measured results recorded elsewhere in this file — including
`## POWERED_RESULT` — were produced under the wrong equation and say nothing
about the paper's architecture.

**Not shared with residual_balance.** That task's torque action builds its
nominal from `interpolated_control["q"]` and adds the residual directly as a
torque, with no `q_hat` term, so it never had this reference to confuse.

**Re-measure if:** the action space or the default stance changes.

**History:**
- 2026-09-02 — split into `controller_q` and `default_q` after external review;
  the contract now uses deliberately different tensors, so a swap fails it.

## POWERED_RESULT

**Current:** measured properly, the residual is **worse** than its prior on
established walking. `std015` `model_999`, 64 environments, 45 minutes,
`--skip-s 5.5`, clustered by environment:

| term | prior | policy | delta | p |
| --- | ---: | ---: | ---: | ---: |
| `linear_tracking` | 0.05517 | 0.05269 | `-4.5%` | `1.9e-04` per-episode |
| **TOTAL** clustered | 0.07682 | 0.07359 | **`-4.2%`** | **`8.7e-03`** |

The per-episode p is not trustworthy on its own; see
`docs/evaluation.md#episodes-are-not-independent-samples`.

**This supersedes every positive figure recorded above.** Those came from
16-environment runs — eight per arm — where the standard error on the difference
is about `3%` of the mean, the same size as the effects claimed, and where the
p-values treated correlated episodes as independent. The same checkpoint scored
`+3.2%` at 16 environments and `-4.2%` at 64.

**On the true model it is a wash.** Rerun with `--nominal`, which drops the
startup randomization both arms were unknowingly being scored under, the deficit
falls to `-1.3%` (p `0.18`) — indistinguishable from the prior. The significant
`-4.2%` holds only under domain randomization, so the residual's real weakness is
robustness to model variation rather than tracking.
`docs/evaluation.md#nominal`

**What the residual actually does.** It starts the robot moving sooner than the
prior, during the `5.5 s` in which the FSM is still standing, and tracks worse
once the gait exists. Unskipped scoring nets those together into an apparent
small gain; separating them shows a startup effect and a walking regression.

**Re-measure if:** anything in the task or policy changes — and at `--num-envs 64`
or more, with the clustered test, never at 16.

**History:**
- 2026-09-01 — the sign reversed under adequate power; the reproduction does not
  currently show a residual that improves on its prior.

## tuning_plateau

**Current:** five paired comparisons, all on the same objective. Configuration
differences between the last four are smaller than the measurement noise.

| run | checkpoint | tracking | p | total | p | prior's tracking |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `kincstr06-cmd050` | `model_999` | `-4.3%` | `0.022` | `-0.9%` | `0.50` | — |
| `paperkick-cmd050` | `model_999` | `+7.9%` | `0.17` | `+5.3%` | `0.11` | 0.04585 |
| `sigma02-cmd015-040` | `model_999` | `+13.2%` | `0.026` | `+5.2%` | `0.082` | 0.04497 |
| `ent001` | `model_150` | `+17.3%` | `0.0067` | `+7.8%` | `0.019` | 0.04401 |
| `std015` | `model_999` | `+9.1%` | `0.081` | `+5.1%` | `0.0997` | 0.04823 |

**The noise floor is the size of the effects being compared.** The last column is
the *same prior* every time — a zero-action arm with no policy — yet it scores
`0.04401` to `0.04823`, a `9%` spread from command draws and kick samples alone.
The configuration differences under test move the total between `+5.1%` and
`+7.8%`, which is the same magnitude. At one seed per configuration they cannot
be separated.

**`ent001`'s `+7.8%` was over-read at the time**, including in this document, as
the entropy fix roughly doubling the effect. It is more likely a favourable draw:
the run before it and the run after it both return about `+5.2%`, and its
checkpoint (`model_150`) differs from theirs.

**What is solid.** Four independent runs put the residual between `+5.1%` and
`+7.8%` on total per-step reward against its own prior, every one positive, with
tracking gains of `+7.9%` to `+17.3%`. The direction is consistent; the magnitude
is not resolved.

**And much of it is the startup, not the gait.** Rescoring `std015` with the
first `5.5 s` of each episode discarded drops the total from `+5.1%` to `+3.2%`
(p `0.42`) and tracking from `+9.1%` to `+2.2%`, because the baseline gains far
more from the skip than the policy does. See `docs/evaluation.md#skip_s`. Every
figure in the table above is the unskipped measure, which is the task as trained;
on established walking alone the advantage is smaller and not significant.

**Seeds, not tuning, is the next spend.** Further configuration changes cannot be
evaluated below this noise floor, whereas repeated seeds of one configuration
would resolve a `~5%` effect and give the variance needed to compare anything
else afterwards.

**Re-measure if:** `--num-envs` or the comparison duration rises — both shrink
the noise floor and would change this conclusion.

**History:**
- 2026-09-01 — recorded after `std015` returned to the `+5%` band, showing the
  `ent001` result was within run-to-run variation.

## mean_speed

**Current:** the planner's cruise speed is now settable at runtime, but the task
does not yet use it. `MEAN_SPEED_OFFSET` and `MEAN_SPEED_COMMANDS` are defined
and unwired.

**A workspace C++ change backs this, and it is not in this repository.**
`~/workspace/src/FootSteps_Planner/src/plugin.cpp` gained two datastore entries
beside the existing `footsteps_planner::configure`:

```cpp
datastore().make_call("footsteps_planner::set_mean_speed",
                      [this](double speed) {
                        config_.add("mean_speed", speed);
                        planner_.v_ = speed;
                      });
datastore().make_call("footsteps_planner::get_mean_speed",
                      [this]() -> double { return planner_.v_; });
```

A `double` because the bindings marshal doubles but refuse
`mc_rtc::Configuration`, which is what the neighbouring `configure` takes, and
refuse `std::shared_ptr<mc_walking::WalkingInterface>` too. `planner_.v_` is
public and already read at `plugin.cpp:95` for a GUI input. Assigning it
directly rather than rebuilding the planner avoids discarding the live plan.

Verified against a running controller: the keys register after `init`, and
`get` / `set 0.30` / `get` reads `0.1 -> 0.3`. The edit lives in the workspace
*source* tree, so a rebuild preserves it and `ninja install` reapplies it; a
`git restore` in `FootSteps_Planner` is what drops it. Edits to files under
`install/` are the opposite — `ninja install` overwrites them, as it did to this
plugin's yaml. Nothing in this repository would explain the task getting slower
after either.

**One host bug this exposed and fixed.** `ControllerHost.configure` validated
every configured datastore callback before `controller.init`, so any entry
registered by a *global plugin* — which is registered during `init` — could never
pass. The existence check now runs after `init`; the binding-capability check
stays at configure, being ordering-independent.

**Raising it changes nothing measurable.** Held at `0.3` through
`datastore_scalar_holds` and swept in-process, the envelope is identical to the
installed `0.1` to three decimals. Cells are mean tracking *error* in `m/s`, not
achieved speed: a `0.60` command reading `0.515` achieved `0.085`.

| commanded `vx` | 0.00 | 0.20 | 0.40 | 0.60 |
| --- | ---: | ---: | ---: | ---: |
| `mean_speed = 0.1` | 0.071 | 0.136 | 0.319 | 0.515 |
| `mean_speed = 0.3` | 0.075 | 0.138 | 0.320 | 0.516 |

**The knob arrives and is ignored.** Read back through `controller_scalars`
from inside a running simulation while commanding `0.6 m/s`:

```
t= 5.0s  live mean_speed=0.3000  baseline=0.1000  measured vx=+0.0252
t=10.0s  live mean_speed=0.3000  baseline=0.1000  measured vx=+0.1279
t=15.0s  live mean_speed=0.3000  baseline=0.1000  measured vx=+0.0612
```

The offset reaches the controllers, the baseline is captured correctly, and the
speed does not change. So the earlier ambiguity is closed in the unhelpful
direction: it is not that the value failed to arrive.

**`mean_speed` is dead code in this planner.** `v_` appears in
`FootSteps_Planner` only at `planner_config.cpp:44` (loaded from config),
`plugin.cpp:109` (shown in a GUI form) and in the two entries added here. It is
never read in `footsteps_planner.cpp`, so nothing in step generation consults
it. The knob works and drives nothing.

**Where the limit is not.** `set_ref_vel` also sets `velocityControl = true`
(`Walking_controller.h:400`), so velocity mode is active and
`UpdatePlanner_input` pushes `reference_velocity` unscaled across the whole
preview horizon. `kinematics_cstr: [0.6, 0.08]` gives `d_h_x = 0.6 m`, which over
a `~1.3 s` step would permit `0.46 m/s`. Neither the mode, the reference, nor the
kinematic rectangle explains a `0.106 m` step.

**`StepRecoveryState` is not it.** Four `double` getters added to
`ismpc_walking` — `step_recovery`, `step_velocity_x`, `walking`, `stopped`,
doubles because `_read_scalar_output` rejects `bool` by design — read in-sim
while commanding `0.6 m/s`:

| t | recov | step_vx | walk | stop | ref_vx | meas_vx |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8.0 | 0 | 0.600 | 1 | 0 | 0.600 | +0.094 |
| 12.0 | 0 | 0.600 | 1 | 0 | 0.600 | +0.143 |
| 20.0 | 0 | 0.600 | 1 | 0 | 0.600 | +0.087 |

The recovery state never latches, the planner is handed the full commanded
velocity, and the robot walks normally without stopping. Everything upstream of
step generation is correct, so the limit is inside the planner or the ISMPC QP.
`step_velocity_x` mirrors `UpdatePlanner_input`'s own expression, so it reports
what the planner receives rather than what was asked for.

**`next_stp_cstr_ratio` is not the cap either.** Set explicitly to `1.0`, a
tenfold relaxation of the installed `0.1`, the envelope is unchanged (tracking
error in `m/s`, as above):

| commanded `vx` | 0.20 | 0.40 |
| --- | ---: | ---: |
| ratio `0.1` | 0.138 | 0.320 |
| ratio `1.0` | 0.138 | 0.321 |

A first attempt commented the key out instead, on the reasoning that mc_rtc
would fall back to the header default of `2`. That run was uninformative:
neither `LogisticController_ismpc.yaml` nor the package's `ismpc_walking.yaml`
defines the key, and `offset_static` is likewise read at
`Walking_controller.h:69` while absent from both — which shows only that mc_rtc
does not *throw* on a missing key, not what value the member ends up with. An
unchanged result was consistent with the default, with zero, and with no change
at all. The explicit `1.0` has no such ambiguity. The installed file has been
restored to `0.1` and verified byte-identical to its backup.

**Superseded candidate:** `next_stp_cstr_ratio`. The installed yaml sets
`ismpc.next_stp_cstr_ratio: 0.1` against a default of `2`
(`ControllerConfiguration.h:58`), a twentyfold tightening. `ISMPC_Solver.cpp:774`
multiplies the next step's ZMP constraint box by it, so `zmp_cstr_square:
[0.14, 0.08]` becomes roughly `1.4 cm x 0.8 cm`. A centre-of-pressure pinned that
close to the foot centre caps the achievable acceleration, and therefore the
gait speed, no matter what velocity is commanded — which is the observed
behaviour. Tested and refuted at an explicit `1.0`; see above.

**Superseded suspect:** `StepRecoveryState`. `UpdatePlanner_input` zeroes
`step_velocity` outright while it is set (`Walking_controller.cpp:431`), and it
latches to `true` at line 565 on the recovery path that also clears `Stop`. A
controller stuck in that state would plan for zero velocity regardless of the
command, which matches a constant slow gait that ignores every input. It is not
exposed as a scalar getter, and the neighbouring `robot_walking`, `stop_phase`
and `double_support` getters return `bool`, which `_read_scalar_output` rejects
by design — so confirming it needs either a new getter or widening that reader.

Measured speed of `0.085 m/s` at `ts` near `1.25 s` implies a step of about
`0.106 m`, which matches `FootManager.deltaTransLimit[0] = 0.1` closely enough to
be the next suspect, with `kinematics_cstr` behind it.

**The worker path is still broken.** In-process
(`use_worker_processes=False`) builds and runs; with worker processes the pool
still fails during `configure` with an empty payload. Training uses workers, so
the task keeps the installed default and the constants stay unwired.

**Re-measure if:** the plugin is rebuilt from clean, or the configure failure is
resolved.

**History:**
- 2026-08-31 — added the scalar entries and `datastore_scalar_holds`; the
  runtime path works standalone and fails inside the worker pool.
