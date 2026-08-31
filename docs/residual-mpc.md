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

**Current:** `vx` in `(0.0, 0.30)`, `vy` and `wz` held at zero. Chosen to straddle
the prior's measured boundary rather than sit inside it. An earlier revision of
this line said `(0.0, 0.60)`; the constant has read `(0.0, 0.30)` throughout.

**Most of the box earns a flat zero, which is not the same as unearned.** The
prior tops out near `0.085 m/s` and `LINEAR_TRACKING_SIGMA` is `0.06`, so the
tracking term against a prior-speed rollout is:

| commanded `vx` | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `exp(-(err/sigma)^2)` | 0.94 | 0.31 | 0.025 | `5.2e-4` | `2.7e-6` |

Above about `0.20` the term is numerically dead: the policy cannot tell `0.25`
from `0.30`, so there is no gradient toward closing the gap, only a flat region.
Headroom the optimiser cannot feel is not headroom. The usable learning band is
roughly `vx <= 0.18`, which argues for widening `sigma` or capping the box near
`0.18` rather than leaving two thirds of it flat.

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
note under `## set_ts`. The runtime setter is the working knob, and like every
other workspace entry here a clean rebuild drops it.

**Raising it is not yet validated as safe.** `0.2714 m/s` was held for 15 s in
two environments with pushes off and no termination. Nothing has been measured
about falls, the ZMP margin, or whether the stabilizer keeps up at a step length
six times the tuned one, and `zmp_cstr_square` is still sized for HRP4.

**Re-measure if:** the plugin is rebuilt from clean, or the planner starts
reading its configuration.

**History:**
- 2026-08-31 — identified as the cap after `mean_speed`, `next_stp_cstr_ratio`,
  `StepRecoveryState`, `zmp_cstr_square` and footstep-QP failure were each
  refuted; two earlier config experiments were inert because the yaml is unread.

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
`get` / `set 0.30` / `get` reads `0.1 -> 0.3`. **A clean workspace rebuild drops
this**, and nothing else in this repository would explain the task getting
slower afterwards.

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
