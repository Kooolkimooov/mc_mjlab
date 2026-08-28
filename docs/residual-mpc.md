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
