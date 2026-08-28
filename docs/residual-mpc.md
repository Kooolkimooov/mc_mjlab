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
