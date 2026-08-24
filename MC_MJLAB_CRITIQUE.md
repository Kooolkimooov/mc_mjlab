## Bottom line

The residual-policy infrastructure is careful, but the current training problem
is mis-specified. The evidence points away from “insufficient policy capacity or
residual authority” and toward three coupled problems:

1. The policy is allowed—and rewarded—to modify the controller continuously,
   without a strong constraint preserving nominal gait quality.
2. The mathematical policy is unbounded while the physical action is clipped and
   not projected onto hardware-feasible torque/position limits.
3. A useful transient checkpoint can be trained past its optimum, while the
   evidence used for promotion comes from one training seed.

I treated “current” as the working tree: HRP5P with `LogisticController_ismpc`
in the modified [mc_rtc.yaml](/home/martin/git/mc_mjlab/etc/mc_rtc.yaml:1), not
the repository’s committed default.

## Current setup

| Component     | Current value                                                                                                                                             |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Rates         | 1 kHz physics, 500 Hz asynchronous mc_rtc, 50 Hz policy                                                                                                   |
| Action        | Position residuals on all 12 leg joints, uniformly bounded at ±0.01 rad; gate disabled                                                                    |
| Observation   | 1,219-dimensional flattened actor input, including 0.4 s histories of controller and gait proxies                                                         |
| Reward        | Continuous command-relative DCM reward plus a duplicate two-second post-push DCM reward; action, rate, slip, upright, momentum, and soft torque penalties |
| Disturbance   | Independent root-velocity changes in x/y, each ±0.4 m/s every 5–7 s; no angular push                                                                      |
| Randomization | Encoder bias and reset pose; no mass, CoM, friction, actuator, contact, or latency randomization                                                          |
| PPO           | 128 environments, 256-step rollout, 500 iterations; 2 epochs × 2 minibatches; adaptive LR from \(10^{-3}\); one-seed experimental screens                 |

See the executable
[environment configuration](/home/martin/git/mc_mjlab/src/mc_mjlab/tasks/residual_balance/residual_balance_env_cfg.py:49),
[PPO configuration](/home/martin/git/mc_mjlab/src/mc_mjlab/tasks/residual_balance/residual_balance_ppo_cfg.py:18),
and the local
[bibliography](/home/martin/git/knowledge_base/humanoid_residual_policies_extended.bib:3).

## Main criticisms

### 1. The objective encourages chronic intervention instead of recovery-specific correction

The actor may apply residuals on every step. Meanwhile, `dcm_stability` pays
continuously and `recovery_dcm` adds a second copy for two seconds after each
push. Although the DCM error is command-relative, it still permits trade-offs
between CoM velocity and CoP rather than explicitly constraining nominal
velocity and gait tracking.

The repository’s own evidence is unusually decisive:

- Deterministic residual magnitude grows in every reported run and is not
  eliminated by zero entropy, fixed exploration variance, or tripling the
  magnitude penalty
  [residual-growth.md](/home/martin/git/mc_mjlab/docs/residual-growth.md:13).
- The residual opposed the controller’s joint-velocity plan on 58.9% of measured
  steps.
- CoM-velocity tracking is negative against baseline in every recorded
  comparison
  [reward-shaping.md](/home/martin/git/mc_mjlab/docs/reward-shaping.md:789).
- A wider recovery reward produced a genuinely useful checkpoint at iteration
  340—hazard ratio 0.691 and survival +15.6 percentage points—but continued
  training grew the residual until iteration 499 had hazard ratio 1.160
  [V4 results](/home/martin/git/mc_mjlab/MC_MJLAB_TRAINING_CONFIG_PROPOSAL_V4.md:101).

That is the signature of objective overoptimization, not inadequate authority.

[Jayasinghe et al.](https://arxiv.org/abs/2603.07775) find that
performance-conditioned activation, directional coherence, and transient
filtering are more important than increasing residual adaptation capacity. Their
online adaptation setting is different, so their numeric parameters should not
be copied, but the structural lesson fits the local evidence closely.

Recommendation:

- Turn nominal gait preservation into an explicit constraint or lexicographic
  criterion: commanded CoM/base velocity, foot-slip, and possibly
  controller-reference tracking may degrade only within a defined budget.
- Apply residual authority through a deployable recovery detector based on DCM
  deviation, IMU/load state, or predicted controller error, with authority
  decaying after recovery.
- Do not simply re-enable the existing coherence gate. Its whole-vector
  position-residual versus velocity-reference comparison showed nearly constant
  attenuation and its apparent improvement disappeared. A replacement should be
  performance-conditioned and preferably per-joint.

### 2. The bounded-action contract is internally inconsistent and not hardware-safe

The actor uses an unbounded Gaussian, but the environment clips its output. The
action penalties also clamp raw actions before scoring them
[mdp.py](/home/martin/git/mc_mjlab/src/mc_mjlab/tasks/mdp.py:62). Consequently,
once the actor mean leaves \([-1,1]\):

- The physical action and penalty are flat.
- PPO’s Gaussian likelihood still models the unclipped action.
- The actor can continue moving its mean through a physically irrelevant region.

This likely contributes to deterministic mean growth. Large-scale
continuous-control experiments found bounded action transforms preferable to raw
clipping, while also supporting small/zero policy-mean initialization
[Andrychowicz et al.](https://arxiv.org/abs/2006.05990).

The safety problem is more serious: `torque_margin` begins penalizing only after
the hardware limit has already been exceeded, and the simulation actuator itself
is not clamped. Even ±0.01 rad corresponds to roughly 22–27% of a leg joint’s
hardware torque limit through the PD gains; larger rejected scales exceeded
hardware capability
[residual-authority.md](/home/martin/git/mc_mjlab/docs/residual-authority.md:14).

Recommendation:

- Replace hard clipping with a tanh-squashed or otherwise correctly bounded
  distribution.
- Penalize the executed physical residual, not a separately clipped raw
  variable.
- Project final commands onto joint position, velocity, and torque feasibility
  before simulation and deployment.
- Move the torque guard below 100% if it is intended to provide margin; retain a
  hard limit independently of the reward.

[RobotDancing](https://arxiv.org/abs/2509.20717) likewise uses per-DOF residual
bounds and hard feasibility projection.

### 3. Uniform residual authority across all leg joints is not justified

All 12 leg joints receive the same ±0.01 rad range and the same scalar
exploration standard deviation. Yet joint leverage, PD gains, torque limits, and
risk differ substantially.

The local experiments correctly show that globally raising authority to 0.03 or
0.20 rad is harmful. Therefore the next step should not be a larger global
scale. It should be a joint-sensitivity study:

- Measure transient DCM/CoP response and torque cost for each joint or joint
  group.
- Compare ankle-only, sagittal-only, hip/knee, and full-leg residuals.
- Normalize bounds by measured sensitivity and hardware margin.
- Consider per-joint gates or standard deviations.

[RobotDancing](https://arxiv.org/abs/2509.20717) reports advantages from
selective residualization, although its dance task does not identify the correct
HRP5P joint subset.

There may also be a more effective action channel than joint offsets. In push
recovery,
[Kasaei et al.](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2023.1004490/full)
found that modulating gait-controller parameters was more effective than
residual actions alone, and combining the two was strongest. Safe mc_rtc
parameters such as step placement, swing timing, CoM/DCM targets, or admittance
gains merit an ablation against joint-only residuals.

### 4. The training distribution is too narrow for robustness claims

The current “push” teleports root velocity rather than applying a force over
time. Because x and y are sampled independently from a square:

- The maximum linear perturbation is \(0.4\sqrt{2}=0.566\) m/s.
- About 21.5% of samples exceed 0.4 m/s in total norm.
- Direction and magnitude are not uniformly distributed in polar coordinates.
- No angular disturbance is trained.

This is a useful standardized stress test, but it can teach a policy specialized
to instantaneous velocity discontinuities. It is not a sufficient model of
physical pushes or dynamics shifts.

There is also no dynamics randomization beyond encoder bias
[parameter provenance](/home/martin/git/mc_mjlab/docs/training-parameter-provenance.md:230).
This makes the current result a nominal-simulator recovery policy rather than a
sim-to-real robust residual.

Recommendation:

- Sample impulse direction uniformly, with an explicit magnitude curriculum.
- Apply finite-duration forces at varying torso heights; add angular impulses
  and held-out directions/magnitudes.
- Stage mass, inertia/CoM, friction/contact, PD gain, motor strength,
  observation delay, controller latency, and force-sensor randomization.
- Keep the initial ranges narrow enough that the base controller remains
  feasible, then widen only after the residual reliably beats the nominal
  baseline.

[I-CTRL](https://arxiv.org/abs/2405.08726),
[RuN](https://arxiv.org/abs/2509.20696), and RobotDancing all combine bounded
residuals with materially broader dynamics or sensor variation.

### 5. The observation stack substitutes width for controller state

The actor input grew from 284 to 1,219 dimensions because controller and gait
channels carry 20 flattened frames
[observations.md](/home/martin/git/mc_mjlab/docs/observations.md:61). This is
compensating for inaccessible mc_rtc state: exact gait phase, next contact,
touchdown time, and footstep plan are replaced with force-derived phase and
sim-only sole velocimeters.

The policy can therefore overfit the exact simulated gait cycle and sensors that
may not exist unchanged on hardware.

Recommendation:

- Expose contact phase, touchdown time, and planned footsteps through the mc_rtc
  bindings.
- Compare the flattened history against a recurrent or compact temporal encoder;
  RuN uses recurrence rather than a large raw history.
- Verify that sole velocity is available through a deployable estimator, then
  train with its noise and delay. Otherwise remove it from the actor and retain
  it only for critic/reward use.
- Ablate history lengths rather than assuming 20 frames is optimal.

### 6. PPO tuning is compensating for a learning-rate scheduler pathology

The 2×2 PPO update was chosen mainly because the adaptive learning-rate
scheduler runs inside the minibatch loop. It reduces the number of possible
\(1.5\times\) LR changes from 20 to four per iteration, but does not correct the
underlying semantics. The repository records both immediate LR collapse with
more updates and delayed divergence with four updates
[ppo.md](/home/martin/git/mc_mjlab/docs/ppo.md:61).

Because training is approximately 99% rollout-bound, additional gradient passes
are essentially free. Broad on-policy studies find repeated passes important for
sample efficiency and also report that PPO value clipping can hurt
continuous-control performance
[Andrychowicz et al.](https://arxiv.org/abs/2006.05990).

Recommendation:

- Log the exact KL used by the scheduler.
- Adapt LR once per rollout, or use KL early stopping within the update.
- Then compare fixed LR values against the repaired adaptive schedule and
  re-test 2×2 versus more epochs.
- Include `use_clipped_value_loss=False` as an ablation, not an assumed
  improvement.
- Leave \(\gamma=0.997\), \(\lambda=0.99\), rollout 256, entropy 0.0005, and
  learned bounded standard deviation alone initially; these already have useful
  local evidence.

### 7. Checkpoint and seed selection are presently too weak

Many evaluation episodes from one trained model quantify environment
stochasticity, not training stochasticity. The current screens use one training
seed, while the best observed behavior is explicitly non-monotonic across
checkpoints.

Before promotion:

- Run deterministic held-out validation at every saved checkpoint.
- Select by a composite gate covering recovery DCM, hazard/survival, commanded
  velocity, slip, and residual magnitude—not training return.
- Use at least 3–5 independent training seeds for promotion; use a power
  calculation or more seeds when effects are small.
- Treat checkpoint comparisons and multiple reward variants as multiple
  comparisons.

This distinction is central to the statistical guidance of
[Colas et al.](https://arxiv.org/abs/1806.08295).

## Practical priority order

1. Fix the action contract: squashed distribution, executed-action
   regularization, and hard feasibility projection.
2. Add nominal-gait constraints and checkpoint-based early stopping.
3. Replace the failed whole-vector gate with recovery-conditioned, per-joint
   authority.
4. Test selective joint residuals and safe mc_rtc parameter modulation.
5. Add staged physical disturbances and dynamics/latency randomization.
6. Repair the PPO KL schedule, then revisit epochs, LR, and value clipping.
7. Promote only after multi-seed validation.

I would specifically retain the zero-initialized actor, position-residual
formulation, conservative 0.01-rad global baseline, asymmetric critic, and long
discount/GAE horizon. The local evidence does not support increasing residual
scale or further tuning entropy and magnitude penalties before correcting the
objective and action interface.
