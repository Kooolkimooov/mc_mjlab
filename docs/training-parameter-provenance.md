# Training parameter provenance

Where every explicit residual-balance training choice came from, including values
that were later removed. This is a lineage audit, not another tuning rationale:
the measurement details remain in the topic-specific documents linked throughout.

## Scope

**Current snapshot:** the 2026-08-21 working tree based on commit `c386cd3`,
mjlab 1.5.1, rsl-rl-lib 5.4.0. An uncommitted reward-constant extraction changes
names but preserves the audited values. Both registered residual-balance tasks
share the PPO configuration and differ only in the residual control channel and
its scale.

Included:

- every explicitly selected PPO/runner value, plus inherited defaults that affect
  optimization or reproducibility;
- every action, observation, reward, termination, event, curriculum, timing, and
  simulation value explicitly selected by the training environment;
- removed values that were once active, and ideas recorded as rejected before
  activation.

Excluded: viewer-only overrides, metrics that do not affect learning, worker-pool
performance knobs, robot parameters read from the selected `RobotModule`, installed
controller YAML, and PD gains. Those external controller inputs are snapshotted by
the runner and are covered by [robots.md](robots.md) and
[coupling.md](coupling.md); they are not training hyperparameters.

“Root” means the earliest source supported by the available record. It does not
mean that a cited paper supplied an exact number.

| Label | Meaning |
| --- | --- |
| **Paper-form** | A paper supports the algorithm, quantity, or semantics, not the exact value. |
| **Upstream-exact** | The exact value is copied through an identifiable code lineage. |
| **Local-measured** | The exact value was selected from this task's measurements or runs. |
| **Controller/hardware** | The value follows mc_rtc, mc_mujoco, or robot hardware. |
| **Local-design** | A deliberate local choice without a claimed external numeric source. |
| **Unresolved** | Git records the value but not why that exact number was first chosen. |

## Provenance graph

The dominant inheritance chain is:

`PPO/GAE papers → rsl_rl implementation defaults → Isaac Lab locomotion convention
→ mjlab G1 velocity/tracking configs → mc_mjlab initial residual task → local measurements`

The exact `(512, 256, 128)`, ELU, PPO clip `0.2`, value coefficient `1.0`, clipped
value loss, Adam `1e-3`, adaptive schedule, `gamma=0.99`, `lambda=0.95`, KL `0.01`,
gradient norm `1.0`, `5 × 4` update structure, 24-step rollout, and 50-iteration
save interval were already together in mjlab's initial public G1 velocity config.
The initial mc_mjlab block is a small recombination: its entropy `0.005` matches
mjlab G1 tracking while its save interval `50` matches G1 velocity. The same
family appears in Isaac Lab locomotion configurations. mc_mjlab subsequently
retuned only the entries called out below.

Primary code anchors:

- [mc_mjlab initial residual task](https://github.com/Kooolkimooov/mc_mjlab/commit/68e22bb36e5b7af764240752aa0fe48725db6326)
- [mjlab initial public G1 PPO config](https://github.com/mujocolab/mjlab/blob/a61e0fffb371f142c0b12d99d541ea65c2260862/src/mjlab/tasks/velocity/config/g1/rl_cfg.py)
- [current mjlab G1 tracking PPO config](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/tracking/config/g1/rl_cfg.py)
- [Isaac Lab locomotion PPO example](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0-beta2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/anymal_d/agents/rsl_rl_ppo_cfg.py)
- [rsl_rl PPO implementation](https://github.com/leggedrobotics/rsl_rl/blob/main/rsl_rl/algorithms/ppo.py)

## PPO algorithm

| Parameter | Current | Root | Lineage and status |
| --- | ---: | --- | --- |
| `class_name` | `PPO` | Paper-form / inherited | rsl_rl default; algorithm from [Schulman et al. 2017](https://arxiv.org/abs/1707.06347). |
| `optimizer` | Adam | Upstream-exact / inherited | rsl_rl and mjlab default; never explicitly ablated here. |
| `learning_rate` | `1e-3` | Upstream-exact | Copied unchanged from mjlab G1. A fixed `3e-4` is only proposed, never active. |
| `schedule` | adaptive | Upstream-exact | Copied unchanged from mjlab/rsl_rl. It changes LR by `1.5×` per minibatch event within `[1e-5, 1e-2]`. |
| `clip_param` | `0.2` | Paper-form / upstream-exact | Copied unchanged; rsl_rl's PPO default and common PPO convention. The paper motivates clipping, not this task's exact value. |
| `value_loss_coef` | `1.0` | Upstream-exact | Copied unchanged from mjlab G1/rsl_rl. No local ablation. |
| `use_clipped_value_loss` | `True` | Upstream-exact | Copied unchanged. Disabling it is proposed as the next isolated ablation, never active. |
| `max_grad_norm` | `1.0` | Upstream-exact | Copied unchanged from mjlab G1/rsl_rl. No local ablation. |
| `normalize_advantage_per_mini_batch` | `False` | Inherited | mjlab/rsl_rl default: normalize over the complete rollout. Not explicitly selected or tested here. |
| `num_learning_epochs` | `2` | Local-measured | Began at mjlab's `5`; reduced with minibatches after the adaptive LR hit its floor from iteration 0. Four schedule events cannot collapse `1e-3` to `1e-5` within one iteration. |
| `num_mini_batches` | `2` | Local-measured | Began at mjlab's `4`; paired with two epochs. At 128 envs and 256 steps this gives two 16,384-sample minibatches and four optimizer updates per iteration. |
| `entropy_coef` | `0.0005` | Local-measured | Began at `0.005`, the exact initial mjlab tracking value. Reduced 10× after policy std grew `0.2→0.52/0.62`; residual exploration is itself a physical disturbance. |
| `desired_kl` | `0.02` | Local-measured | Began at mjlab/rsl_rl `0.01`; doubled after the adaptive rate immediately pinned to `1e-5`. It is a schedule target, not PPO's clipping parameter. |
| `gamma` | `0.997` | Paper-form / local-measured | Began at mjlab `0.99`. Increased after a policy improved immediate ZMP agreement while worsening delayed falls; gives a 6.7 s geometric horizon at 50 Hz. |
| `lam` | `0.99` | Paper-form / local-derived | Began at mjlab `0.95`. The [GAE paper](https://arxiv.org/abs/1506.02438) supplies the estimator and bias/variance role; `0.99` was derived locally so `gamma×lambda` covers delayed falls. |

The implementation-choice caution comes from
[Andrychowicz et al.](https://arxiv.org/abs/2006.05990): low-level PPO choices
materially affect results. It does not prescribe this table's task-specific values.
The complete measurements are in [ppo.md](ppo.md).

## Actor and critic

| Parameter | Current | Root | Lineage and status |
| --- | ---: | --- | --- |
| actor/critic `hidden_dims` | `(512, 256, 128)` | Upstream-exact | Copied unchanged from mjlab's G1 locomotion/tracking config, whose initial public commit already contained it. No evidence in this repository traces the exact widths further than the Isaac Lab locomotion convention. |
| activation | ELU | Upstream-exact | Copied unchanged with the widths. |
| observation normalization | `True` both | Upstream-exact | Copied from mjlab G1 after its upstream config enabled both in commit `84c01e84`; not locally ablated. |
| actor class | `ZeroInitMLPModel` | Local-measured | Local subclass of rsl_rl `MLPModel`; zeroes the final mean head after measuring an untrained RMS action of `0.0944`. |
| critic class | `MLPModel` | Inherited | mjlab/rsl_rl default. |
| distribution | scalar Gaussian | Upstream-exact | Copied from mjlab/rsl_rl. |
| CNN / RNN | disabled | Inherited | `cnn_cfg=None`, `rnn_type=None`; inherited recurrent dimensions and CNN sharing settings are inert. |
| `init_std` | `0.1` | Local-design / measured check | Initial task chose `0.2`; lowered after a 500-iteration policy remained below baseline. The exact first `0.2` has no deeper recorded numeric root. |
| `std_range` | `(0.05, 0.30)` | Local-measured | Added after std fell to `0.044` in one run and exceeded `0.5` in earlier runs. No upstream equivalent supplied these bounds. |
| mean-head initialization | exactly zero | Local-measured | Makes iteration 0's deterministic action equal the zero-residual controller; fixes behavior not controlled by `init_std`. |

## Runner and training budget

| Parameter | Current | Root | Lineage and status |
| --- | ---: | --- | --- |
| `seed` | `42` | Inherited | mjlab runner default. The environment seed is `None`, so the runner/wrapper is responsible for seeding. Final claims are proposed to use seeds 42/43/44; only 42 is the default. |
| `num_envs` | `128` | Initial local design | Present from the first residual task. It is memory-limited by one ~70 MB controller per env; no paper root. |
| `num_steps_per_env` | `256` | Local-derived | History `24→48→96→256`; current 5.12 s rollout covers the 4.58 s window containing 95% of the GAE mass. |
| `POLICY_STEPS_PER_ENV` | `128,000` | Local-design | Replaces a typed iteration budget. At 256 steps this derives 500 iterations. |
| `max_iterations` | `500` derived | Local-derived | `round(128000/256)`, minimum one. It is not an independent tuning parameter anymore. |
| total transitions | `16,384,000` | Derived | `128 envs × 128,000 steps/env`; changes with CLI `num_envs`. |
| `save_interval` | `20` | Local-derived | Was mjlab's 50. Changed with rollout 96→256 to preserve approximately one checkpoint per 5k policy steps/env. |
| `logger` | W&B | Initial local choice | Explicit since the first task; mjlab default is also W&B. |
| W&B project/tags | `mjlab` / empty | Inherited | mjlab runner defaults. |
| run name | empty | Inherited | mjlab default; timestamp and experiment ID identify the run unless CLI overrides it. |
| experiment name | full task ID | Local reproducibility | Position and torque registrations use different IDs so resume cannot cross control modes. |
| `obs_groups` | actor/critic mapping | Inherited | mjlab default `actor→actor`, `critic→critic`. |
| `clip_actions` | `None` | Inherited | Runner does not add another clip; the action term owns physical scale and clip. |
| resume | `False` | Inherited | mjlab default; CLI may override. |
| resume selectors | run `.*`, checkpoint `model_.*.pt` | Inherited | mjlab defaults; dormant unless resume/load is requested. |
| upload model | `True` | Inherited | mjlab default. |
| runner implementation | `ResidualBalanceOnPolicyRunner` | Local instrumentation | Adds diagnostics and controller/config provenance without changing PPO updates. |

Iteration history and matched-budget comparisons are in [ppo.md](ppo.md#training-budget).

## Timing and simulation

| Parameter | Current | Root | Lineage and status |
| --- | ---: | --- | --- |
| physics `timestep` | `0.001` s | Controller/hardware | Exact value from `mc_mujoco`'s HRP5P/JVRC XML. |
| MuJoCo integrator | Euler | Controller/hardware | Exact `mc_mujoco` model setting/default behavior. |
| solver / Jacobian | Newton / dense | Controller/hardware | Exact `mc_mujoco` robot XML values. |
| solver iterations | `50` | Controller/hardware | Exact `mc_mujoco` robot XML value. |
| tolerance | `1e-10` | Controller/hardware | Exact `mc_mujoco` robot XML value. |
| cone | pyramidal | Inherited/controller | mjlab default and `mc_mujoco` XML agree. |
| `frameskip` | `2` | Controller-derived | mc_rtc period divided by 1 ms sim period: controller at 500 Hz. |
| policy `decimation` | `20` | Initial local design | 50 Hz policy, holding a residual across ten controller periods. Exact numeric root beyond the initial task is unresolved. |
| reward scaling | by `step_dt` | Inherited | mjlab default; all configured reward weights are rates per second. |
| terrain | flat plane | Initial local design | No rough-terrain curriculum; contact properties come from external assets. |
| `njmax` | `1500` | Upstream-exact / local validation | Copied from mjlab humanoid locomotion and retained because a sprawled robot otherwise overflows the per-world constraint budget. |
| `nconmax` | `100` | Local safety budget | Present from the first task; intended for falls. Exact external numeric root is unresolved. |
| gravity / `impratio` | `-9.81` m/s² / `1.0` | Inherited | mjlab/MuJoCo defaults; not locally tuned. |
| line search / CCD | `50`, `0.01`; CCD `50` | Inherited | Unoverridden mjlab `MujocoCfg` defaults; no local ablation. |
| episode length | `90` s | Local-measured | History `10→16→60→90`; current value follows the installed indefinite-walk FSM and restores ~22% baseline survival after adding push warm-up. |
| finite horizon | `False` | Inherited / paper-form | mjlab default. The 90 s cap is an artificial training truncation, so it bootstraps; consistent with [Pardo et al.](https://arxiv.org/abs/1712.00378). |
| automatic reset | `True` | Inherited | Required by rsl_rl's rollout loop; mjlab default. |

The immediate source is the installed, untracked
`$HOME/workspace/install/share/mc_mujoco/HRP5P/xml/HRP5Pmain.xml`. The
[mc_mujoco repository](https://github.com/Kooolkimooov/mc_mujoco) does not track
the HRP5P asset, so there is no honest public per-line link for these exact values.

## Residual action

| Parameter | Current | Root | Lineage and status |
| --- | ---: | --- | --- |
| residual joints | legs only | Local-design | First task used the robot's full residual set; changed to legs because this is balance/locomotion. Robot-specific exclusions remain inherited from `RobotModule`. |
| position scale/clip | `±0.20` rad | User-directed / unmeasured | History `0.1→0.01→0.03→0.01→0.20`. The current 20× sweep was requested, not evidence-backed, and can request 4.4–5.4× hardware torque limits. |
| torque scale/clip | `±10` Nm | Local-design | Channel-specific authority; exact numeric root is unresolved in Git notes. |
| scale map | `{'.*': scale}` | Local safety design | Re-expresses the former scalar while ensuring every actuator is explicitly covered and clipped. |
| raw clip point | `1.0` | Derived | `clip == scale`, so a unit raw action reaches the physical bound. |
| `GATE_STRENGTH` | `0.0` | Paper-form then locally pruned | Gate form was inspired by the stability-alignment idea in [Jayasinghe et al.](https://arxiv.org/abs/2603.07775); enabled at `1.0`, then disabled after attenuation stayed flat and the tracking gain vanished. |
| `GATE_ALPHA_REF` | `0.5` rad/s | Local-measured | Norm reference selected from 60k env-steps; dormant while strength is zero. |

The residual paper supports bounded/directionally aligned authority, not this
repository's exact gate equation or constants. See
[residual-authority.md](residual-authority.md).

## Observations

The initial six-term observation skeleton—base linear/angular velocity, projected
gravity, relative joint position/velocity, and previous action—was copied from
mjlab's velocity task. Actor corruption plus a noise-free critic also follows
mjlab's asymmetric actor/critic pattern.

| Term or parameter | Current | Root | Lineage and status |
| --- | ---: | --- | --- |
| base linear velocity noise | `±0.02` | Local-measured | Began at copied `±0.1`; set from signal RMS `0.110`, rather than mjlab's faster locomotion scale. |
| base angular velocity noise | `±0.03` | Local-measured | Began at copied `±0.2`; set from RMS `0.121`. |
| projected gravity noise | `±0.05` | Upstream-exact / locally retained | Same as mjlab velocity; checked against local RMS `0.577`. |
| joint position noise | `±0.01` | Upstream-exact / locally retained | Same as mjlab. Encoder-bias application was later fixed with `biased=True`. |
| joint velocity noise | `±0.05` | Local-measured | Began at copied `±1.5`; original noise was about 12× this task's RMS signal. |
| controller position-error noise | `±0.01` | Local-design | Mirrors joint-position corruption so the actor cannot reconstruct privileged true angles. |
| short history | `5` frames | Initial local design | 0.1 s at 50 Hz on base velocities and executed actions. Exact five-frame root is unresolved. |
| `CONTROLLER_HISTORY` | `20` frames | Local-measured/design | Increased controller/gait channels from 5 to 20 to cover 0.4 s of a roughly 1 s gait cycle. |
| controller ref position/velocity/error | 20 frames | Coupling-derived | Added because residual control requires the base controller's plan and tracking error. |
| planned ZMP and CoM velocity | 20 frames | Local failure analysis | Added after rewards scored plan quantities the actor could not see. |
| foot load share / sole velocities | 20 frames | Local proxy design | Sim-side proxies because mc_rtc datastore gait phase is unavailable. |
| inferred gait phase | 20 frames | External-form / local-measured | Sin/cos representation follows `leo_mjlab`'s phase convention; the load-difference phase-plane estimator is local. `PHASE_RATE_REF=7.1` is measured RMS `|d_dot|`. |
| minimum normal force | `20` N | Unresolved local constant | Shared hidden default for load share, ZMP/DCM, slip, and gait phase. Git records no calibration for the exact threshold. |
| critic push recency | `tau=2.0` s | Local-design | Bounded privileged input; shares the two-second recovery timescale but has no recorded external numeric root. |
| critic last push velocity | exact sampled delta | Local-design | Privileged exogenous input to reduce return variance. |
| critic encoder bias | sampled bias | mjlab mechanism / local asymmetric use | Actor suffers it; critic observes it. |
| critic measured ZMP offset | exact sim-side value | Local asymmetric design | Gives the critic true CoP while actor retains observer-realistic signals. |
| observation delays/clips/scales | none | Inherited | mjlab term defaults: no delay, no observation clip, unit scale. |
| history layout | flattened, term-major | Inherited | mjlab term/group defaults. |
| observation NaN policy | disabled | Inherited | mjlab group default; rewards separately sanitize non-finite values. |

See [observations.md](observations.md) for dimensions and measurements.

## Rewards

All weights below are multiplied by `step_dt=0.02`. The DCM quantity and its
relationship to CoP are Paper-form, supported by
[Zhang et al.](https://doi.org/10.3390/mi13071095); exact kernels, widths, and
weights are local.

| Reward | Current parameters | Root | Lineage and status |
| --- | --- | --- | --- |
| termination penalty | `-200` | Local-measured | Began at `-2000`; reduced because a fall produced a 1000× gradient outlier and the critic/LR schedule failed. |
| upright | `-2.0`, squared flat-orientation error | Upstream-exact | Copied unchanged from the initial task/mjlab primitive. Exact weight root beyond the initial task is unresolved. |
| DCM stability | weight `+1.0`, std `0.05` m | Paper-form / local-measured | DCM form from humanoid walking literature; std scores baseline 0.57 and p90 push landing 0.05. Current target is command-relative, validated by a three-regime gate. |
| recovery DCM | weight `+1.0`, std `0.05` m, window `2.0` s | Local-measured, provisional window | Replaced recovery ZMP tracking. Window came from a pre-fix recovery profile and is explicitly marked for remeasurement. |
| angular momentum | `-0.005` | Local-measured | Local QP-gap idea; first guess `-0.05` would consume 58% of the objective, so reduced 10×. |
| foot slip | `-1.0`, 20 N load threshold | Upstream-form / local-measured | Locomotion penalty form exists in mjlab; local implementation uses sole velocimeters. First proposed `-0.1` was inert and was raised 10× as a guard. |
| torque margin | base `-0.05`, soft ratio `1.0`, warm-up 25 steps | leo_mjlab-exact form / local measurement | Peak/log1p form and initial weight descend from `leo_mjlab` commit `4412aa33`; threshold moved there from 0.7 to the hardware limit 1.0. Local warm-up suppresses a measured 22× reset transient. |
| torque curriculum | `-0.05→-0.20→-0.50` at policy steps `0, 48k, 96k` | leo_mjlab-form / local-derived | Same gradual-safety rationale as leo_mjlab; current stages are local and expressed in common policy steps. |
| residual magnitude | `-0.1`, squared raw action clamped to `±1` | Upstream-form / local fix | Weight came from the July reward rework; clamp added after an irrelevant raw action near 16 made the value loss diverge. |
| residual rate | `-0.1`, squared change of raw actions clamped to `±1` | Upstream-exact weight / local fix | Exact weight matches mjlab locomotion `action_rate_l2`; local implementation clamps both endpoints to the physical action bound. |

No paper is claimed as the numeric root for any reward weight or exponential
standard deviation. Detailed baseline shares and ablations are in
[reward-shaping.md](reward-shaping.md).

## Disturbances and reset randomization

| Parameter | Current | Root | Lineage and status |
| --- | ---: | --- | --- |
| push linear range | x/y `[-0.4, 0.4]` m/s | mjlab form / local-measured | First copied mjlab's `±0.5`; angular part was later disabled and linear magnitude calibrated to ~20.8% baseline survival. |
| push angular range | roll/pitch `0` | Local-design | Began at copied `±0.5`; disabled since the task's walking-controller tuning era. |
| push interval | uniform `5–7` s | Local-design | Began at `2–5`; changed with difficulty tuning. Exact range is not paper-derived. |
| push warm-up | `10` s | Local-measured | Added after 528 baseline episodes showed 48% of deaths came from the first push during posture settling. |
| base reset x/y | `±0.5` m | mjlab-exact | Copied from mjlab velocity reset range; safe because the controller frame is reconciled on reset. |
| base reset yaw | `±pi` | mjlab-exact | Copied from mjlab velocity reset range. |
| base reset velocity | none | Controller constraint | mc_rtc reset accepts encoders/pose but no velocity, so random velocity would create inconsistent state. |
| encoder bias | uniform `±0.01` rad | mjlab/tracking exact | Same exact range as mjlab tracking; applied at startup and now actually consumed by actor/controller-visible joint position. |
| friction / mass / CoM DR | absent | Deliberately pruned initial concept | The first docstring mentioned friction and CoM randomization, but they were never wired in the initial commit. The task currently perturbs pushes, reset pose, and encoder bias only. |

See [difficulty.md](difficulty.md).

## Terminations

| Term | Current | Root | Lineage and status |
| --- | ---: | --- | --- |
| time out | 90 s, truncation | Paper-form / local duration | Artificial continuing-task boundary; bootstraps. |
| fell over | tilt `45°` | Local-design | History `60°→45°`. Exact threshold is not traced to a paper or upstream task. |
| collapsed | height `<0.7× nominal`, only if not tilted | Local-design | Replaced fixed `nominal_height-0.25`; ratio is robot-portable but exact 0.7 is unresolved. Made exclusive with tilt so labels partition failures. |
| controller failed | terminal | Controller semantics | Base controller failure is part of the task outcome. |
| worker failed | truncation | Paper-form / infrastructure semantics | Exogenous process failure bootstraps rather than paying the fall penalty, following time-limit/exogenous truncation logic. |

## Pruned after being active

| Parameter or term | Active values | Root | Why pruned |
| --- | --- | --- | --- |
| alive reward | `+2`, then `+1` | Initial local / mjlab primitive | Episode sum mostly measured duration and encouraged survival without distinguishing walking. Replaced by termination penalty and denser objectives. |
| stance-height reward | `-20` at nominal root height | Initial local | Fixed stance height paid a walking controller to stop; walking naturally dips below half-sitting height. |
| plan-motion reward | `+1.5`, tanh scale `0.5` | Local-measured construction | Controller reference norm rises before a fall because the controller thrashes, so the reward had the wrong sign. |
| progress reward | `+1.5`, tanh speed `0.1` m/s | Local-measured construction | Removed with the alive/walking shaping family during reward simplification; function remains only as a diagnostic building block. |
| ZMP tracking reward | std `0.05`, weight `1.0→0.05` | Controller quantity / local-measured | Pruned 2026-08-19: agreement with a plan the QP already optimizes; near-null policy delta and +standing bias. Metric retained. |
| CoM velocity tracking reward | horizontal std `0.05`, vertical `0.005`, weight `1.0→0.05` | Controller quantity / local-measured | Pruned 2026-08-19: negative against baseline in every comparison. Metric retained. |
| recovery ZMP tracking | std `0.05`, weight `1.0`, window `2.0` s | Local-measured | Replaced by recovery DCM so the residual is paid for stability rather than reproducing the QP plan. |
| zero-referenced DCM | std `0.05`, weight `1.0` | Paper-form implemented incorrectly for walking | Corrected to command-relative DCM on 2026-08-20 because zero-relative scoring rewarded standing over commanded walking. |
| coherence gate enabled | strength `1.0`, alpha ref `0.5` | Paper-form / local-measured | Disabled after 1500 iterations: mean attenuation changed only `+0.008` and its apparent tracking benefit disappeared. Code retained at strength zero. |
| residual scale `0.1` | position `0.1` rad | Initial local | Through real gains produced 220–270% hardware-limit authority; reverted to `0.01`. |
| residual scale `0.03` | position `0.03` rad | Local authority probe | Intended to improve CoP authority; deterministic evaluation instead widened all tracking deficits, so reverted. |
| termination penalty `-2000` | one terminal step | Unresolved initial local | Produced critic targets and KL behavior far outside dense-reward scale; reduced to `-200`. |
| PPO `5×4` updates | five epochs, four minibatches | Upstream-exact | Twenty adaptive-LR events could collapse LR 3325× in one iteration; changed to `2×2`. |
| `gamma=0.99`, `lambda=0.95` | upstream defaults | Upstream-exact | Credit/GAE traces were too short for delayed topples; changed to `0.997/0.99`. |
| rollouts `24/48/96` | 0.48/0.96/1.92 s | Upstream then local intermediates | Progressively lengthened to cover the corrected GAE mass window; current 256. |
| `init_std=0.2` | scalar Gaussian | Unresolved initial local | Lowered to 0.1; zero mean initialization was the more important fix. |
| unbounded learned std | no range | Upstream behavior | Both collapse below 0.05 and growth above 0.5 were observed; bounded locally. |
| action penalties beyond the physical clip | unclamped raw action | Upstream primitive | Could change value targets without changing robot behavior; both magnitude and rate are now clamped at raw ±1. |

## Rejected before activation

| Proposal | Root | Reason it never became a training parameter |
| --- | --- | --- |
| measured-ZMP support-polygon margin | Locally proposed | CoP lies in the active contact hull by construction; nearly tautological. |
| commanded-vs-measured joint-position reward | Locally proposed | Stiff PD keeps the error small while falling and reference-plus-residual turns it into another residual penalty. |
| fixed learning rate `3e-4` | PPO audit proposal | Requires logging the schedule-driving KL and an isolated ablation first. |
| no clipped value loss | PPO audit proposal | Identified as the first clean PPO ablation; not yet run. |
| zero entropy | PPO audit proposal | Must be isolated after rollout/lambda questions; never active. |
| symmetry augmentation | Common locomotion technique | No verified left/right map for every history and controller channel, and the controller can have real phase asymmetry. |
| ankle-specific residual scales | Local authority analysis | Supported by config machinery but not measured by a joint-subset probe. |
| friction, mass, and CoM randomization | mjlab locomotion convention | Not justified for the present controller-residual question and not active. |

## Unresolved roots and audit flags

The following exact values are choices, not established constants: torque residual
scale `10` Nm, policy decimation `20`, `nconmax=100`, tilt `45°`, collapse ratio
`0.7`, minimum contact force `20` N, initial short history `5`, push interval
`5–7` s, and critic push-recency `tau=2` s. They have plausible engineering
roles, but the repository does not contain evidence tracing the exact numerals to
a paper, upstream environment, or calibration. They should be described as local
design values until measured.

Two audit hazards surfaced:

1. The stale statement that `residual_rate` was still unclamped was corrected in
   `reward-shaping.md`; commit `8b80696` and current `mdp.action_rate_l2` clamp
   both actions at `RAW_CLIP`.
2. The current position authority `0.20` rad invalidates safety measurements made
   at `0.01` rad. It is explicitly an unmeasured sweep, not a validated default.

## Source register

| Source | What it actually supports |
| --- | --- |
| [Schulman et al., PPO](https://arxiv.org/abs/1707.06347) | Clipped surrogate PPO form; not this task's full numeric block. |
| [Schulman et al., GAE](https://arxiv.org/abs/1506.02438) | GAE form and lambda bias/variance tradeoff. |
| [Andrychowicz et al., on-policy choices](https://arxiv.org/abs/2006.05990) | Need to treat implementation choices as experimental variables. |
| [Pardo et al., time limits](https://arxiv.org/abs/1712.00378) | Bootstrapping artificial/exogenous truncations. |
| [Zhang et al., DCM walking](https://doi.org/10.3390/mi13071095) | DCM/CoP walking dynamics and stability interpretation. |
| [Jayasinghe et al., residual recovery](https://arxiv.org/abs/2603.07775) | Bounded residual authority and directional alignment concept; not the local gate equation. |
| [rsl_rl configuration](https://github.com/leggedrobotics/rsl_rl/blob/main/docs/guide/configuration.rst) | Optimizer defaults and exact adaptive-PPO parameter meanings. |
| [mjlab G1 velocity config](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/velocity/config/g1/rl_cfg.py) | Main structural ancestor; the initial task instead took entropy `0.005` from the sibling G1 tracking family. |
| [leo_mjlab torque penalty](https://github.com/Leonassim/mjlab/commit/4412aa33) | Peak/log1p hardware-margin form, `-0.05` initial weight, and soft ratio `1.0`. |
| installed `mc_mujoco/HRP5P/xml/HRP5Pmain.xml` | Immediate source of simulation timestep and solver settings; the asset is not tracked in the public mc_mujoco repository. |
