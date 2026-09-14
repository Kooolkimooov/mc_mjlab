# Residual-growth screen

Four one-seed screens ran on 2026-08-21 for 188 iterations, 48,128 policy steps
per environment, with 128 environments and 30 controller workers. Each model 187
was compared deterministically with its own zero-residual arm using 64 environments
and 30 workers for 12 minutes. Three comparisons retained K=4 episodes per
environment; `rg-fixedstd010` retained K=3 in both arms. Statistics below use one
mean per environment.

Raw training events are under `logs/rsl_rl/`; paired episode rows and reports are
under `logs/comparisons/`, both named for the run and model 187.

## entropy_coef

**Current:** this screen selected `0.0005`: zero entropy did not suppress
deterministic policy-mean growth enough to pass. The active training config moved
to `0.00005` on 2026-09-01 together with ResidualMPC's tighter `0.15` standard-
deviation ceiling; that cross-task setting has not yet repeated this screen.

**Re-measure if:** the action distribution, standard-deviation parameterization
or residual reward changes.

**History:**
- `rg-entropy0` set the coefficient to zero. Its last-60-iteration
  `policy_mean_rms` was 0.04774 against 0.04696 for `rg-ref01` (+1.7%), and its
  slope was only 4.8% smaller. Deterministic tracking was +1.04% (p=0.675) and
  hazard ratio 1.049, but the arm failed the mean-growth gate.

## learn_std

**Current:** `True`. Fixed standard deviation reduced sampled exploration but did
not clear the policy-mean gate.

**Re-measure if:** a later objective produces a different mean-growth mechanism
or learned standard deviation reaches either configured bound.

**History:**
- `rg-fixedstd010` held std at 0.1 with zero entropy. Exploration RMS stayed at
  0.10001, while its policy-mean tail was only 7.3% below reference and its slope
  only 42.8% lower, short of the 20%-tail or 50%-slope gate. Tracking was +0.67%
  (p=0.771), hazard ratio 1.022 and deterministic raw action L2 34.6% below
  reference.

## residual_magnitude

**Current:** weight `-0.1`. A threefold penalty suppressed the deterministic
action but made physical behavior worse, so `-0.3` is pruned.

**Re-measure if:** the action scale, action dimension or objective changes; the
reward operates on raw action and its physical meaning depends on those values.

**History:**
- `rg-mag03` used weight `-0.3`. Its policy-mean slope was 73.7% below reference,
  the only training-side arm to pass, and deterministic raw action L2 was 73.4%
  lower after dividing the logged reward rate by its 3x weight. It then failed
  all behavioral gates: tracking -0.28% (p=0.894), hazard ratio 1.139 against the
  1.10 limit, episode length -2.91 s, and foot-slip cost +90.5% (p=0.000347).

## screening decision

**Current:** no arm was promoted to a 500-iteration run. The screen retained
position scale `0.01`, entropy `0.0005`, learned std, magnitude weight `-0.1`
and rate weight `-0.1`.

| Arm | Mean tail vs ref | Mean slope vs ref | Tracking | Hazard | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `rg-ref01` | reference | reference | -1.17% | 0.975 | production reference |
| `rg-entropy0` | +1.7% | -4.8% | +1.04% | 1.049 | mean-growth gate failed |
| `rg-fixedstd010` | -7.3% | -42.8% | +0.67% | 1.022 | mean-growth gate failed |
| `rg-mag03` | -0.7% | -73.7% | -0.28% | 1.139 | behavioral gates failed |

**Re-measure if:** the optimized objective changes. These results say residual
growth follows the learned deterministic policy rather than explicit entropy
pressure alone; further PPO tuning is deferred in favor of examining the
objective.

The first objective follow-up is closed (recovery-only DCM screen, written up in
`MC_MJLAB_TRAINING_CONFIG_PROPOSAL_V3.md`, removed in `c100f0a`; read it with
`git show c100f0a^:MC_MJLAB_TRAINING_CONFIG_PROPOSAL_V3.md`):
recovery-only DCM shaping reduced deterministic residual magnitude 67.6% but
worsened its own recovery score 8.14%, so it was not promoted.

The saturation follow-up (`MC_MJLAB_TRAINING_CONFIG_PROPOSAL_V4.md`, likewise
removed in `c100f0a`)
found a behaviorally useful model at iteration 340, but deterministic residual
magnitude had already exceeded reference and continued growing until hazard was
1.160 at iteration 499. The wider recovery objective was not adopted.
