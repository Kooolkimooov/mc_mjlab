# PPO settings

Following mjlab's locomotion configs, with the deviations below. All in
`tasks/residual_balance/residual_balance_ppo_cfg.py`.

The single fact that shapes most of these: **every run to 2026-07-31 trained with
the learning rate pinned to rsl_rl's 1e-5 floor from iteration 0 and never left
it** — 20899 iterations of essentially no learning in the longest one. Several
constants here are attempts on that.

## init_std

**Current:** `0.1`. The residual should start near zero: a unit `init_std` would
hand the controller a huge random offset on step one and knock it over.

**History:**
- `0.2` was still too much of one. Measured against the zero-residual baseline
  through `scripts/compare_to_baseline.py`, a 500-iteration policy scored 18.6%
  survival against the baseline's 26.5% and was below it on both tracking terms,
  reaching only 1238 steps against 1330. It spends the whole run climbing back out
  of the hole its own random initialization dug. A residual policy should start
  indistinguishable from the controller it wraps and improve from there, never
  much worse.

## std_range

**Current:** `(0.05, 0.30)`.

The floor is aimed at the learning rate as much as at exploration. The adaptive
schedule halves the rate whenever measured KL exceeds 2x `desired_kl`, and for
Gaussians `KL ~ dmu^2 / (2 sigma^2)` — so a shrinking sigma inflates KL for an
unchanged weight step. Over the 2026-08-12_18-30-10 run `Policy/mean_std` fell
0.0999 -> 0.0440, a 5x inflation on its own, and the rate sat pinned at its 1e-5
floor for 54% of the last 500 iterations. Clamping sigma at roughly half
`init_std` addresses the collapse and the pinning together.

The upper bound is the old worry, not the current one: at 0.005 entropy the std
used to climb 0.2 -> 0.52 unchecked.

## entropy_coef

**Current:** `0.0005` — an order of magnitude below mjlab's locomotion configs
(0.005), because on a *residual* task the exploration noise is itself a
disturbance: it goes through the real PD gains onto the joints the controller is
balancing on.

Reduced, not zeroed. Zero does not remove the noise — `std` stays learnable and
starts at `init_std` either way — it removes the pressure to *grow* it, and the
opposite failure (std collapsing early onto a brittle local optimum) is the more
expensive one to discover late. The `std_range` clamp bounds the inflation from
both ends anyway.

**History:**
- At 0.005 nothing pushed back: `Policy/mean_std` climbed 0.2 -> 0.52 over 20899
  iterations, and 0.2 -> 0.62 over the 14417 before that. That was ruinous only in
  combination with `residual_scale = 0.1`, where it left the dither at 115-140% of
  the hardware torque limit; back at 0.01 the same std costs ~13-16%, which is
  untidy rather than fatal.

## desired_kl

**Current:** `0.02` — the one parameter with authority over the learning rate on
this task.

rsl_rl's adaptive schedule halves the rate whenever the measured KL exceeds 2x
this, clamped to a 1e-5 floor.

**Re-measure if:** the rate still pins. The next move is 0.03, and the suspect
after that is `obs_normalization`: the running normalizer shifts between when
`old_actions_log_prob` is stored at collection and when the update runs, which
inflates *measured* KL with no weight change at all. rsl_rl logs no KL scalar to
confirm that from the outside.

**History:**
- At `0.01` it hit the floor in the first iteration of *every* run so far and
  never left it (8% of iterations above it across 500, peaking at 5e-5). The
  policy still learns, but at a crawl: it was still climbing toward the
  zero-residual baseline from below when the 500-iteration run ended.

## num_steps_per_env

**Current:** `48`, against mjlab's locomotion default of 24.

Collection here is ~99% mc_rtc: at 128 envs the measured split is 2.9 s
collecting against 0.03 s learning, so a longer rollout is nearly free per sample.
24 steps is 0.48 s of horizon against episodes of 10-30 s, which leaves GAE almost
pure bootstrap off a critic that was not converging. 48 halves the number of
updates and doubles the horizon each advantage is estimated over, which is also
the cheapest relief for the KL blowups that floored the learning rate.
