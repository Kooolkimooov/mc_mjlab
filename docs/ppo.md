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
- 2026-08-17 — that diagnosis was right and this was the wrong lever. `init_std`
  never touched the larger of the two terms; see `ZeroInitMLPModel` below.

## ZeroInitMLPModel

**Current:** the actor's `class_name`, a subclass that zeroes the mean rows of the
output layer after construction. Defined in `tasks/zero_init_actor.py`, selected
through rsl_rl's documented `"module.path:Attr"` form, so nothing is patched.

**Why it exists.** rsl_rl never initializes the actor's mean head. `MLPModel`
calls `distribution.init_mlp_weights`, but `GaussianDistribution` — the scalar-std
class this task uses — inherits the base no-op, because it keeps std in a separate
`nn.Parameter` and has no std rows to zero. (The implementation that *does* zero
std rows belongs to `HeteroscedasticGaussianDistribution`, which is not in use
here.) So the output layer keeps `nn.Linear`'s default init.

Measured on this exact actor — 284 obs, hidden `(512, 256, 128)`, ELU, unit-variance
inputs, which is what `obs_normalization = True` delivers:

| untrained mean action | value |
| --- | --- |
| RMS | **0.0944** |
| peak | **0.4748** |
| `init_std` for comparison | 0.100 |

Iteration 0 is therefore not "zero residual plus dither". It is a **deterministic
random state-feedback law** the size of the exploration noise, peaking at 4.7x it —
and unlike dither it does not average away, because it is a fixed function of
exactly the state the stabilizer is reacting to.

**What it buys:** iteration 0 becomes identical to the zero-residual baseline, so
every `compare_to_baseline.py` reading starts from parity rather than from a
deficit the policy must first climb out of.

**Re-measure if:** rsl_rl is upgraded. Assert rather than trust — sample the
untrained actor and require `mean_head_magnitude(...) < 1e-3`.

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

## gamma

**Current:** `0.997`, against mjlab's locomotion default of 0.99.

At `step_dt = 0.02` the credit horizon is `1/(1-gamma)` steps: 0.99 gives 100
steps, **2 s**, against episodes of ~50 s. 0.997 gives 333 steps, **6.7 s**.

**Why it moved.** The 2026-08-14 run produced a residual that improved the ZMP
match while *increasing* topples and tracking ~10% worse per step overall — see
[reward-shaping.md](reward-shaping.md#residual-harm-at-gamma099). A residual can
pull the centre of pressure toward the plan now and destabilise the gait several
seconds later, and at a 2 s horizon the discounting never presents that bill. The
signature fits: `zmp_error` fell monotonically while `fell_over` rose 0.19 -> 0.23
and survival did not improve.

**Why it is affordable now.** A longer horizon leans harder on the critic, and
until the -200 termination penalty landed the critic was not converging
(`Loss/value` 0.7-5.5). It now sits at ~0.02 for a whole run.
`num_steps_per_env` doubled alongside for the same reason — 1 s of rollout cannot
support a 6.7 s horizon without GAE becoming almost pure bootstrap.

**Re-measure if:** episode length changes a lot, or `Loss/value` stops
converging. The horizon should stay well inside the episode but comfortably
longer than the delay between a residual acting and the fall it causes.

**Tested, and it held.** The 2026-08-15 run answered the falsification test at
1150 iterations: `fell_over` went from 24.1% against a 6.2% baseline to **19.6%
against 19.6%** — the delayed-topple failure mode the horizon was blamed for is
gone. The learning rate came off the floor as a side effect (below). The scale
change bundled with it did not survive; see
[residual-authority.md](residual-authority.md#residual_scale).

## num_steps_per_env

**Current:** `96`, against mjlab's locomotion default of 24.

Collection here is ~99% mc_rtc: at 128 envs the measured split is 2.9 s
collecting against 0.03 s learning, so a longer rollout is nearly free per sample.
24 steps is 0.48 s of horizon against episodes of 10-30 s, which leaves GAE almost
pure bootstrap off a critic that was not converging. 48 halved the number of
updates and doubled the horizon each advantage is estimated over, which is also
the cheapest relief for the KL blowups that floored the learning rate. 96 came
with `gamma = 0.997`: 1.9 s of rollout for a 6.7 s horizon.

## Run 2026-08-14_19-14-46_std-floor

Stopped by SIGINT at 3320 of 15000 iterations because the residual turned out to
be worse than doing nothing — see
[reward-shaping.md](reward-shaping.md#residual-harm-at-gamma099). This is the
first run with `std_range`, `desired_kl = 0.02` and `entropy_coef = 0.0005`, and
the first with random initial pose re-enabled.

| iteration | 0 | 500 | 1400 | 2400 | 3100 |
| --- | --- | --- | --- | --- | --- |
| `Loss/value` | 0.0255 | 0.0208 | 0.0197 | 0.0203 | 0.0204 |
| `Policy/mean_std` | 0.0920 | 0.0596 | 0.0513 | 0.0506 | 0.0500 |
| `Train/mean_reward` | 19.9 | 30.7 | **32.7** | 31.8 | 31.3 |
| `Train/mean_episode_length` | 1906 | 2575 | **2649** | 2583 | 2563 |
| `Episode_Metrics/zmp_error` | 0.0649 | 0.0584 | 0.0562 | 0.0543 | 0.0542 |
| `Episode_Reward/residual_magnitude` | -0.0174 | -0.0305 | -0.0366 | -0.0434 | -0.0450 |
| `Episode_Termination/fell_over` | 0.173 | 0.191 | 0.198 | 0.229 | 0.230 |

**What worked.** `Loss/value` sits at ~0.02 for the whole run. The -200
termination penalty fixed the critic — it was 0.7-5.5 in every run before it.

**What half-worked.** The learning rate does now leave the 1e-5 floor (max 7.6e-5
over the run) where previously it never did, so `desired_kl = 0.02` bought
something. But it is floored 44% of iterations overall and **55% of the last
200**, median exactly 1e-5. `Policy/mean_std` fell 0.1 -> 0.05 and pinned on the
`std_range` clamp by ~iteration 1400; the clamp stops sigma falling further, it
does not undo the KL inflation that already happened.

**What did not work.** Reward and episode length peaked around iteration 1400 and
drifted *down*. `zmp_error` kept improving but was plateauing (0.0562 -> 0.0542
over 1700 iterations). `residual_magnitude` grew monotonically throughout: the
policy learned to intervene **more**, +48% between iterations 500 and 3100, while
`fell_over` rose from 0.19 to 0.23.

**Read the first column with suspicion.** At iteration 0-200 few episodes have
finished, so length and termination shares are biased toward short episodes — the
1/duration trap in [evaluation.md](evaluation.md). Only iteration 500 onward is
trustworthy here.

**Conclusion: 2000 iterations past ~1400 bought nothing.** More iterations at
these settings is not the missing ingredient; see `gamma` for the hypothesis that
replaced it.

## Run 2026-08-15_07-30-27_scale03-gamma997

Killed at 1150 of 4000 iterations (by an agent, not by a stopping rule — but the
stopping rule had already fired). Tested `gamma = 0.997` +
`num_steps_per_env = 96` + `residual_scale = 0.03` as one bundle. **The horizon
half worked and was kept; the scale half was refuted and reverted.**

Against the std-floor run at the same iteration (mean over 1110-1150):

| | std-floor (γ.99, scale .01) | scale03-γ997 |
| --- | --- | --- |
| `Loss/learning_rate` floored, last 200 | 51% | **16%** |
| `Loss/learning_rate` median | 1.0e-05 | **3.0e-05** |
| `Policy/mean_std` | 0.0525 (on the 0.05 clamp) | **0.0682** |
| `Loss/value` | 0.020 | 0.078 |
| `Episode_Metrics/zmp_error` | 0.0550 | 0.0886 |
| per-step `Train/mean_reward` | 0.01224 | 0.00693 |
| `Episode_Termination/fell_over` | 0.272 | 0.261 |

`Episode_Metrics/zmp_grounded` is 1.0000 in both, so the error comparison needs no
denominator correction.

**The learning rate unpinned without touching `desired_kl`.** That is the result
worth keeping: `std_range` and `desired_kl = 0.02` half-fixed the pinning, and the
longer horizon finished the job. `mean_std` also came off the 0.05 clamp. The
4x rise in `Loss/value` is expected — returns are ~3.3x larger at γ.997.

**The training-time metrics look worse and are not the verdict.** Exploration
dither was 4x larger in rad (0.03 x 0.068 against 0.01 x 0.0525), which alone
would degrade `zmp_error` under sampling. The deterministic comparison against the
zero-residual baseline is the honest read, and it also came out worse — that is
what condemned the scale, not this table. See
[residual-authority.md](residual-authority.md#residual_scale) for it.
