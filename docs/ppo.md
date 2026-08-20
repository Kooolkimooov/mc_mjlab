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

## num_learning_epochs

**Current:** `2` with `num_mini_batches = 2`, against mjlab's 5 and 4.

**Their product is the number of learning-rate adaptation events per iteration.**
rsl_rl's adaptive schedule runs *inside* the minibatch loop (`ppo.py:234-249`,
within the flat `mini_batch_generator`; `ppo.py:321` confirms
`num_updates = num_learning_epochs * num_mini_batches`), not once per iteration.
Each event can multiply the rate by 1.5 either way.

| epochs x minibatches | events | worst-case collapse in one iteration |
| --- | --- | --- |
| 5 x 4 | 20 | 1.5^20 = **3325x** |
| 2 x 2 | 4 | 1.5^4 = **5x** |

Falling 1e-3 -> the 1e-5 floor needs `log(100)/log(1.5) = 11.4` consecutive
divisions — reachable inside a single iteration at 20 events, impossible at 4.
This is the mechanism behind `desired_kl`'s "it hit the floor in the first
iteration of every run so far".

The schedule is also biased downward within an iteration: KL is measured against
the collection-time policy, so later minibatches — and every minibatch after the
first epoch — have accumulated more drift and read higher. Down-steps cluster,
while recovery needs eleven consecutive *low* readings that the later epochs
structurally cannot supply.

**It is free in wall clock.** Learning is 0.039 s against 6.94 s of collection, so
this task is ~99% collection-bound; fewer gradient steps costs no time. The
minibatch grows to 6144 samples, which is the more stable KL estimate anyway.

**It is NOT free in stability, and this doc said otherwise until 2026-08-18.** The
event count cuts both ways: 4 events cap the schedule's *recovery* at 1.5^4 = 5x
per iteration, where 20 events could pull the rate down 3325x inside one. Fewer
events is better against the chronic pinning and **worse against an acute
blow-up.** `Run 2026-08-17_15-38-02_zeroinit-4ev` diverged at iteration ~2945 and
never came back: `Loss/value` went 0.047 -> 15.9 between iterations 2940 and 2950,
by which point the rate had reached the floor several iterations too late.

That divergence had a second cause that mattered more, and it is fixed rather than
traded off — see `action_l2` in
[reward-shaping.md](reward-shaping.md#action_l2). Keep 2 x 2, but do not read the
event count as a pure win.

**Re-measure `desired_kl` only after this**, not alongside it — with the event
count down, 0.02 may already be enough.

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

rsl_rl's adaptive schedule divides the rate by 1.5 whenever the measured KL
exceeds 2x this, clamped to a 1e-5 floor.

**Re-measure only after `num_learning_epochs`**, which is the multiplier sitting
in front of this one and is the larger lever. The suspect after both is
`obs_normalization`: the running normalizer shifts between when
`old_actions_log_prob` is stored at collection and when the update runs, which
inflates *measured* KL with no weight change at all. rsl_rl logs no KL scalar to
confirm that from the outside. That one is real but bounded — `EmpiricalNormalization`
moves its mean by ~`batch / count`, so by iteration 218 (count ~2.7M against a
12288-sample batch) it is a 1/n effect. It explains "floored from iteration 0",
not "floored 55% of the last 200".

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
`num_steps_per_env` first doubled alongside for the same reason — 1 s of rollout
cannot support a 6.7 s horizon without GAE becoming almost pure bootstrap. The
current rollout and lambda close the remaining mismatch below.

**Re-measure if:** episode length changes a lot, or `Loss/value` stops
converging. The horizon should stay well inside the episode but comfortably
longer than the delay between a residual acting and the fall it causes.

**Tested, and it held.** The 2026-08-15 run answered the falsification test at
1150 iterations: `fell_over` went from 24.1% against a 6.2% baseline to **19.6%
against 19.6%** — the delayed-topple failure mode the horizon was blamed for is
gone. The learning rate came off the floor as a side effect (below). The scale
change bundled with it did not survive; see
[residual-authority.md](residual-authority.md#residual_scale).

## lam

**Current:** `0.99`, against the previous `0.95`.

`gamma` alone is not PPO's multi-step credit trace. GAE weights fall by
`gamma * lambda` on each step. The old pair gave `0.997 * 0.95 = 0.94715`: a
mean trace length of only 19 steps (0.38 s), with 95% of its mass inside 1.10 s.
Calling that configuration a 6.7 s credit assignment was therefore wrong.

At `lambda = 0.99`, the factor is 0.98703. Its mean trace length is 77 steps
(1.54 s), and 95% of the mass lies inside 229 steps (4.58 s). The critic still
bootstraps the tail, but a delayed topple now contributes materially to the
advantage that updates the action which preceded it.

**Re-measure if:** advantage variance or `Loss/value` rises sharply. Lambda trades
bias for variance; the deterministic baseline comparison, not training return,
decides whether the longer trace helped.

## num_steps_per_env

**Current:** `256` (5.12 s), against mjlab's locomotion default of 24.

Collection here is ~99% mc_rtc: at 128 envs the measured split is 2.9 s
collecting against 0.03 s learning, so a longer rollout is nearly free per sample.
24 steps is 0.48 s of horizon against episodes of 10-30 s, which leaves GAE almost
pure bootstrap off a critic that was not converging. 48 halved the number of
updates and doubled the horizon each advantage is estimated over, which is also
the cheapest relief for the KL blowups that floored the learning rate. 96 came
with `gamma = 0.997`: 1.9 s of rollout for a 6.7 s discount horizon, but still
covered less than half the 4.58 s GAE mass window after fixing `lambda`. A
256-step rollout covers that window and leaves 0.54 s for its tail.

`save_interval` moved from 50 to 20 at the same time. Checkpoints therefore stay
approximately equally dense in experience: every 5120 policy steps per env now,
against 4800 before. Iteration numbers across the two configurations are not
sample-count comparable.

## Training diagnostics

**Current:** four scalars under `Diagnostics/`, from
`residual_balance_diagnostics.ppo_diagnostics`. rsl_rl logs three losses and the
learning rate and nothing else, and `learn()` exposes no hook, so the runner
wraps the one `Logger.log` call each iteration makes and writes them there.

| scalar | what it answers |
| --- | --- |
| `approx_kl` | how far the policy moved over the whole rollout |
| `clip_fraction` | share of samples whose likelihood ratio left the `clip_param` band |
| `explained_variance` | `1 - Var(returns - values) / Var(returns)`; 0 is a mean predictor |
| `action_saturation` | share of action components at `RAW_CLIP` |

All four are read off the rollout *after* `PPO.update()` returns, which is safe
because `RolloutStorage.clear()` resets the write cursor and nothing else: the
buffers stay intact until the next `act()` overwrites them.

**`approx_kl` is not the KL the schedule reacts to.** The adaptive schedule
compares a *per-minibatch* KL against `desired_kl` before each gradient step and
can move the rate four times per iteration; this scalar is the end-of-iteration
divergence over every sample. They answer different questions — "was the step
size safe" against "how far did we go" — and the second was the one no run had.

**Why they were added:** the learning rate is the control that has decided every
run here (see the note at the top of this file), and its input was never
recorded. `explained_variance` is the other half: `Loss/value` is in task units
and cannot say whether the critic is fitting anything.

**Re-measure if:** nothing — these are instruments, not constants. But note the
cost, one extra forward pass over the rollout per iteration, against ~99% of
iteration time spent in mc_rtc collection.

## Training budget

**Current:** `POLICY_STEPS_PER_ENV = 128_000`. `max_iterations` is derived from it
and `num_steps_per_env` rather than typed: `128_000 / 256 = 500`, the same 500 as
before.

Iteration counts are not comparable once the rollout length moves, and this task
has moved it three times:

| configuration | calculation | steps/env |
| --- | --- | --- |
| previous | 96 x 500 | 48,000 |
| current | 256 x 500 | 128,000 |
| matched short run | 256 x 188 | 48,128 |

So a screening comparison against the earlier era wants ~48,000 steps per env,
and only a promising setting is worth the full 128,000.

Every run records what it was given: `training_budget.json` beside the run's
`base_controller_config/` snapshot, and `infos["training_budget"]` in every
checkpoint. Both carry `policy_steps_per_env` and `total_transitions`, since the
second also moves with `num_envs`.

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

## Run 2026-08-17_15-38-02_zeroinit-4ev

3000 iterations in 7 h 13 m. **The first run to beat the zero-residual controller,
and the first to diverge.** Tested `ZeroInitMLPModel` and `2 x 2` learning epochs x
minibatches; otherwise `gamma = 0.997`, `residual_scale = 0.01`, and the *old*
plan-matching reward and 284-dim observations, so its comparisons line up directly
with `scale01-gamma997`.

**The optimiser was healthy for the first time.** Over 2410 iterations the rate was
floored **0.0%** of the time at a median of 4.4e-04, against 19% / 3.0e-05 for
`scale01` and 44% / 1.0e-05 for `std-floor`. `Policy/mean_std` sat at 0.11-0.12 and
never approached the 0.05 clamp; `Loss/value` held ~0.10.

**Zero-init showed too.** `Episode_Reward/residual_magnitude` began at -0.0052 at
iteration 40, rather than jumping straight to a random feedback law.

**It peaked early and decayed.** Best smoothed `Episode_Metrics/zmp_error` was
0.06072 at **iteration 1320**; the final 1600 iterations moved it nowhere
(0.0625 at 2999) while reward drifted 26 -> 22.5 -> 26.

Against the baseline, deterministic, per-step:

| | `model_1000` (n=136/arm) | `model_2900` (n=224/arm) |
| --- | --- | --- |
| `zmp_tracking` | **+3.7%** (p = 0.003) | **-5.5%** (p = 1.3e-09) |
| `com_velocity_tracking` | -5.0% | -6.3% |
| survival | **+11.5pp** (p = 0.029) | +5.8pp (p = 0.14) |
| `collapsed` | -24.6pp | -12.8pp |
| `corr(residual, episode length)` | +0.12 | +0.27 |

`model_1000` is positive in all four length bands; `model_2900` is negative in all
four. The correlation staying positive is the horizon fix holding — it was **-0.60**
at `gamma = 0.99`, where intervening more went with dying sooner.

**Then it diverged at iteration ~2945** and never recovered, ending at reward -291
with a NaN at 2993. Two causes, one traded off and one fixed: the 4-event schedule
could not throttle fast enough (`num_learning_epochs` above), and `action_l2` was
charging unboundedly for raw actions past the clip
([reward-shaping.md](reward-shaping.md#action_l2)).

**Conclusion: cap this configuration near 1500 iterations.** Its useful work is
done by ~1300, everything after is decay, and the last 50 iterations were actively
destructive. Only `save_interval = 50` preserved the checkpoint worth having.
