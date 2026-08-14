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

**Falsifiable.** If `residual_magnitude` still climbs and the per-step tracking
deficit is still ~10% at iteration 1000, the horizon was not the constraint.

## num_steps_per_env

**Current:** `48`, against mjlab's locomotion default of 24.

Collection here is ~99% mc_rtc: at 128 envs the measured split is 2.9 s
collecting against 0.03 s learning, so a longer rollout is nearly free per sample.
24 steps is 0.48 s of horizon against episodes of 10-30 s, which leaves GAE almost
pure bootstrap off a critic that was not converging. 48 halves the number of
updates and doubles the horizon each advantage is estimated over, which is also
the cheapest relief for the KL blowups that floored the learning rate.
[38;2;0;136;0;03m## Run 2026-08-14_19-14-46_std-floor[39;00m

Stopped[38;2;187;187;187m [39m[38;2;170;34;255;01mby[39;00m[38;2;187;187;187m [39mSIGINT[38;2;187;187;187m [39m[38;2;170;34;255;01mat[39;00m[38;2;187;187;187m [39m[38;2;102;102;102m3320[39m[38;2;187;187;187m [39m[38;2;170;34;255;01mof[39;00m[38;2;187;187;187m [39m[38;2;102;102;102m15000[39m[38;2;187;187;187m [39miterations[38;2;187;187;187m [39mbecause[38;2;187;187;187m [39mthe[38;2;187;187;187m [39mresidual[38;2;187;187;187m [39mturned[38;2;187;187;187m [39m[38;2;170;34;255;01mout[39;00m[38;2;187;187;187m [39m[38;2;170;34;255;01mto[39;00m
be[38;2;187;187;187m [39mworse[38;2;187;187;187m [39m[38;2;170;34;255;01mthan[39;00m[38;2;187;187;187m [39mdoing[38;2;187;187;187m [39mnothing[38;2;187;187;187m [39m—[38;2;187;187;187m [39msee
[reward[38;2;102;102;102m-[39mshaping.md](reward[38;2;102;102;102m-[39mshaping.md[38;2;0;136;0;03m#residual-harm-at-gamma099). This is the[39;00m
[38;2;170;34;255;01mfirst[39;00m[38;2;187;187;187m [39mrun[38;2;187;187;187m [39m[38;2;170;34;255;01mwith[39;00m[38;2;187;187;187m [39m`std_range`,[38;2;187;187;187m [39m`desired_kl = 0.02`[38;2;187;187;187m [39m[38;2;170;34;255;01mand[39;00m[38;2;187;187;187m [39m`entropy_coef = 0.0005`,[38;2;187;187;187m [39m[38;2;170;34;255;01mand[39;00m
the[38;2;187;187;187m [39m[38;2;170;34;255;01mfirst[39;00m[38;2;187;187;187m [39m[38;2;170;34;255;01mwith[39;00m[38;2;187;187;187m [39m[38;2;170;34;255;01mrandom[39;00m[38;2;187;187;187m [39m[38;2;170;34;255;01minitial[39;00m[38;2;187;187;187m [39mpose[38;2;187;187;187m [39mre[38;2;102;102;102m-[39menabled.

[38;2;102;102;102m|[39m[38;2;187;187;187m [39miteration[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m500[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m1400[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m2400[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m3100[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m
[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m---[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m---[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m---[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m---[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m---[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m---[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m
[38;2;102;102;102m|[39m[38;2;187;187;187m [39m`Loss/value`[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.0255[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.0208[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.0197[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.0203[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.0204[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m
[38;2;102;102;102m|[39m[38;2;187;187;187m [39m`Policy/mean_std`[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.0920[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.0596[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.0513[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.0506[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.0500[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m
[38;2;102;102;102m|[39m[38;2;187;187;187m [39m`Train/mean_reward`[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m19.9[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m30.7[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m**[39m[38;2;102;102;102m32.7[39m[38;2;102;102;102m**[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m31.8[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m31.3[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m
[38;2;102;102;102m|[39m[38;2;187;187;187m [39m`Train/mean_episode_length`[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m1906[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m2575[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m**[39m[38;2;102;102;102m2649[39m[38;2;102;102;102m**[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m2583[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m2563[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m
[38;2;102;102;102m|[39m[38;2;187;187;187m [39m`Episode_Metrics/zmp_error`[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.0649[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.0584[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.0562[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.0543[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.0542[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m
[38;2;102;102;102m|[39m[38;2;187;187;187m [39m`Episode_Reward/residual_magnitude`[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m-[39m[38;2;102;102;102m0.0174[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m-[39m[38;2;102;102;102m0.0305[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m-[39m[38;2;102;102;102m0.0366[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m-[39m[38;2;102;102;102m0.0434[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m-[39m[38;2;102;102;102m0.0450[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m
[38;2;102;102;102m|[39m[38;2;187;187;187m [39m`Episode_Termination/fell_over`[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.173[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.191[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.198[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.229[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.230[39m[38;2;187;187;187m [39m[38;2;102;102;102m|[39m

[38;2;102;102;102m**[39mWhat[38;2;187;187;187m [39mworked.[38;2;102;102;102m**[39m[38;2;187;187;187m [39m`Loss/value`[38;2;187;187;187m [39msits[38;2;187;187;187m [39m[38;2;170;34;255;01mat[39;00m[38;2;187;187;187m [39m[38;2;102;102;102m~[39m[38;2;102;102;102m0.02[39m[38;2;187;187;187m [39m[38;2;170;34;255;01mfor[39;00m[38;2;187;187;187m [39mthe[38;2;187;187;187m [39mwhole[38;2;187;187;187m [39mrun.[38;2;187;187;187m [39mThe[38;2;187;187;187m [39m[38;2;102;102;102m-[39m[38;2;102;102;102m200[39m
termination[38;2;187;187;187m [39mpenalty[38;2;187;187;187m [39m[38;2;0;187;0;01mfixed[39;00m[38;2;187;187;187m [39mthe[38;2;187;187;187m [39mcritic[38;2;187;187;187m [39m—[38;2;187;187;187m [39mit[38;2;187;187;187m [39mwas[38;2;187;187;187m [39m[38;2;102;102;102m0.7[39m[38;2;102;102;102m-[39m[38;2;102;102;102m5.5[39m[38;2;187;187;187m [39m[38;2;170;34;255;01min[39;00m[38;2;187;187;187m [39m[38;2;170;34;255;01mevery[39;00m[38;2;187;187;187m [39mrun[38;2;187;187;187m [39m[38;2;170;34;255;01mbefore[39;00m[38;2;187;187;187m [39mit.

[38;2;102;102;102m**[39mWhat[38;2;187;187;187m [39mhalf[38;2;102;102;102m-[39mworked.[38;2;102;102;102m**[39m[38;2;187;187;187m [39mThe[38;2;187;187;187m [39mlearning[38;2;187;187;187m [39mrate[38;2;187;187;187m [39mdoes[38;2;187;187;187m [39mnow[38;2;187;187;187m [39m[38;2;170;34;255;01mleave[39;00m[38;2;187;187;187m [39mthe[38;2;187;187;187m [39m[38;2;102;102;102m1e-5[39m[38;2;187;187;187m [39m[38;2;0;160;0mfloor[39m[38;2;187;187;187m [39m(max[38;2;187;187;187m [39m[38;2;102;102;102m7.6e-5[39m
[38;2;170;34;255;01mover[39;00m[38;2;187;187;187m [39mthe[38;2;187;187;187m [39mrun)[38;2;187;187;187m [39m[38;2;170;34;255;01mwhere[39;00m[38;2;187;187;187m [39mpreviously[38;2;187;187;187m [39mit[38;2;187;187;187m [39m[38;2;170;34;255;01mnever[39;00m[38;2;187;187;187m [39mdid,[38;2;187;187;187m [39mso[38;2;187;187;187m [39m`desired_kl = 0.02`[38;2;187;187;187m [39mbought
something.[38;2;187;187;187m [39mBut[38;2;187;187;187m [39mit[38;2;187;187;187m [39m[38;2;170;34;255;01mis[39;00m[38;2;187;187;187m [39mfloored[38;2;187;187;187m [39m[38;2;102;102;102m44[39m[38;2;102;102;102m%[39m[38;2;187;187;187m [39m[38;2;170;34;255;01mof[39;00m[38;2;187;187;187m [39miterations[38;2;187;187;187m [39moverall[38;2;187;187;187m [39m[38;2;170;34;255;01mand[39;00m[38;2;187;187;187m [39m[38;2;102;102;102m**[39m[38;2;102;102;102m55[39m[38;2;102;102;102m%[39m[38;2;187;187;187m [39m[38;2;170;34;255;01mof[39;00m[38;2;187;187;187m [39mthe[38;2;187;187;187m [39m[38;2;170;34;255;01mlast[39;00m
[38;2;102;102;102m200[39m[38;2;102;102;102m**[39m,[38;2;187;187;187m [39mmedian[38;2;187;187;187m [39mexactly[38;2;187;187;187m [39m[38;2;102;102;102m1e-5[39m.[38;2;187;187;187m [39m`Policy/mean_std`[38;2;187;187;187m [39mfell[38;2;187;187;187m [39m[38;2;102;102;102m0.1[39m[38;2;187;187;187m [39m[38;2;102;102;102m->[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.05[39m[38;2;187;187;187m [39m[38;2;170;34;255;01mand[39;00m[38;2;187;187;187m [39mpinned[38;2;187;187;187m [39m[38;2;170;34;255;01mon[39;00m[38;2;187;187;187m [39mthe
`std_range`[38;2;187;187;187m [39mclamp[38;2;187;187;187m [39m[38;2;170;34;255;01mby[39;00m[38;2;187;187;187m [39m[38;2;102;102;102m~[39miteration[38;2;187;187;187m [39m[38;2;102;102;102m1400[39m;[38;2;187;187;187m [39mthe[38;2;187;187;187m [39mclamp[38;2;187;187;187m [39mstops[38;2;187;187;187m [39msigma[38;2;187;187;187m [39mfalling[38;2;187;187;187m [39mfurther,[38;2;187;187;187m [39mit
does[38;2;187;187;187m [39m[38;2;170;34;255;01mnot[39;00m[38;2;187;187;187m [39m[38;2;170;34;255;01mundo[39;00m[38;2;187;187;187m [39mthe[38;2;187;187;187m [39mKL[38;2;187;187;187m [39minflation[38;2;187;187;187m [39mthat[38;2;187;187;187m [39malready[38;2;187;187;187m [39mhappened.

[38;2;102;102;102m**[39mWhat[38;2;187;187;187m [39mdid[38;2;187;187;187m [39m[38;2;170;34;255;01mnot[39;00m[38;2;187;187;187m [39m[38;2;170;34;255;01mwork[39;00m.[38;2;102;102;102m**[39m[38;2;187;187;187m [39mReward[38;2;187;187;187m [39m[38;2;170;34;255;01mand[39;00m[38;2;187;187;187m [39mepisode[38;2;187;187;187m [39mlength[38;2;187;187;187m [39mpeaked[38;2;187;187;187m [39maround[38;2;187;187;187m [39miteration[38;2;187;187;187m [39m[38;2;102;102;102m1400[39m[38;2;187;187;187m [39m[38;2;170;34;255;01mand[39;00m
drifted[38;2;187;187;187m [39m[38;2;102;102;102m*[39mdown[38;2;102;102;102m*[39m.[38;2;187;187;187m [39m`zmp_error`[38;2;187;187;187m [39mkept[38;2;187;187;187m [39mimproving[38;2;187;187;187m [39mbut[38;2;187;187;187m [39mwas[38;2;187;187;187m [39mplateauing[38;2;187;187;187m [39m([38;2;102;102;102m0.0562[39m[38;2;187;187;187m [39m[38;2;102;102;102m->[39m[38;2;187;187;187m [39m[38;2;102;102;102m0.0542[39m
[38;2;170;34;255;01mover[39;00m[38;2;187;187;187m [39m[38;2;102;102;102m1700[39m[38;2;187;187;187m [39miterations).[38;2;187;187;187m [39m`residual_magnitude`[38;2;187;187;187m [39mgrew[38;2;187;187;187m [39mmonotonically[38;2;187;187;187m [39mthroughout[38;2;102;102;102m:[39m[38;2;187;187;187m [39mthe
policy[38;2;187;187;187m [39mlearned[38;2;187;187;187m [39m[38;2;170;34;255;01mto[39;00m[38;2;187;187;187m [39mintervene[38;2;187;187;187m [39m[38;2;102;102;102m**[39mmore[38;2;102;102;102m**[39m,[38;2;187;187;187m [39m[38;2;102;102;102m+[39m[38;2;102;102;102m48[39m[38;2;102;102;102m%[39m[38;2;187;187;187m [39m[38;2;170;34;255;01mbetween[39;00m[38;2;187;187;187m [39miterations[38;2;187;187;187m [39m[38;2;102;102;102m500[39m[38;2;187;187;187m [39m[38;2;170;34;255;01mand[39;00m[38;2;187;187;187m [39m[38;2;102;102;102m3100[39m,[38;2;187;187;187m [39m[38;2;170;34;255;01mwhile[39;00m
`fell_over`[38;2;187;187;187m [39mrose[38;2;187;187;187m [39m[38;2;170;34;255;01mfrom[39;00m[38;2;187;187;187m [39m[38;2;102;102;102m0.19[39m[38;2;187;187;187m [39m[38;2;170;34;255;01mto[39;00m[38;2;187;187;187m [39m[38;2;102;102;102m0.23[39m.

[38;2;102;102;102m**[39m[38;2;170;34;255;01mRead[39;00m[38;2;187;187;187m [39mthe[38;2;187;187;187m [39m[38;2;170;34;255;01mfirst[39;00m[38;2;187;187;187m [39m[38;2;170;34;255;01mcolumn[39;00m[38;2;187;187;187m [39m[38;2;170;34;255;01mwith[39;00m[38;2;187;187;187m [39msuspicion.[38;2;102;102;102m**[39m[38;2;187;187;187m [39m[38;2;170;34;255;01mAt[39;00m[38;2;187;187;187m [39miteration[38;2;187;187;187m [39m[38;2;102;102;102m0[39m[38;2;102;102;102m-[39m[38;2;102;102;102m200[39m[38;2;187;187;187m [39mfew[38;2;187;187;187m [39mepisodes[38;2;187;187;187m [39mhave
finished,[38;2;187;187;187m [39mso[38;2;187;187;187m [39mlength[38;2;187;187;187m [39m[38;2;170;34;255;01mand[39;00m[38;2;187;187;187m [39mtermination[38;2;187;187;187m [39mshares[38;2;187;187;187m [39mare[38;2;187;187;187m [39mbiased[38;2;187;187;187m [39mtoward[38;2;187;187;187m [39mshort[38;2;187;187;187m [39mepisodes[38;2;187;187;187m [39m—[38;2;187;187;187m [39mthe
[38;2;102;102;102m1[39m[38;2;102;102;102m/[39mduration[38;2;187;187;187m [39mtrap[38;2;187;187;187m [39m[38;2;170;34;255;01min[39;00m[38;2;187;187;187m [39m[evaluation.md](evaluation.md).[38;2;187;187;187m [39m[38;2;170;34;255;01mOnly[39;00m[38;2;187;187;187m [39miteration[38;2;187;187;187m [39m[38;2;102;102;102m500[39m[38;2;187;187;187m [39monward[38;2;187;187;187m [39m[38;2;170;34;255;01mis[39;00m
trustworthy[38;2;187;187;187m [39mhere.

[38;2;102;102;102m**[39mConclusion[38;2;102;102;102m:[39m[38;2;187;187;187m [39m[38;2;102;102;102m2000[39m[38;2;187;187;187m [39miterations[38;2;187;187;187m [39mpast[38;2;187;187;187m [39m[38;2;102;102;102m~[39m[38;2;102;102;102m1400[39m[38;2;187;187;187m [39mbought[38;2;187;187;187m [39mnothing.[38;2;102;102;102m**[39m[38;2;187;187;187m [39mMore[38;2;187;187;187m [39miterations[38;2;187;187;187m [39m[38;2;170;34;255;01mat[39;00m
these[38;2;187;187;187m [39m[38;2;170;34;255;01mset[39;00mtings[38;2;187;187;187m [39m[38;2;170;34;255;01mis[39;00m[38;2;187;187;187m [39m[38;2;170;34;255;01mnot[39;00m[38;2;187;187;187m [39mthe[38;2;187;187;187m [39mmissing[38;2;187;187;187m [39mingredient;[38;2;187;187;187m [39msee[38;2;187;187;187m [39m`gamma`[38;2;187;187;187m [39m[38;2;170;34;255;01mfor[39;00m[38;2;187;187;187m [39mthe[38;2;187;187;187m [39mhypothesis[38;2;187;187;187m [39mthat
replaced[38;2;187;187;187m [39mit.
