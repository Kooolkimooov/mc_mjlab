# Difficulty

How hard the residual balance task is: how hard the robot is shoved, when the
shoving starts, and how long an episode runs. All three interact — changing one
without the others moves the baseline survival rate, which is the number the
task is actually calibrated against.

Constants live in `src/mc_mjlab/tasks/residual_balance/residual_balance_env_cfg.py`.

## finite_impulse_curriculum

**Current:** training uses a finite external wrench at the root, not a velocity
teleport. Each disturbance samples a uniform body-frame planar direction, an
`0.08-0.20 s` duration, and a `0-0.25 m` vertical moment arm. Force is computed
from the environment's randomized total robot mass so its time integral equals
the requested equivalent delta velocity. The first event lands after a
`10-12 s` warm-up, then repeats every `5-7 s`.

The equivalent-delta-velocity curriculum is `[0.10, 0.25] m/s` initially,
`[0.10, 0.40] m/s` after 48,000 environment steps, and `[0.10, 0.50] m/s` after
96,000. `disturbance="velocity"` retains the old velocity kick only for detector
calibration and compatibility evaluation.

**Re-measure if:** mass, policy step time, force application semantics, root body,
or controller gait changes.

**History:**
- 2026-08-24 — live verification over 16 environments x 1000 steps delivered
  32 impulses. Equivalent-delta-velocity error was at most `7.5e-08 m/s`;
  duration and `0.25 m` moment-arm assertions passed. The accepted detector
  reached maximum authority `0.741`, mean duty `0.941%`, and exactly zero
  inactive residual.

## stratified_finite_impulse_curriculum

**Current:** the `Position-Ankle-Matched-Impulse` task replaces the global-step
impulse curriculum with a stationary mixture drawn once per environment reset.
It exists because training and qualification did not share a support.

**The mismatch.** `finite_impulse_curriculum` holds `[0.10, 0.25] m/s` for the
first 48,000 policy steps. The 13-arm screen ran 188 iterations, which is 48,128
policy steps per environment, so every arm trained essentially entirely inside
`[0.10, 0.25]`. `PairedDisturbances` then scores `current_kick` and
`finite_impulse` at `0.40 m/s` and `robust` at `[0.50, 0.60] m/s` — 1.6x to 2.4x
above the training support. `promotion()` requires
`sum(policy hazard) / sum(baseline hazard) <= 0.90` across all four scenarios,
and the two measured baseline hazards are `9.375%` for finite impulse against
`78.125%` for robust, so that sum is dominated by the band the policy never saw.

The stage-0 achievement run measured exactly that split: finite-impulse hazard
improved from `9.375%` to `6.25%` while robust hazard worsened from `78.125%` to
`90.625%`, for an overall ratio of `1.107`. The policy improved what it trained
on and degraded what it did not. Advancing past stage 0 required passing a gate
containing the robust band, so the curriculum could not reach the difficulty its
own gate scored.

**The mixture.** Bands are `QUALIFICATION_MATCHED_BANDS`, weights are
`QUALIFICATION_MATCHED_WEIGHTS`:

| Cohort | Equivalent delta velocity | Share | Covers |
| --- | --- | ---: | --- |
| standing | none | 20% | `nominal` gait, duty, and foot-slip gates |
| band 0 | `[0.10, 0.25] m/s` | 25% | the historical training support |
| band 1 | `[0.25, 0.40] m/s` | 25% | `current_kick` and `finite_impulse` at `0.40` |
| band 2 | `[0.40, 0.60] m/s` | 30% | `robust` at `[0.50, 0.60]` |

Standing keeps 20% because `nominal` carries its own promotion gates: authority
duty at most `5%`, and `zmp_error` and `foot_slip` upper-CI regression under
`5%`. Band 2 takes the largest share because it dominates the hazard denominator,
but not more: a fallen robot has no recovery window to track, so flooding the
rollout with the hardest band would starve the very `recovery_dcm_error` measure
that the promotion gate ranks on. 45% of episodes stay at or below the
historical distribution.

**Why stationary rather than a curriculum.** The 500-iteration budget run took
its best 60-iteration ZMP and recovery windows at iterations 175 and 178,
immediately before the stage boundary at 188, then degraded and produced NaNs at
488. Non-stationary difficulty is implicated in that decay, and any schedule that
reaches the robust band at 96,000 steps never arrives inside a screen budget.
Sampling the mixture at reset rather than per step keeps an episode's difficulty
fixed while it runs, so a recovery window is not scored across a difficulty
change.

**How to read it.** The `impulse_speed` metric reports the equivalent delta
velocity of each environment's last impulse. Against the ankle screen arm it must
rise; if it does not, the mixture is not reaching the sampler. The hypothesis is
falsified if robust-scenario hazard does not improve relative to the ankle
control, and it is only supported by paired qualification, never by the training
curves.

**Re-measure if:** `PairedDisturbances` magnitudes, the promotion hazard gate,
episode length, reset rate, or the baseline failure boundary changes.

**History:**
- 2026-08-27 — created after the stage-0 achievement run showed in-distribution
  improvement and out-of-distribution degradation on the same checkpoint.

## curriculum_diagnostics

**Current:** Two additive ankle-authority tasks isolate the difficulty change
that coincides with the long run's early optimum. `Curriculum-Frozen` keeps the
impulse range at `[0.10, 0.25] m/s`. `Curriculum-Gradual` holds that range through
48,000 policy steps, then linearly raises its upper bound to `0.40 m/s` at 80,000
and `0.50 m/s` at 112,000. Both keep the torque-margin weight at `-0.05`, so the
diagnostic changes only impulse difficulty.

`scripts/run_curriculum_diagnostics.py` runs two 2-iteration, 8-environment smoke
tests before two 220-iteration, 128-environment diagnostics. Smoke tests use
TensorBoard; diagnostics use W&B by default. Every run enables mjlab's NaN guard,
saves locally every 20 iterations, and records resumable state under
`logs/curriculum_diagnostics/`.

The 220-iteration budget reaches 56,320 policy steps. The gradual arm therefore
ends with a `0.289 m/s` upper bound: enough to cross the old 48,000-step boundary
without introducing the old instantaneous `0.25 -> 0.40 m/s` jump. This is a
mechanism diagnostic, not a promotion run. Compare pre-boundary and late-window
ZMP/recovery metrics, then apply paired deterministic qualification to any
shortlisted checkpoint.

**Re-measure if:** the rollout length, policy-step budget, disturbance stages,
authority set, or torque-margin schedule changes.

**History:**
- 2026-08-26 — prepared the frozen-versus-gradual experiment after the 500-iteration
  budget run reached its best 60-iteration ZMP and recovery windows at iterations
  175 and 178, just before the old stage boundary at iteration 188, then degraded
  and produced NaNs at iteration 488.

## achievement_finite_impulse_curriculum

**Current:** the separately registered
`Position-Ankle-Curriculum-Achievement` task is the main-capable curriculum. Its
stage is independent of `common_step_counter`; only a held-out report accepted
by `AchievementCurriculumBridge` changes it. The physical stages are equivalent
body-frame delta-velocity ranges `[0.10, 0.25]`, `[0.10, 0.40]`, and
`[0.10, 0.50] m/s`. Push direction, interval, duration, point of application,
residual authority, observations, randomization, and the `-0.05` torque-margin
weight stay fixed, so one physical challenge changes.

The reset-time mixtures list standing first, followed by physical stages zero
through two:

| Active stage | Standing | Stage 0 | Stage 1 | Stage 2 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 25% | 75% | 0% | 0% |
| 1 | 15% | 25% | 60% | 0% |
| 2 | 15% | 15% | 20% | 50% |

Existing episodes keep their sampled level when a report arrives; the new
mixture enters as environments reset. This prevents an all-environment step
change and retains both no-push balance and previously qualified recovery.

**Re-measure if:** stage-zero feasibility, the baseline failure boundary,
episode reset rate, push implementation, residual authority, or qualification
variance changes.

**History:**

- 2026-08-26 — tranche 5 added reset-cohort rehearsal and held-out stage state;
  the frozen and gradual tasks remain step-schedule diagnostics.

## AchievementCurriculumBridge

**Current:** the runner watches `<run>/curriculum/qualification.json` at PPO
iteration boundaries. A valid report must describe exactly one checkpoint from
the active run, match the requested stage contract, contain nominal,
current-kick, finite-impulse, and robust scenarios, and use at least two unique
seeds. Two distinct eligible reports qualify the stage. Three distinct
ineligible reports demote one stage, including from a previously mastered final
stage. A stage/checkpoint/seed identity hash prevents the same evidence from
counting twice even if JSON formatting or reported metrics change.

Stage qualification copies the evaluated checkpoint to
`qualified_stage_<stage>_model_<iteration>.pt`. Rollback changes future training
cohorts but deliberately does not rewind the live policy or optimizer; use the
preserved checkpoint for an explicit policy rewind. `state.json`,
`qualification_request.json`, and append-only `events.jsonl` expose the live
protocol. Every subsequent model checkpoint embeds the stage, pass/regression
streaks, processed-evidence hashes, mastery state, last-good path, and one
qualified-checkpoint path per stage. Full resume restores those values exactly;
actor-only evaluation does not change them. A report absent from the restored
checkpoint's processed hashes is replayed after a same-directory resume, so an
iteration-boundary decision is not silently discarded before the next save.

Held-out stage pushes use `0.25`, `0.40`, and `0.50 m/s` for current-kick and
finite-impulse scenarios. Their robust guards use `[0.30, 0.35]`,
`[0.45, 0.50]`, and `[0.55, 0.60] m/s`, respectively, with the existing stage-one
physics randomization. Promotion still uses the unchanged safety, nominal,
recovery, and hazard gates in `qualify_checkpoints.py`.

**Re-measure if:** the promotion gates, paired-sample variance, required seed
count, stage magnitudes, or checkpoint cadence changes.

**History:**

- 2026-08-26 — schema 1 established two-pass advancement, three-regression
  rollback, stage-contract hashing, last-good preservation, and checkpointed
  resume continuity.

## randomization_stage

**Current:** stage zero is the standard task. Archived `Position-Robust1` samples
friction `+-10%`, active PD gains `+-5%`, and motor strength `+-5%`. Actor
observations sample `0-1` policy-step delay, while
actuator commands sample `0-1` mc_rtc controller-period delay (`0-2` physics
substeps). Archived `Position-Robust2` doubles every range and permits two delay
steps. Both require `MC_MJLAB_REGISTER_ARCHIVED_TASKS=1`. Randomization is static
per environment construction; ordinary training does not enable it before the
standard-task promotion gate is met.

Mass/inertia and COM are intentionally excluded. Expanding either family for
per-world randomization and recomputing MuJoCo constants changed HRP5P's walking
dynamics even when every requested perturbation was exactly zero.

**Re-measure if:** mjlab's per-world inertial-field expansion, observation delay,
actuator delay, or effort-limit implementations change.

**History:**
- 2026-08-25 — excluded mass/inertia and COM after component ablation isolated a
  pre-push failure. Nominal and every other component survived 8/8 for 12 s;
  pseudo-inertia, direct mass/inertia scaling, and COM offset each survived only
  1/8. The latter two still failed with their perturbations set exactly to zero,
  locating the incompatibility in field expansion/recomputation rather than the
  requested ranges. After removing those fields, the combined stage-one profile
  and its nominal control each survived 16/16 for 12 s with no worker failure.
- 2026-08-24 — added as explicit task variants so robustness is a gated stage,
  not a silent change to the standard task.
- 2026-08-24 — a live smoke test rejected interpreting controller delay as a
  full 20 ms policy step: the cohort repeatedly reset before its first impulse.
  Controller delay now uses the coupling's actual 2 ms period.
- 2026-08-24 — mjlab's generic PD-gain randomizer was also rejected because it
  scaled the armature-derived construction defaults, silently replacing the
  reference `PDgains_sim.dat` values. Scaling the active gains instead passed a
  stage-one live run over 4 environments x 1000 steps: 8 impulses, maximum
  authority `0.516`, duty `0.722%`, exact inactive zeroing, and impulse error
  `5.2e-08 m/s`.

## map_com_stability

**Current:** `scripts/map_com_stability.py` changes the compiled simulator model
while leaving mc_rtc's robot model nominal. It moves the root torso's inertial
position enough to produce the requested initial whole-robot COM offset, then
runs zero-residual, push-free walking. This bypasses the invalid per-world
inertial-field expansion path.

A seed-42 screen used 4 environments per point and a 12 s horizon. Every point
had zero worker failures. The transition intervals are:

| initial COM offset | last 4/4 survival | partial survival | first 0/4 survival |
| --- | ---: | ---: | ---: |
| backward x | -55 mm | none sampled | -60 mm |
| forward x | +80 mm | +85 mm: 2/4; +90 mm: 1/4 | +95 mm |
| negative y | -45 mm | -50 mm: 3/4; -55 mm: 1/4 | -60 mm |
| positive y | +50 mm | none sampled | +55 mm |
| vertical z | -100 to +100 mm | none | not reached |

The intended `+-5 mm` COM range is therefore well inside this short-horizon
envelope. These are aggregate initial offsets produced through the torso, not
independent per-link errors, and the extreme torso shifts are diagnostic rather
than plausible morphology. Run with a longer horizon and more seeds before
treating a boundary as a controller guarantee.

**Re-measure if:** the robot model, walking controller, installed gait, encoder
bias, horizon, or method used to distribute COM error changes.

**History:**
- 2026-08-25 — added the compiled-model sweep after runtime COM randomization
  falsely made 7/8 nominal-value environments fall. Genuine compiled COM error
  remained stable across at least `+-40 mm` on every axis.

## push_velocity

**Current:** `0.4` — the task's difficulty dial. Both directions ruin training:
too gentle and mc_rtc never falls, so the best residual is no residual; too hard
and the robot falls whatever the residual does, which plateaus the policy at a
fraction of an episode. 0.4 is deliberately past the point where the baseline
copes, so that the residual's contribution shows up as survival rather than
being hidden inside a controller that would have coped anyway.

It lives in the module rather than as a CLI flag because it sits inside an event
term's `velocity_range` dict, which tyro does not flatten.

**Re-measure if:** the sampling changes, or the walk window moves.

The number to watch is the baseline's survival rate, not the push magnitude. The
ceiling is where the robot falls whatever the residual does.

**Verified 2026-08-14 at the current symmetric sampling: 0.4 is right.** Baseline
survival 20.8% [11.7%, 34.3%] over 48 trimmed episodes (74 completed, 24 envs x
8 min), against the ~22% `WALK_WINDOW_S` was sized for. Post-warm-up hazard
0.0212/s against the 0.019/s predicted; mean episode 48.3 s, median 42.5 s.

There had been reason to doubt it: 0.4 was calibrated during the 2026-07-31 runs,
whose working tree had `"x": (push_velocity, push_velocity)` — a degenerate range,
so every push was exactly +0.4 in x *and* +0.4 in y, a constant 0.566 m/s shove in
the same direction every time. Sampled symmetrically the same 0.4 is much gentler
(mean |v| ~ 0.31 m/s, random direction, sometimes ~0). The two effects evidently
cancelled; the measurement above is what settles it.

**Do not read survival off a short training run.** A 40-iteration run gives
1920 steps per env, and no episode can reach the 4500-step cap inside that, so
`time_out` is structurally impossible and only falls complete: the same data
filtered to that window reads 0% survival and a 22 s mean, against the true 20.8%
and 48.3 s. `Episode_Termination/*` and `Mean episode length` in the training log
are biased by 1/duration until runs are long relative to the cap — see
[evaluation.md](evaluation.md#fixed-episodes-per-env-not-everything-that-finished).

**History:**
- 0.1 already lost the zero-residual baseline about half its episodes
  (`fell_over` + `collapsed` vs `time_out`, over 65 s x 16 envs). A *walking*
  robot is far easier to topple than a standing one, which is why that reads
  timid next to mjlab's velocity task (+/-0.5).
- 2026-08-03 — raised to 0.4, deliberately past that point.
- 2026-08-14 — briefly committed as `0.0`, which disabled every disturbance in
  the task; caught in review and amended back to 0.4.

## push_angular_velocity

**Current:** `0.0` — the `roll`/`pitch` components of the push are off. It has
been 0.0 since the task was introduced; the dial exists so angular disturbance
can be added without restructuring the event term.

## warmup_s

**Current:** `10.0` — how long an episode runs before pushes begin. The cadence
is untouched: the term suppresses rather than reschedules, so pushes still arrive
every 5-7 s once they start, and the first real push lands on the first tick
after the warm-up, which desynchronises it across envs instead of hitting every
robot at the same phase.

**Re-measure if:** the posture-settle time changes. The warm-up exists to clear
it.

**How it suppresses:** `push_and_record` drops the envs still inside `warmup_s`
and returns without pushing — it does **not** reschedule. `EventManager` owns the
countdown and re-samples it whenever the term fires regardless of what the term
returns, so skipping here delays the *first* push without altering the 5-7 s
cadence that follows. It also means a timer-watching observer counts pushes that
never landed; see [evaluation.md](evaluation.md#binning-by-time-since-a-push).

**History:**
- Measured over 528 zero-residual episodes, the hazard rate is 0.005/s before the
  first push, 0.122/s across the 4-8 s window that contains it, and ~0.019/s flat
  for the rest of the episode. **48% of all deaths were that one event**, which
  lands while the robot is still finishing its ~4 s posture settle: the task was
  scoring a startup lottery rather than push recovery while walking.
- Removing it makes the task easier (survival ~20% -> ~39% at a 60 s cap), which
  is why `WALK_WINDOW_S` grew alongside.

## episode_length_s

**Current:** `90.0` — how long the base controller actually walks, and therefore
how long an episode is worth running. An episode running past the walk trains the
residual on a stationary robot, which is the opposite of this task — doubly so
since a standing robot outscores a walking one on both tracking terms (0.75 vs
0.66 for ZMP).

**Re-measure if:** the installed FSM changes. This tracks the *installed*
controller, not anything in this repo.

The FSM currently walks indefinitely, so the ceiling is ours to pick:
`Logistic::FSMMoveBoxTableToLeftShelf` begins with `Walking::WalkCmdVelImpl`
(`targetCmdVel: [0.1, 0, 0]`, `timeout: 1000.0`) rather than
`Logistic::GoToTable`, which is commented out. That edit lives in the installed
workspace file
(`~/workspace/install/lib/mc_controller/etc/LogisticController_ismpc.yaml`), not
in this repo, and a workspace rebuild reverts it. The top-level `transitions:`
map alone does not show this — it ends at `Logistic::Demo`, and the walk is
inside that Meta state's own transitions.

**History:**
- `16.0` — sized for the stock config, which walked 1 m to the table and then
  stood for good.
- `60.0` — after the installed FSM was changed to walk indefinitely. ~4 s of
  posture settling then ~6 m of walking.
- `90.0` — when `PUSH_WARMUP_S` arrived; the two have to move together. The
  warm-up removes the first-push massacre, which on its own would have lifted
  survival from ~20% to ~39% and given away the headroom `PUSH_VELOCITY = 0.4`
  was calibrated for. At the measured post-warm-up hazard of ~0.019/s,
  `exp(-0.019 * (T - 10))` puts 90 s back at ~22%: the same difficulty as before,
  with the mortality spread across steady walking instead of piled onto one
  startup event. It also buys 13.3 pushes per episode against 10, and a third
  fewer resets per hour — worth real throughput, since an env reset destroys and
  rebuilds its mc_rtc controller.

**Survival numbers from before the warm-up change are not comparable to ones
after it.**
