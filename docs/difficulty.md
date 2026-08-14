# Difficulty

How hard the residual balance task is: how hard the robot is shoved, when the
shoving starts, and how long an episode runs. All three interact — changing one
without the others moves the baseline survival rate, which is the number the
task is actually calibrated against.

Constants live in `src/mc_mjlab/tasks/residual_balance/residual_balance_env_cfg.py`.

## PUSH_VELOCITY

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

## PUSH_ANGULAR_VELOCITY

**Current:** `0.0` — the `roll`/`pitch` components of the push are off. It has
been 0.0 since the task was introduced; the dial exists so angular disturbance
can be added without restructuring the event term.

## PUSH_WARMUP_S

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

## WALK_WINDOW_S

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
