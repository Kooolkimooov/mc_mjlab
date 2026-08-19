# Evaluation

How to find out whether a residual policy actually beat mc_rtc, and the sampling
traps that make it easy to get a confident wrong answer. Scripts in `scripts/`.

## Why the training curves cannot answer it

`Episode_Reward/*` comes from a *stochastic* policy still carrying its
exploration noise, which on this task is a disturbance in its own right — it goes
through the real PD gains onto the joints doing the balancing. The sums also
correlate with episode length at r = +0.98, so they move when the robot survives
longer rather than when it tracks better.

## compare_to_baseline.py

Both arms run in the same env, with the same pushes, terminations and reward
terms, and the policy is evaluated deterministically (`MLPModel.forward` returns
the distribution mean), so a deficit is the learned mean being worse rather than
the dither.

```sh
uv run python scripts/compare_to_baseline.py --checkpoint logs/.../model_499.pt
```

Two things it does that a naive A/B does not, both learned by getting them wrong
first.

### Both arms at once

The envs are split in half and stepped together in one loop, so the two arms
share the wall-clock window, the machine load and the worker pool. Running one
arm and then the other would put the second on controllers already worn by the
first, and would let anything that drifts during the run land on only one of them.

The cost is that the pairing is no longer within-env: an env belongs to one arm
for the whole run, so `encoder_bias` (a per-env startup constant that moves
survival on its own) differs between the arms rather than cancelling. With envs
split evenly that is a wash in expectation, and the run is half as long.

### Fixed episodes per env, not "everything that finished"

A fall recycles in a few seconds while a survivor runs to the cap, so counting
every episode completed inside a wall-clock budget samples them proportional to
1/duration. **That inflates a failure rate badly — a true 10% can read 67% — and
it inflates it by a different factor for each arm**, which biases the very
difference being measured.

Taking the first K episodes of every env is unbiased because the selection never
looks at how long an episode lasted. The raw counts are printed alongside so the
size of that correction stays visible.

K is counted over **every env the arm owns**, not over the envs that appear in
the results. Seeding it from completed episodes instead silently drops any env
still inside its first episode at the deadline — which is to say the survivors.
That is a selection on duration again, the exact bias this exists to avoid. An
arm with such an env reports K=0 and says so rather than producing a
clean-looking number computed off the fallers alone.

### Resetting without corrupting the observation history

`ManagerBasedRlEnv.reset()` ends with
`observation_manager.compute(update_history=True)`, and `compute_group` appends
to every term's `CircularBuffer` for *all* envs, not just `env_ids` — the
buffer's write pointer is shared across the batch, so it cannot be otherwise.
`step()` has already appended this step's frame, so calling `reset()` in the loop
gives every `history_length=5` term two frames on any step where at least one env
finished, which is most of them. The window then spans half the time it should
and carries near-duplicate frames.

That is not symmetric across the arms: the baseline applies a zero action and
reads no observations at all, so the distortion lands only on the policy — it
biases the difference the script exists to measure. Hence `_reset_done`, which
makes the three calls `step()`'s own `auto_reset` path makes, minus the recompute.

The residual cost, stated rather than hidden: `get_observations()` returns the
cached buffer, so the first action of a new episode is computed from the dead
episode's last observation. One step per episode, against every step of every
episode the other way. From the next step on the histories are exactly what
training produces — the reset zeroed those rows, so the next append backfills all
five slots with the fresh frame.

### Per-step rates beat survival

**Divide episode length out and read that first.** Survival to the cap is the
headline number but it is a badly underpowered one: at ~20% it needs roughly
n = 150 per arm to resolve a 5pp difference. The per-step tracking rates resolved
a 10% deficit at n = 50 with p < 0.001, on the same data where survival came back
p = 0.70.

Measured the hard way on 2026-08-14. Three comparisons of the same training run:

| checkpoint | n/arm | survival verdict | per-step verdict |
| --- | --- | --- | --- |
| `model_1499` (previous run) | 48 | -2.1pp, p = 0.80 | not computed |
| `model_1050` | 18/24 | +0.0pp, p = 1.00 | not computed |
| `model_3050` | 48/54 | +3.2pp, p = 0.70 | **-10%, p < 0.001** |

The first two said "no detectable difference" and were read as parity. The third,
with barely more data, showed the policy was significantly *worse* all along —
the survival statistic simply could not see it. An underpowered null is not
evidence of equivalence, and this task's survival rate is underpowered at any
sample size a 20-minute run can reach.

The episode-sum rewards do not substitute: they correlate with episode length at
r = +0.98, so a policy with shorter episodes shows lower sums whether or not it
tracks worse. Only the length-normalised rate separates the two.

### Reading the output

`auto_reset = False` is essential: `step()` otherwise resets terminated envs *in
place* before returning, zeroing both `episode_length_buf` and the reward
manager's `_episode_sums`, so the episode being measured is erased before the
caller sees `terminated`.

Reward sums correlate strongly with episode length on this task, so a reward
delta that tracks the length delta is not independent evidence.

Quartiles are reported because the mean alone hides the shape: episode length is
strongly bimodal (an early fall against a full survival), so a mean that moves
can mean either "falls got later" or "more episodes reached the cap". `sem` is
what two runs have to differ by to mean anything. Quartiles use the "inclusive"
linear-interpolation convention numpy/pandas use, so they match anything the data
gets pasted into.

`--drop-obs` exists for checkpoints that predate an observation term: the actor's
first layer and its `obs_normalizer` are both sized by the concatenated width, so
adding a term retires every earlier checkpoint with a `size mismatch` at load.
Removing one from the *middle* of a group would silently shift later terms into
the wrong column rather than fail, so it is only sound for terms appended at the
end.

## Never score the last checkpoint

**Score the best-metric checkpoint, and confirm the choice with a second read.**
The last checkpoint is not the best one and can be actively broken: `zeroinit-4ev`
peaked on `Episode_Metrics/zmp_error` at iteration 1320, decayed for 1600
iterations, and then diverged outright in its last 50, so `model_2999` was corrupt
while `model_1000` was the best result this project has produced. Reading only the
final checkpoint would have reported the opposite of the truth in both directions.

The cheap procedure: smooth `Episode_Metrics/zmp_error` over ~60 iterations, take
the argmin, and compare that checkpoint plus one much later. Two reads bracket the
trajectory — a single one cannot distinguish "still improving" from "peaked and
decaying", and those call for opposite decisions about run length.

## Comparing a checkpoint whose config has moved on

Checkpoints die whenever the observation width changes, so scoring an older one
means reconstructing the cfg it trained under. A `git worktree` at the right commit
plus `PYTHONPATH=<worktree>/src:$PYTHONPATH` shadows the installed package without
touching the working tree or any live run. Two things bite:

- **Prepend to `PYTHONPATH`, never replace it.** The mc_rtc bindings arrive on it
  from the sourced workspace; clobbering it fails at `import mc_rbdyn`.
- **Copy `etc/mc_rtc.yaml` into the worktree.** It is untracked, so the worktree
  gets the *committed* `MainRobot`, and a different robot means a different joint
  count and a `size mismatch` that looks exactly like a stale-checkpoint error.

## probe_residual_authority.py

Answers a different question, and the one to ask first when a reward looks blind:
**can the action reach the objective at all?**

The residual_balance reward proved to be blind to the policy — with episode
length divided out, the per-step `zmp_tracking` rate is the same for a zero
residual and for every trained checkpoint (0.01183 / 0.01187 / 0.01196, SE
0.00015). Two things can cause that: the reward is badly shaped, or the action
cannot move the objective. This settles the second before anyone spends time on
the first, because at `residual_scale = 0.01` rad (~0.57 degrees of joint offset)
it is entirely possible the residual simply cannot shift the centre of pressure.

The probe drives a **constant** residual instead of a policy and measures
`mdp.zmp_error` — the distance in metres between the measured centre of pressure
and the one mc_rtc planned. A constant offset is the bluntest possible input: if
a full-scale one does not move that distance, nothing a policy does will either.

```sh
uv run python scripts/probe_residual_authority.py --level 0    --minutes 10
uv run python scripts/probe_residual_authority.py --level 1.0  --pattern alternating
uv run python scripts/probe_residual_authority.py --level 1.0  --pattern noise
```

Compare runs on `zmp_error mean`. Authority is adequate if a full-scale constant
residual shifts it by >= 0.007 m (20% of the ~0.036 m operating error).

### Binning by time since a push

The probe also bins the error by time since the last push, which gives the
recovery profile — how long after a disturbance the tracking error stays
elevated. That is the window a disturbance-gated reward should pay on, so the
sweep sizes `RECOVERY_WINDOW_S` as well as justifying it.

Getting that binning right took two corrections, both of which had inflated the
early bins — the ones that size the window:

- **Detect the push from the term's own record, not the interval timer.**
  `EventManager` re-samples the countdown whenever the term *fires*, including
  the ticks `push_and_record` suppresses during `PUSH_WARMUP_S`. A timer-based
  detector therefore counts pushes that never landed, and with a 10 s warm-up
  that is the first ~2 ticks of every episode. `mdp.steps_since_push` reads
  `last_push_step`, which is only set when a push actually happens.
- **Do not treat an episode reset as a push.** Setting the age to zero on reset
  binned the following 8 s of a *fresh* episode as post-push recovery — and those
  8 s contain the ~4 s posture settle, which carries the largest ZMP errors in
  the run. `steps_since_push` reports a huge age for a push older than the
  current episode, so reset envs drop out of the filter instead.

The gate is `age >= 1`, matching `recovery_tracking`'s: at age 0 the push has
been applied but no physics has run on it yet, so that sample predates its
effect. Matching the gate keeps the bins aligned with the steps the reward is
actually paid on.

See `RECOVERY_TRACKING_WEIGHT` in [reward-shaping.md](reward-shaping.md) for the
profile itself and the caveat that the recorded numbers predate these fixes.
