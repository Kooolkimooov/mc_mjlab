# Unattended training watchdog

## watch_training.py

**Current:** every residual-balance trainer with a log directory publishes a
PID-bound runtime marker and one heartbeat per completed PPO iteration under
`<run>/watchdog/`. `scripts/watch_training.py` attaches only when both the
explicit PID and Linux process-start token match that marker. It monitors
trainer and checkpoint freshness, physical GPU memory, controller-worker
failures after attachment, and optional paired-qualification regressions.

Launch training in its usual tmux window. Once `<run>/watchdog/runtime.json`
exists, attach the monitor in another window:

```sh
run_dir=/absolute/path/to/the/run
trainer_pid=$(jq -r .pid "$run_dir/watchdog/runtime.json")
uv run python scripts/watch_training.py \
  --pid "$trainer_pid" --run-dir "$run_dir" --max-level stop
```

The defaults escalate at these levels:

| Signal | Warn | Preserve | Stop |
| --- | ---: | ---: | ---: |
| Heartbeat age | 5 min | 15 min | 30 min |
| Checkpoint age | 1 h | 2 h | 4 h |
| GPU free memory | 4 GiB | 2 GiB | 768 MiB for two polls |
| New controller-worker failures | 1 | 3 | 8 |
| Consecutive qualification regressions | 1 | 2 | 3 |

`--max-level warn` is observation-only. `preserve` permits checkpoint requests
but never ends the trainer. `stop` permits a cooperative stop, but only after a
preserve request has been acknowledged. The monitor never sends `SIGKILL` or
`SIGTERM`: if a trainer is too wedged to honor a preserve request, the request
stays pending and the verdict remains explicit about the missing acknowledgement.

The GPU index is the physical index reported by `nvidia-smi`, not CUDA's
post-`CUDA_VISIBLE_DEVICES` index. Override `--gpu-index` when those differ.
Thresholds are command-line options and the effective set is copied into every
verdict.

**Re-measure if:** iteration time, checkpoint density, worker reliability, GPU
sharing, or checkpoint size changes materially. Heartbeat thresholds should be
several times longer than the slowest healthy iteration, and checkpoint
thresholds several times longer than the configured save interval.

**History:**

- 2026-08-26 — one-env live runs forced the full preserve-before-stop path. The
  final protocol run issued exactly two requests: preserve saved
  `watchdog/model_27.pt`, then stop saved `watchdog/model_28.pt` before a clean
  `stopped_by_watchdog` exit. An earlier stop checkpoint loaded on CPU at
  iteration 220 with its effective manifest and curriculum state intact.
- 2026-08-26 — a separate 40-iteration run remained attached through normal
  budget exhaustion and produced `completed`, not a crash or watchdog-failure
  verdict.

## qualification-report

**Current:** `--qualification-report <qualification.json>` adds held-out
policy degradation to the operational guards. A report that already exists at
attachment is deliberately ignored. The first two newly written reports form a
post-attach median baseline; later reports regress when eligibility is lost,
hazard ratio rises by more than `0.10`, or recovery gain falls by more than
`0.05`. Both margins and the baseline sample count are configurable.

The qualifier is not run inside the trainer or monitor. Schedule
`qualify_checkpoints.py` separately and rewrite the same report path after each
candidate checkpoint. A partially written report is treated as transient and
retried. This keeps expensive held-out simulation out of the training process
and prevents a qualifier crash from being mistaken for trainer failure.

**Re-measure if:** multi-seed variance from tranche 5 establishes tighter or
looser margins. Until then, qualification escalation is opt-in by providing the
report path.

**History:**

- 2026-08-26 — the monitor began consuming the promotion contract introduced by
  `qualify_checkpoints.py` without changing its gates or treating training
  curves as qualification evidence.

## watchdog artifacts

**Current:** the protocol is a small set of independently readable JSON files:

| File | Writer | Meaning |
| --- | --- | --- |
| `runtime.json` | trainer | immutable PID, process-start token, run path, and protocol capability |
| `heartbeat.json` | trainer | iteration, timings, PPO diagnostics, CUDA memory, terminations, and latest checkpoint |
| `status.json` | trainer | `starting`, `running`, `completed`, `failed`, or `stopped_by_watchdog` |
| `request.json` | monitor | pending `preserve` or `stop` request and its reason |
| `response.json` | trainer | accepted/rejected request and durable checkpoint path |
| `verdict.json` | monitor | latest health decision, thresholds, evidence, and next action |
| `events.jsonl` | monitor | append-only escalation and lifecycle history |
| `model_<iteration>.pt` | trainer | watchdog-preserved checkpoint inside this directory |

All replaceable JSON files are written through an atomic rename. Normal budget
completion, checkpointed watchdog stop, reported Python/OOM failure, and exit
without a terminal status are separate machine-readable verdicts. A watchdog
stop returns control to mjlab normally so the environment and controller pool
still close.

**Re-measure if:** distributed training is enabled. Heartbeats are rank-zero,
but cooperative preserve/stop is intentionally rejected for multi-GPU runs
until all ranks can acknowledge the same iteration boundary.

**History:**

- 2026-08-26 — protocol version 1 added. Checkpoint metadata records that the
  runner bridge was active, while external policy thresholds live in the
  verdict so they do not become part of the actor or resume contract.
