# Native training CPU profile

## ControllerInstance::step

**Current:** the 2026-09-14 capture attributes 91.06% of worker user-space CPU
samples to `mc_control::MCGlobalController::run()`. Native input marshalling is
1.24% and output extraction is 1.05%. The main measured costs are inside the
controller, particularly task/constraint preparation and robot kinematics.

| Inclusive function | Worker CPU samples |
| --- | ---: |
| `ControllerInstance::step` | 93.32% |
| `mc_control::MCGlobalController::run` | 91.06% |
| `mc_solver::TasksQPSolver::runOpenLoop` | 61.41% |
| `tasks::qp::QPSolver::preUpdate` | 38.87% |
| `mc_rbdyn::Robot::forwardKinematics` | 24.04% |
| `tasks::qp::TrajectoryTask::update` | 20.10% |
| `tasks::qp::CollisionConstr::update` | 13.03% |
| `tasks::qp::QLDQPSolver::solve` | 10.85% |
| `mc_control::MCController::runObserverPipelines` | 8.69% |
| `Walking_controller::run` | 4.96% |
| `ControllerInstance::apply_input` | 1.24% |
| `ControllerInstance::apply_output` | 1.05% |

These are inclusive percentages: callers include their callees, and rows must
not be added. `preUpdate` includes trajectory and collision updates. Its 38.87%
is substantially larger than the numerical QLD solve's 10.85%. `ql0002_` alone
accounts for 10.82% self time. Collision geometry and repeated kinematics are
promising places to inspect before changing solver algorithms.

The worker executable's own instructions account for approximately 0.68% of
worker samples, summing the rounded leaf rows; calls it makes into libraries
are counted in those libraries. Nanomsg accounts for approximately 0.05% leaf
time. This does not measure blocked IPC time or prove that the process topology
is optimal. Kernel execution, scheduling delays, GPU execution and blocked
time are outside this user-space CPU profile.

**Re-measure if:** controller tasks, collision pairs, robot, environment/worker
counts, process/thread configuration, native code or installed libraries change.

**History:**
- 2026-09-14 — restarted the user's configuration in `mc_mjlab_training:2.1`:
  HRP5P/LogisticController_ismpc residual-balance position control, 512
  environments, 64 workers (8 controllers each), 15 iterations, TensorBoard,
  run name `cpp-profile-15`, trainer PID `1730917`. The previous trainer and
  its workers had already exited by restart time; no kill was needed.
- Recorded `cpu-clock:u` at 99 Hz with 8192-byte DWARF stack samples from the
  trainer and every live worker. Capture ended at 11:31:59 JST; first/last
  sample timestamps were `259634.947116` / `259665.403562`, a 30.456446-second
  span during the first rollout after controller initialization. There were
  58,184 samples and zero lost samples: 52,820 worker samples and 5,364 samples
  from trainer-process threads. Sample periods total 533.5353482 worker CPU
  seconds and 54.18181764 trainer-process CPU seconds; these aggregate across
  cores and are not elapsed time.
- Machine: Ryzen 9 9950X3D, 32 logical CPUs, RTX 5080. Native build:
  `Release`, `-O3 -DNDEBUG`, branch HEAD
  `0f1e8bce201142960f629d713143aaac12f2a7a8` plus the existing uncommitted
  logging edits. No controller code or training hyperparameters were changed
  for the capture. The run directory contains the startup Git diff and normal
  controller-provenance snapshots.
- First iteration: collection `99.97525143623352 s`, learning
  `0.14902186393737793 s`. Iteration 2: collection `79.25883960723877 s`,
  learning `0.10346460342407227 s`; cumulative falls, collapsed episodes,
  controller failures and worker failures were all zero at that heartbeat.
  The first iteration includes warmup and profiling overhead; this is not a
  controlled throughput comparison against earlier runs.

## OpenMP barrier waiting

**Current:** in the trainer's `train` thread-name group, leaf CPU samples are
86.40% `gomp_barrier_wait_end` and 3.15% `gomp_team_barrier_wait_end` in PyTorch's
bundled `libgomp.so.1`: 89.55% combined. This is on-CPU barrier waiting, not
89.55% of total training elapsed time. Workers account for most aggregate CPU
time in this capture.

A useful next experiment is a matched run with reduced PyTorch/OpenMP CPU
threading, measuring rollout time and worker utilization. No such experiment
was performed here, so a speedup or its magnitude is not established. The
trainer's Python/CUDA ancestry is mostly absent from the recovered call stacks;
its leaf symbols remain available, but this capture cannot locate every
barrier's originating Python operation.

**Re-measure if:** PyTorch CPU thread count, OpenMP waiting policy, tensor
placement, worker count or rollout operations change.

**History:** 2026-09-14 — identified from the same 58,184-sample capture.

## Profiling artifacts

**Current:** artifacts are stored with the run at
`logs/rsl_rl/Mc-Mjlab-Residual-Balance-Logisticcontroller-Ismpc-Hrp5P-Position/2026-09-14_11-30-33_cpp-profile-15/native-profile/`:

- `training-perf.data`: original approximately 488 MiB recording.
- `workers-self.txt`, `workers-inclusive.txt`, `trainer-self.txt`,
  `trainer-inclusive.txt`: perf tables filtered by command name and normalized
  to that group's samples, with inline expansion disabled.
- `perf-tables.json`: machine-readable versions of those tables; authoritative
  source for the percentages above.
- `stacks.folded`, `summary.json`, `analyze.py`, `reports.py`: recovered stack
  export, process totals and reproducible analysis helpers. Folded stacks omit
  samples with no recovered frames and should not replace the perf leaf tables.
- `perf.data`: the earlier, unusable capture of the interrupted run, containing
  one CUDA background-thread sample. It is not used in any result above.

Capture command, with `profile_pids` containing the trainer and 64 live workers:

```sh
perf record -e cpu-clock:u -F 99 --call-graph dwarf,8192 \
  -p "$profile_pids" -o training-perf.data -- sleep 30
```

From the repository, rerun either analysis helper with `uv run --no-sync python`
and its full path. The helpers resolve input files beside themselves.

| Captured input | SHA-256 |
| --- | --- |
| Installed worker executable | `36b38ddde9ad4052b9a395b2748fd9d5518623afa8f7d2a45c85e9aca5c7cf01` |
| Installed Python extension | `dfc3833eb382b4f7294f1f326f7b6f4b9051d76b500fd438802948f30b24b388` |
| Project mc_rtc YAML | `dbefb7e305e95a9eb61bc89f49615099d634ffe896d1c1596d5484916c2ddee5` |
| User mc_rtc YAML | `6ab47211a23d5a66e25dccc699ce9b6d0d49fd1f4f7762613595f99629254d05` |
| Installed LogisticController_ismpc YAML | `d242025866628dcc745bf07719f9d8f789b33caa0c8077d1c79f6af774ce0c2b` |

**Re-measure if:** artifacts are removed or their binary/configuration identity
no longer matches the workload under discussion.

**History:** 2026-09-14 — recording and tables retained under the profiling run.
