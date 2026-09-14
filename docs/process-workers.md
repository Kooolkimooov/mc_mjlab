# Implementing process workers

## IPCSocket

**Current:** the native IPC path uses nanomsg `NN_PAIR` and JSON through Glaze's
automatic aggregate reflection. `send(layout)` and `receive<IoLayout>()` use
the layout's existing members directly, without a separate field list or loader.
The IPC targets and controller core use C++20.
CMake FetchContent downloads Glaze 7.0.2 and CLI11 2.6.1 at pinned commits into
the build tree. No system installation is required; the first configure needs
network access unless sources are supplied through FetchContent's local overrides.
Commands, complete layouts and completion/error replies
use the same channel; controller input/output arrays remain shared-memory data,
not JSON. Messages are limited to 1 MiB. Nanomsg owns its background I/O threads;
controller methods still execute on the worker's main thread.

`IPCSocket(endpoint, Mode::Bind)` is the parent endpoint and
`Mode::Connect` is the child endpoint. Use a unique `ipc://` path in a private
directory for each worker generation. `ControllersManager` owns that directory
and removes it after closing the channels and confirming child exit. Endpoints
include the worker's row range and generation, so queued commands cannot cross
generations.

`ControllersManager` partitions controller rows, launches one process per slice,
sends each `WorkerStartMessage`, and awaits its startup `Reply`. After that,
`dispatch(Command::...)` and `collect()` allow one request per worker to be
outstanding. A completed reply means the worker command finished, not that every
controller row is healthy; inspect the output status column separately. PAIR does
not automatically retry commands as nanomsg REQ does.

Manager shutdown drains an outstanding operation, sends `Stop`, waits for its
reply and process exit, then forcibly terminates and reaps workers that exceed the
bounded shutdown deadline. Closing a nanomsg socket can discard queued messages;
immediate destruction is not a delivery guarantee. Peer death does not produce
the raw socket's EOF, so collection deadlines provide the failure signal.

Build `mc_rtc_worker` and launch it with
`mc_rtc_worker --endpoint <ipc-endpoint> --config <configuration-path> --num-controllers <count>`.
CLI11 provides argument validation and `--help`.
Startup maps and binds the worker's shared-memory slice and constructs a host
before replying. Binding does not initialize controllers or write output, which
lets a replacement stay quarantined until the failed environments reset.

For the native checks, configure with `MC_RTC_INTERFACE_TESTS=ON`, build
`test_worker_ipc`, then run `ctest --test-dir build -R '^worker_ipc$'
--output-on-failure`. The test uses freshly spawned processes, a layout larger
than 64 KiB, error replies, pending-request rejection, timeout invalidation and
the actual worker executable's startup/error/stop path.

**Re-measure if:** transport, message format, deadlines or shutdown ownership changes.

**History:**
- 2026-09-08 — replaced raw socket scaffolding with nanomsg IPC; spawning and
  shared-memory mapping were still user-led implementation steps.
- 2026-09-11 — the native manager owns spawning, mapping and recovery.

## ControllersHost

**Current:** a synchronous native building block, constructed with a config
path and controller count. It creates no worker threads, has no timeout and
does not contain native crashes. `ControllersManager` runs one host on each
worker process's main thread.

The manager owns scheduling and deadlines while Python owns shared-memory
allocation. Each child owns a host and a contiguous environment-row range, so
controller construction and destruction stay inside the child.

**Re-measure if:** the worker launch mechanism or native buffer interface changes.

**History:**
- 2026-09-08 — removed the threaded backend for a user-led process implementation.
- 2026-09-11 — wired the host into the native manager's process backend.

## ControllersManager

**Current:** `ControllersManager(configuration_path, num_controllers,
num_workers, configuration, timeout_ms)` partitions rows into contiguous slices
that differ by at most one. It dispatches every available worker before
collection and returns failed row ids without modifying their shared output.

Dispatch send errors, operation error replies, receive errors and timeouts all
retire a worker generation. Collection first harvests unaffected workers, then
kills and reaps failed processes before returning their rows. It does not retry
the lost command. The Python action reports an infrastructure truncation and
requests replacement from its episode-reset callback, keeping process startup
off the failure-detection step. A multi-row worker waits until its complete slice
has reset. The fresh worker is bound but uninitialized, so the next reset-bearing
control period initializes it. Replacement startup is attempted once; failure
closes the manager and raises with the worker index, generation and row range.

The native manager test uses controlled child processes to cover hangs, exits,
error replies, uneven slices, repeated generations, startup failure and bounded
shutdown. The Python binding test wedges an actual worker inside the probe
controller and verifies that only its slice fails and reset initializes its
replacement.

The parent owns shared-memory unlinking. Workers close their mappings after the
manager has stopped them, and constructor/respawn failures use the same cleanup
path as explicit close.

**Re-measure if:** worker grouping, notification transport, startup cost or
controller workload changes.

**History:**
- 2026-09-11 — native quarantine and process respawning implemented in the
  manager; the removed Python pool remains documented in `coupling-history.md`.
- 2026-09-11 — deferred replacement launch from collection to episode reset.

## ControllersManager::collect

**Current:** Ctrl-C can leave the trainer inside native collection for up to
one full `controller_timeout_ms` per outstanding worker. The trainer and workers
share the terminal foreground process group. SIGINT exits the workers, while
Python's pending `KeyboardInterrupt` is not raised until the native call returns.
Releasing the GIL in the binding does not service Python signals. IPC retries
`EINTR` with the remaining timeout, without treating it as cancellation.

Collection waits on workers sequentially with a fresh timeout for each one.
Nanomsg does not turn peer exit into an immediate EOF, and the loop does not
check child liveness before waiting. With 64 outstanding workers and the action's
60000 ms default, this permits roughly 64 minutes of waiting. Already collected
or queued replies reduce that bound. Failure logging and child retirement occur
only after the entire collection loop, explaining both terminal silence and
unreaped worker zombies during the delay.

`ControllersManager::close` has a separate scaling issue: it waits on each
worker's outstanding reply for 1000 ms before checking whether the child is
already dead. Four dead, pending workers cost approximately four seconds there;
64 can cost approximately 64 seconds. This is an alternative cleanup path,
not necessarily an additional delay after collection has already retired them.

The corrective design needs prompt cancellation/signal servicing at the Python
boundary, child-exit detection, and a shared deadline across worker waits.
Shutdown should notify all surviving workers before collecting their replies,
and reap dead children without awaiting IPC. No implementation change was made
as part of this diagnosis.

**Re-measure if:** signal handling, process groups, IPC deadlines, collection or
shutdown scheduling changes.

**History:**
- 2026-09-14 — the user interrupted profiling trainer PID `1730917`. The
  trainer remained in `poll_schedule_timeout.constprop.0`, and all 64 workers
  were `Z+` in the same foreground process group `1730885`. Live stack attachment
  was denied by ptrace restrictions; the precise live frame was not captured.
- An isolated native-manager probe stopped its own workers, dispatched a step,
  then sent those workers SIGINT followed by SIGCONT and confirmed exit signal
  2 with `waitid(WNOWAIT)`. Worker timeout was 1000 ms. Two SIGINTs sent to the
  probe's Python process after 0.05 and 0.15 seconds raised `KeyboardInterrupt`
  only after collection finished: `1.0019837799773086 s` for one worker and
  `4.00398956600111 s` for four. A separate four-worker close took
  `4.003622866002843 s`. One trial per condition; these establish timeout
  scaling, not timing variance. The probe used fresh test workers, not the
  user's interrupted training processes.
- Reproduction and raw results:
  `/tmp/mc-mjlab-exit-probe-UKYDX4/probe.py` and `results.json`.

## OutputGuard

**Current:** a scoped RAII `dup2` of fds 1 and 2 to `/dev/null`, constructed
from the row's `log_offset` flag at the three `ControllerInstance` entry points
(`initialize`, `reset`, `step`). It does **not** reliably silence mc_rtc, and
cannot: mc_rtc's loggers are asynchronous, so a scoped redirect races the
drain. Two leak paths remain open, and `console_output="none"` closes neither.

`mc_rtc/src/mc_rtc/logging.cpp:39,56,73` builds all three loggers with
`spdlog::create_async_nb<...>`. `mc_rtc::log::error(...)` enqueues and returns;
the `fwrite` happens later on spdlog's background thread. Any message drained
after the guard's destructor restores the fds lands on the real terminal. Most
messages are suppressed, a minority escape — the signature of a race, not of a
broken flag.

`GlobalConfiguration` is built in `ControllersHost`'s constructor
(`controllers_host.cpp:10-11`) from `worker.cpp:38`, before any row exists, so
no guard covers the config-parse burst at worker startup. It repeats per worker
and again on every respawn.

The removed Python implementation had neither leak, by construction: its worker
redirected once at startup and never restored (a `tempfile.TemporaryFile`, not
`/dev/null`, so error replies could attach mc_rtc's own text), and `"single"`
gave environment zero a dedicated worker so every other worker could be
silenced wholesale. Restoring that shape — decide silence once per worker
process, from a spawn argument — removes the race rather than narrowing it, and
covers the startup burst in the same move. `MC_MJLAB_WORKER_LOG_DIR`
(per-worker `worker-<pid>.log` plus `faulthandler.enable()`) was lost in the
port and has no native equivalent.

**Re-measure if:** mc_rtc stops constructing its loggers with
`create_async_nb`, or worker silencing moves out of the per-row flag.

**History:**
- 2026-09-11 — reproduced against real `LogisticController_ismpc`, 2 workers,
  6000 steps (12 s controller time), log flag asserted `0` on every row at every
  500-step checkpoint. One line escaped, between checkpoints 4000 and 4500:
  `[error] ZMP cannot be computed, projected force too small 0`
  (`mc_rbdyn/ZMP.cpp:22`, a plain `mc_rtc::log::error` from inside the guarded
  `step()`). All other controller chatter over the same run was suppressed. The
  same run leaked the startup burst twice, once per worker:
  `[info] Loading default global configuration ...`,
  `[info] Loading additional global configuration ...` (x2),
  `[info] GlobalPluginPaths: ...`.
- 2026-09-11 — identified as a regression from the native port, not a
  long-standing quirk: `5214fa2` ("Configurable mc_rtc console output
  (none/single/all)") silenced whole worker processes at startup and placed
  environment zero alone for `"single"`. The native port kept the per-row flag
  and applied it as a scoped guard, which reads as a faithful translation but
  drops the property that made it work.
