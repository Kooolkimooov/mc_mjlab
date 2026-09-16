# How the build is wired

Notes on the scikit-build-core settings in `pyproject.toml` and what depends on
them.

## editable.verbose

**Current:** `false` — keeps stdout clean for callers that capture it.

`editable.rebuild = true` makes the generated editable finder run
`cmake --build & --install` whenever the package is imported and a `.cpp` is
stale. With the upstream default `editable.verbose = true`, that finder prints

```
Running cmake --build & --install in <build-dir>
```

to **stdout** (the cmake output itself goes to stderr). Any caller that captures
stdout gets that banner ahead of its real output:

```sh
task_id="$(uv run python -c "
from mc_mjlab.tasks.naming import get_task_name
print(get_task_name('zero_residual', 'position'))
")"
# task_id was two lines; `play` then got an invalid task name
```

Reproduced 2026-09-16 with `scripts/demos/run_test_mc_rtc.sh`. Importing
`mc_mjlab.tasks` is what triggers it there — the package initializer walks every
task sub-package, which pulls in the compiled `mc_rtc_interface` — but the
banner is not specific to that import: it precedes the first import of the
extension by any caller, so hardening one script or moving one module only moves
the hazard.

Turning verbosity off loses nothing on failure: the finder captures the build's
stdout when it is quiet and prints it as `ERROR: ...` on a non-zero return.
`SKBUILD_EDITABLE_VERBOSE=1` restores the live stream for a build being
debugged, and `=0` forces it off regardless of this setting.

**Re-measure if:** scikit-build-core changes which stream the rebuild banner
goes to, or stops surfacing captured stdout on a failed quiet rebuild — then the
quiet setting either becomes unnecessary or starts hiding build errors.

**History:**
- 2026-09-16 — found via `run_test_mc_rtc.sh`, whose `$(...)` capture of the
  task id picked up the banner and passed a two-line task name to `play`. Set
  `editable.verbose = false`; the capture then returned the id alone.
