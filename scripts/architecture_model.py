#!/usr/bin/env python3
"""What the architecture docs say in prose; the facts around it are extracted."""

from dataclasses import dataclass

BANNER = (
  "<!-- Generated from the source. Do not edit. -->\n"
  "<!-- Rebuild: uv run python scripts/generate_architecture_docs.py -->"
)

INDEX_INTRO = """\
Six views of the same system, generated from the source. They are an index into
the rest of `docs/`, not a replacement for it: a diagram states structure, and
never restates a measurement. Where a number matters, the section links to the
note that owns it.

Read them in order the first time. `code-map` is the machine's own view and is
derived end to end; the others pair extracted facts with prose that explains what
the extraction cannot.
"""

INDEX_HOW = """\
## How these stay true

Every fact below is read out of the source at generation time, so the only way
these files can be wrong is if they are stale — and `--check` makes staleness a
build failure rather than a surprise:

```sh
uv run python scripts/generate_architecture_docs.py           # rewrite
uv run python scripts/generate_architecture_docs.py --check   # fail on drift
uv run python scripts/generate_architecture_docs.py --live    # runtime inventory
```

Extraction uses off-the-shelf tools where they exist — `griffe` for the API
model, `grimp` for the import graph, `pyreverse` for the class diagram, `tomllib`
for the entry point — and stdlib `ast` for the four things none of them cover:
manager terms by construction site, `IoLayout`'s offsets, the worker pipe
vocabulary, and the statement order inside `apply_actions`.

The prose lives in `scripts/architecture_model.py`, beside the extraction rules.
Editing a generated file directly is pointless; the next run overwrites it.

## Drawing conventions

| Mark | Means |
| --- | --- |
| subgraph | a process boundary |
| cylinder | shared memory, or a file on disk |
| thick arrow | the per-step hot path |
| dashed arrow | asynchronous, lagged, or a failure that is swallowed |

Node labels are always quoted, because an unquoted bracket or dot ends the label
early. No `classDef` fills: every renderer here draws in the reader's own theme,
so emphasis is carried by shape and edge instead.
"""


@dataclass(frozen=True)
class Layer:
  """One generated document: its prose, and which builders fill it."""

  slug: str
  title: str
  summary: str
  claim: str
  sections: tuple[str, ...]
  notes: tuple[tuple[str, str], ...] = ()
  hazard: str = ""
  links: tuple[tuple[str, str], ...] = ()
  intro: str = ""


LAYERS: tuple[Layer, ...] = (
  Layer(
    slug="code-map",
    title="Code map",
    summary="the machine's own view: import graph, class hierarchy, module census",
    claim=(
      "Everything on this page is derived end to end. Nothing here is asserted "
      "by a human, so nothing here can be out of date without `--check` failing."
    ),
    sections=("module_census", "package_graph", "class_diagram", "binding_imports"),
    notes=(
      (
        "Why the graph is collapsed",
        "At module granularity the package is a hairball, and the three robot "
        "directories contribute three identical fans of the same shape. Collapsed "
        "to packages it is small enough to read, and the shape that matters — a "
        "DAG apart from one deliberate cycle — survives the collapse.",
      ),
      (
        "The cycle is on purpose",
        "`robots` imports the per-robot constants modules and each of those "
        "imports back into `robots`. `RobotSpec` holds callables rather than "
        "resolved values precisely so that importing the registry does not force "
        "a robot module, and therefore the mc_rtc bindings, to load.",
      ),
    ),
    links=(("How the robots differ", "../robots.md"),),
  ),
  Layer(
    slug="system-context",
    title="System context",
    summary="what this repo owns, what mjlab owns, what must exist outside",
    claim=(
      "This repo is a plugin. mjlab's console scripts drive it, the mc_rtc "
      "workspace supplies the controllers and the meshes, and a single entry "
      "point connects the two."
    ),
    sections=("entry_points", "context_diagram", "external_inputs"),
    notes=(
      (
        "Registration happens during `import mjlab`",
        "`register_mjlab_task` takes *built* cfgs, not factories, so the entry "
        "point does not merely announce the tasks — importing mjlab constructs "
        "every env cfg this repo ships, which needs a live mc_rtc workspace.",
      ),
      (
        "Nothing about the robot is transcribed",
        "Joint order, stance, base pose and limits are read from the mc_rtc "
        "`RobotModule`; the meshes, MJCF and PD gains symlink out of the "
        "workspace on first use.",
      ),
    ),
    hazard=(
      "mjlab's loader catches everything. An unsourced workspace does not raise "
      "— it prints a warning line, and the tasks are simply absent from "
      '`list-envs`. Any puzzling "my task does not exist" starts there.'
    ),
    links=(
      ("Setup for each external input", "../../README.md"),
      ("What the coupling does with them", "../coupling.md"),
    ),
  ),
  Layer(
    slug="process-topology",
    title="Process topology",
    summary="who owns which memory, and what crosses between the two sides",
    claim=(
      "One mc_rtc controller per environment. The two sides share no address "
      "space and no GIL: everything between them crosses two shared-memory "
      "blocks and a pipe per worker."
    ),
    sections=(
      "topology_diagram",
      "pipe_protocol",
      "io_layout",
      "status_column",
      "failure_diagram",
    ),
    notes=(
      (
        "Why processes at all",
        "Controller construction is serial and memory-heavy, and the Cython "
        "marshalling holds the GIL — only the patched binding's `run()` releases "
        "it. The ceiling on environment count is RAM, not the GPU.",
      ),
      (
        "Forkserver, not spawn or fork",
        "Spawn would re-import torch and mjlab per worker; fork is unsafe. The "
        "preload must not pull in numpy, and `OPENBLAS_NUM_THREADS` is pinned to "
        "1 before the first launch so every worker and every respawn inherits it.",
      ),
      (
        "Blocks are passed by name",
        "So a respawned worker re-attaches to the same memory instead of the pool "
        "having to rebuild it.",
      ),
    ),
    hazard=(
      "Folding worker failure into `controller_failed` taught the critic that "
      "some states are worth &minus;200 for reasons absent from the observation. "
      "It is configured `time_out`, so the value bootstraps and nothing is charged."
    ),
    links=(
      ("Timeouts, quarantine and the row picture", "../coupling.md#controllerpool"),
      ("Why the truncation", "../coupling.md#worker-failure-is-a-truncation"),
      ("Sensor routing", "../coupling.md#sensor-routing"),
    ),
  ),
  Layer(
    slug="control-step",
    title="One control step",
    summary="the order of one period, and why targets are one period stale",
    claim=(
      "The deliberate deviation from mc_mujoco: a controller step is dispatched "
      "without blocking and collected one control period later, so the solve "
      "overlaps the sim substeps that follow it."
    ),
    sections=("step_sequence", "rate_stack", "reset_sequence"),
    intro=(
      "The beats below are not asserted — they are the statement order inside "
      "`apply_actions` and the reset path, read from the source. Only the "
      "reasoning around them is written by hand."
    ),
    notes=(
      (
        "The lag is the trade",
        "Targets lag the state that produced them by one controller period, "
        "bought in exchange for overlapping the solve with the GPU sim. One step "
        "is outstanding per worker, which is what makes `collect` a plain await "
        "rather than a queue.",
      ),
      (
        "Encoders are fed biased",
        "On purpose. These are the robot's encoders, so the controller's own "
        "state estimate carries the calibration error the policy observes rather "
        "than being handed ground truth the real one never sees.",
      ),
      (
        "Collect comes first at reset",
        "A step may still be in flight, and the workers must be finished reading "
        "the input block before anything overwrites it. The extracted order shows "
        "this directly.",
      ),
    ),
    hazard=(
      "Inside the reset the worker teleports the *estimated* robot and only then "
      "re-runs the controller reset. Skipping that killed 39% of episodes with "
      "nothing to show for it but a survival rate."
    ),
    links=(
      ("Dispatch and interpolation", "../coupling.md#dispatch-and-interpolation"),
      ("Reset pose seeding", "../coupling.md#reset-pose-seeding"),
      ("The control laws", "../coupling.md#position-control-law"),
    ),
  ),
  Layer(
    slug="task-wiring",
    title="Task wiring",
    summary="one yaml file to task ids to env cfgs to managers to the runner",
    claim=(
      "`etc/mc_rtc.yaml` decides the task ids, and the task id decides which "
      "checkpoints a run may resume."
    ),
    sections=(
      "wiring_diagram",
      "task_ids",
      "registration_table",
      "term_census",
      "injected_classes",
    ),
    notes=(
      (
        "Only sub-packages register",
        "`tasks/__init__` walks its sub-packages. A task added as a bare module "
        "beside it is never imported and would silently never register.",
      ),
      (
        "Ids are generated, not written",
        "`.title()` and `_`-to-`-` are applied to the whole string, so an id is "
        "not the yaml's spelling. Run `list-envs`; never hand-assemble one.",
      ),
      (
        "Terms are counted at their construction site",
        "The manager dicts are built and then mutated — conditional `|=` merges, "
        "a `.pop`, and a comprehension that derives the whole critic group — so "
        "the dict displays alone would report no critic terms at all. Counting "
        "constructor calls instead includes every conditional variant, which is "
        "why these numbers exceed what any single task registers.",
      ),
    ),
    hazard=(
      "Every `Episode_Reward/*` is an episode sum and moves with survival time. "
      "`zmp_error` only means something divided by `zmp_grounded`."
    ),
    links=(
      ("Why each observation exists", "../observations.md"),
      ("What the weights buy", "../reward-shaping.md"),
      ("Curricula and disturbances", "../difficulty.md"),
    ),
  ),
  Layer(
    slug="robot-assets",
    title="Robot and assets",
    summary="MainRobot to a loaded MJCF with the right gains and sensors",
    claim=(
      "`MainRobot` is the single source of truth for which robot runs: it selects "
      "the mjlab entity, and it appears in every task id."
    ),
    sections=("robot_registry", "asset_diagram", "gain_ordering"),
    notes=(
      (
        "Three robots, parallel by construction",
        "Each constants module is thin and differs only in the robot-specific "
        "names — root body, foot bodies, deactivated joints — delegating the rest "
        "to the shared configuration modules.",
      ),
      (
        "Unnamed collision geoms",
        "The robot XMLs ship unnamed collision geoms, so mjlab's name-based "
        "presets would match nothing — and a preset matching nothing drops the "
        "robot through the floor. Each spec names its geoms before disabling them "
        "by group, and the flag recording that cannot be inferred from a "
        "non-empty `collisions` field.",
      ),
    ),
    hazard=(
      "Without the real gains a walking controller falls. The torque action then "
      "zeroes them again, because it applies the PD fallback itself."
    ),
    links=(
      ("Geoms, gains, sensors, assets", "../robots.md"),
      ("Invariants and traps", "../coupling.md#invariants-and-traps"),
    ),
  ),
)

SYMBOLIC = """\
The expressions stay symbolic because their inputs are not in the source.
`num_targets` and the lengths of `imu` and `wrenches` come from the mc_rtc
`RobotModule` and from `HostMetadata` probed inside a worker, so a static read
can name every column but never count them.
"""

LIVE_INTRO = """\
Written by `--live`, from a real `import mjlab` on a sourced workspace. It is
excluded from `--check`: it describes one machine at one moment, and it is the
only page here that cannot be regenerated without the mc_rtc bindings.
"""

LIVE_TERMS = """\
The static census counts constructor calls, which includes every conditional
variant and cannot see the critic group at all, because that group is derived by
a comprehension. These are the terms the managers actually hold.
"""

NOT_IN_SOURCE = """\
Three things a reader may want are deliberately absent, because they are not in
the source to extract:

- **Why each weight, band and threshold is what it is.** The value is a literal;
  the measurement behind it lives in `docs/`.
- **The mc_rtc robot's real identity** — joint names, `refJointOrder`, effort
  limits, sensor names. All of it arrives from the `RobotModule` at runtime.
- **Which values are load-bearing versus ablated-and-kept.** That distinction is
  a lab record, not a program property.
"""

SECTION_TITLES: dict[str, str] = {
  "module_census": "Module census",
  "package_graph": "Package import graph",
  "class_diagram": "Class hierarchy of the coupling",
  "binding_imports": "Where the mc_rtc bindings enter",
  "entry_points": "The one entry point",
  "context_diagram": "The system in context",
  "external_inputs": "What has to exist outside",
  "topology_diagram": "The two sides",
  "pipe_protocol": "The worker pipe vocabulary",
  "io_layout": "IoLayout, column by column",
  "status_column": "The status column",
  "failure_diagram": "Failure containment",
  "step_sequence": "The step, in source order",
  "rate_stack": "The rate stack",
  "reset_sequence": "Reset, in source order",
  "wiring_diagram": "From yaml to a running env",
  "task_ids": "The task ids",
  "registration_table": "Every registration",
  "term_census": "Manager term census",
  "injected_classes": "Classes injected by name",
  "robot_registry": "The robot registry",
  "asset_diagram": "Asset resolution",
  "gain_ordering": "PD gains, and when they can be applied",
}

BUILDER_PROSE: dict[str, str] = {
  "pipe_protocol": (
    "Commands the worker handles, against commands the pool actually sends. The "
    "generator reads both ends and fails if they disagree, so a renamed command "
    "cannot pass silently."
  ),
  "io_layout": (
    "The column layout of both shared blocks, one row per env. These are not "
    "transcribed: each offset is a single-expression `@property`, so the table "
    "below is those expressions, resolved against each other in dependency order."
  ),
  "status_column": (
    "One column, three values. A dead worker and a collapsed QP look identical "
    "from the sim and must not be scored the same."
  ),
  "term_census": (
    "Manager terms by the constructor used to build them, across the task's env "
    "cfg module."
  ),
  "task_ids": (
    "Resolved against the committed `etc/mc_rtc.yaml`, through the repo's own "
    "`get_task_name` — the same function the demo script and the task packages "
    "call, so these are the real ids and not a reconstruction."
  ),
  "injected_classes": (
    'Classes handed to rsl_rl as `"module:Class"` strings. No import edge '
    "exists for these, so the import graph above cannot show them and a rename "
    "would not break any import."
  ),
  "class_diagram": (
    "Straight out of `pyreverse`, which reads the source with astroid rather "
    "than importing it."
  ),
  "package_graph": "Internal imports, collapsed from module to package.",
  "module_census": "",
  "binding_imports": (
    "Exactly which modules reach the mc_rtc bindings. Both imports are written "
    "so that the rest of the package loads without them: one is guarded by "
    "`try/except ImportError`, the other is deferred into a cached function body."
  ),
  "entry_points": (
    "Everything mjlab needs to find this package. `train`, `play` and "
    "`list-envs` are mjlab's own console scripts; this repo ships none."
  ),
  "external_inputs": (
    "Paths and files that live outside the repo and must exist before anything "
    "steps a controller."
  ),
  "rate_stack": (
    "Read from the literal arguments in the env cfg. The residual is held across "
    "one policy step, which is several controller periods."
  ),
  "robot_registry": (
    "The registry maps the yaml's `MainRobot` to a spec. Note that a key need "
    "not match its directory name."
  ),
  "gain_ordering": (
    "The one ordering constraint on this page. `PDgains_sim.dat` is indexed by "
    "the controller's `ref_joint_order`, which the main process does not learn "
    "until a worker has built a controller and replied with its metadata — so "
    "the real gains can only overwrite the armature-derived defaults at "
    "action-term init, never at cfg time."
  ),
  "wiring_diagram": "",
  "context_diagram": "",
  "topology_diagram": "",
  "failure_diagram": "",
  "step_sequence": "",
  "reset_sequence": (
    "The reset path, in the order the source runs it. Draining the outstanding "
    "dispatch comes first because the workers must be done reading the input "
    "block."
  ),
  "asset_diagram": "",
}


PAGE_CSS = """
*, *::before, *::after { box-sizing: border-box; }

:root {
  --paper: #edf0ef;
  --surface: #fbfcfb;
  --ink: #12211f;
  --muted: #5b6a68;
  --rule: #d1d8d6;
  --rule-soft: #e1e6e4;
  --accent: #0b6a60;
  --hazard: #944c24;
  --hazard-soft: #f0e5db;
  --plate-bg: #fbfbf9;
  --plate-rule: #dcdfda;
  --plate-ink: #26261f;
  --lift: 0 1px 2px rgba(18, 33, 31, .05), 0 14px 34px -26px rgba(18, 33, 31, .7);
}

:root:not([data-theme="light"]) { color-scheme: light; }

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --paper: #0c1312;
    --surface: #121b19;
    --ink: #e3eae8;
    --muted: #93a29f;
    --rule: #24312e;
    --rule-soft: #1b2624;
    --accent: #4fbfae;
    --hazard: #d59263;
    --hazard-soft: #2c2018;
    --lift: 0 1px 2px rgba(0, 0, 0, .4), 0 14px 34px -26px rgba(0, 0, 0, .9);
  }
}

:root[data-theme="dark"] {
  color-scheme: dark;
  --paper: #0c1312;
  --surface: #121b19;
  --ink: #e3eae8;
  --muted: #93a29f;
  --rule: #24312e;
  --rule-soft: #1b2624;
  --accent: #4fbfae;
  --hazard: #d59263;
  --hazard-soft: #2c2018;
  --lift: 0 1px 2px rgba(0, 0, 0, .4), 0 14px 34px -26px rgba(0, 0, 0, .9);
}

body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  font-size: 16px;
  line-height: 1.62;
  -webkit-font-smoothing: antialiased;
}

code, .mono { font-family: "IBM Plex Mono", ui-monospace, monospace; }

code {
  font-size: .86em;
  background: var(--rule-soft);
  border-radius: 3px;
  padding: .08em .32em;
}

a { color: var(--accent); text-underline-offset: 2px; }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }

.wrap { max-width: 1180px; margin: 0 auto; padding: 0 28px; }

.masthead { border-bottom: 1px solid var(--rule); background: var(--surface); }

.masthead .wrap {
  display: grid;
  gap: 30px 56px;
  grid-template-columns: minmax(0, 1fr);
  padding-top: 56px;
  padding-bottom: 44px;
}

@media (min-width: 900px) {
  .masthead .wrap { grid-template-columns: minmax(0, 1fr) auto; align-items: end; }
}

.eyebrow {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11.5px;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--accent);
  margin: 0 0 14px;
}

h1 {
  font-family: Archivo, ui-sans-serif, system-ui, sans-serif;
  font-weight: 700;
  font-size: clamp(2.15rem, 1.3rem + 3vw, 3.3rem);
  letter-spacing: -.024em;
  line-height: 1.03;
  text-wrap: balance;
  margin: 0;
}

.lede { margin: 16px 0 0; max-width: 62ch; font-size: 1.06rem; color: var(--muted); }

.strip { margin: 0; }
.strip svg { width: 100%; max-width: 680px; height: auto; display: block; }

.sv-l {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 13px;
  font-weight: 500;
  fill: var(--ink);
}

.sv-s {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11px;
  fill: var(--muted);
}

.sv-a { stroke: var(--rule); stroke-width: 1; }
.sv-t { stroke: var(--accent); stroke-width: 1.2; }
.sv-b { stroke: var(--rule); stroke-width: 1; stroke-dasharray: 2 4; }

.strip figcaption { margin-top: 10px; font-size: 13px; color: var(--muted); max-width: 46ch; }

.shell {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 0 56px;
  padding-top: 40px;
  padding-bottom: 72px;
}

@media (min-width: 900px) {
  .shell { grid-template-columns: 200px minmax(0, 1fr); }
}

.rail { display: none; }

@media (min-width: 900px) {
  .rail {
    display: block;
    position: sticky;
    top: 32px;
    align-self: start;
    max-height: calc(100vh - 64px);
    overflow-y: auto;
  }
}

.rail h2 {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11px;
  letter-spacing: .16em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 500;
  margin: 0 0 14px;
}

.rail ol { list-style: none; margin: 0; padding: 0; display: grid; gap: 2px; }

.rail a {
  display: grid;
  grid-template-columns: 30px minmax(0, 1fr);
  align-items: baseline;
  gap: 6px;
  padding: 7px 8px 7px 0;
  color: var(--ink);
  text-decoration: none;
  border-top: 1px solid var(--rule-soft);
  font-size: 14.5px;
}

.rail li:first-child a { border-top: 0; }

.rail a span:first-child {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11.5px;
  color: var(--accent);
}

.rail a:hover { color: var(--accent); }

.rail .source {
  margin-top: 22px;
  padding-top: 16px;
  border-top: 1px solid var(--rule-soft);
  font-size: 12.5px;
  color: var(--muted);
  line-height: 1.5;
}

main { display: grid; gap: 72px; min-width: 0; }

.layer { min-width: 0; scroll-margin-top: 24px; }

.tag {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11.5px;
  letter-spacing: .16em;
  color: var(--accent);
  margin: 0 0 6px;
}

.layer h2, .layer h3, .layer h4 { text-wrap: balance; }

.layer h2 {
  font-family: Archivo, ui-sans-serif, system-ui, sans-serif;
  font-weight: 600;
  font-size: clamp(1.5rem, 1.2rem + 1.1vw, 1.95rem);
  letter-spacing: -.018em;
  line-height: 1.14;
  margin: 0 0 14px;
}

.layer h3 {
  font-family: Archivo, ui-sans-serif, system-ui, sans-serif;
  font-size: 1.06rem;
  font-weight: 600;
  margin: 38px 0 10px;
  padding-top: 14px;
  border-top: 1px solid var(--rule-soft);
}

.layer p { max-width: 70ch; }

.layer pre.mermaid {
  margin: 20px 0;
  background: var(--plate-bg);
  border: 1px solid var(--plate-rule);
  border-radius: 4px;
  box-shadow: var(--lift);
  color: var(--plate-ink);
  padding: 22px 20px;
  overflow-x: auto;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 13px;
  text-align: center;
}

.tablewrap { overflow-x: auto; margin: 16px 0; }

table { border-collapse: collapse; font-size: .9rem; width: 100%; }

th, td {
  text-align: left;
  padding: 7px 14px 7px 0;
  border-bottom: 1px solid var(--rule-soft);
  vertical-align: top;
  font-variant-numeric: tabular-nums;
}

th {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11px;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--muted);
  font-weight: 500;
  border-bottom: 1px solid var(--rule);
}

ul { max-width: 70ch; }

footer { border-top: 1px solid var(--rule); background: var(--surface); }

footer .wrap { padding-top: 26px; padding-bottom: 40px; font-size: 13.5px; color: var(--muted); }

footer p { margin: 0 0 6px; max-width: 78ch; }

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
"""
