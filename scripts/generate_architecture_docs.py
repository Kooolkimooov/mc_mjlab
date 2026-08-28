#!/usr/bin/env python3
"""Write docs/architecture from the source; --check turns drift into a failure."""

import argparse
import ast
import difflib
import re
import sys
import tempfile
import textwrap
from collections.abc import Sequence
from pathlib import Path

# Sibling scripts, resolved by the interpreter's script directory at runtime.
import architecture_extract as ex  # ty: ignore[unresolved-import]
import architecture_model as md  # ty: ignore[unresolved-import]

OUT = ex.ROOT / "docs" / "architecture"
ACTIONS = ex.SRC / "mc_mjlab" / "actions"
TASKS = ex.SRC / "mc_mjlab" / "tasks"
HOST = ACTIONS / "mc_rtc_controller_host.py"
POOL = ACTIONS / "mc_rtc_controller_pool.py"
ACTION = ACTIONS / "mc_rtc_residual_action.py"
POSITION = ACTIONS / "mc_rtc_residual_joint_position_actions.py"
TORQUE = ACTIONS / "mc_rtc_residual_joint_torque_actions.py"
REGISTRY = ex.SRC / "mc_mjlab" / "robots" / "robots_registry.py"
RB = TASKS / "residual_balance"
ZR = TASKS / "zero_residual"

# Which side of the process boundary each class lives on. Not a program
# property, so it is asserted here and every name is checked against the model.
MAIN_SIDE = (
  "McRtcResidualActionBase",
  "ControllerIoBinding",
  "ControllerPool",
  "RecoveryAuthority",
)
WORKER_SIDE = ("ControllerHost",)
EXTERNAL_PATHS = (
  ("MC_RTC_YAML_PATH", "controller and robot selection"),
  ("robots.mc_mujoco_assets.MC_MUJOCO_SHARE_DIR", "meshes, MJCF and PD gains"),
  (
    "tasks.residual_balance.residual_balance_env_cfg.RECOVERY_DETECTOR_PATH",
    "the calibrated recovery gate",
  ),
)


def wrap(text: str) -> str:
  """Reflow prose paragraphs at 80 columns, leaving fences, tables and lists alone."""
  out: list[str] = []
  buffer: list[str] = []
  fenced = False

  def flush() -> None:
    if buffer:
      out.extend(textwrap.wrap(" ".join(buffer), width=80, break_long_words=False))
      buffer.clear()

  for line in text.splitlines():
    if line.startswith("```"):
      flush()
      fenced = not fenced
      out.append(line)
    elif fenced or not line or line[0] in "|-#<" or line.startswith("  "):
      flush()
      out.append(line)
    else:
      buffer.append(line.strip())
  flush()
  return "\n".join(out)


def fence(kind: str, body: str) -> str:
  """One fenced block."""
  return f"```{kind}\n{body.strip()}\n```"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
  """A markdown pipe table."""
  lines = [
    "| " + " | ".join(header) + " |",
    "| " + " | ".join("---" for _ in header) + " |",
  ]
  lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
  return "\n".join(lines)


def node_id(name: str) -> str:
  """A mermaid-safe identifier for an arbitrary dotted name."""
  return re.sub(r"[^A-Za-z0-9]", "_", name)


def require(names: tuple[str, ...]) -> None:
  """Fail generation if an asserted class name no longer exists."""
  known = {path.rsplit(".", 1)[-1] for path in ex.classes()}
  missing = sorted(set(names) - known)
  if missing:
    raise SystemExit(f"asserted classes no longer exist: {', '.join(missing)}")


# --- builders ---------------------------------------------------------------


def module_census() -> str:
  """Count modules and classes per package."""
  counts: dict[str, int] = {}
  for module in ex.internal_modules():
    counts[ex.bucket(module)] = counts.get(ex.bucket(module), 0) + 1
  rows = [(f"`{name}`", str(count)) for name, count in sorted(counts.items())]
  rows.append(("**total**", f"**{len(ex.internal_modules())}**"))
  return (
    f"{len(ex.internal_modules())} modules and {len(ex.classes())} classes, "
    f"{ex.raw_edge_count()} internal import edges before collapsing.\n\n"
    + table(("Package", "Modules"), rows)
  )


def package_graph() -> str:
  """Draw the collapsed import graph."""
  edges = ex.package_edges()
  names = sorted(set(edges) | {t for targets in edges.values() for t in targets})
  lines = ["flowchart LR"]
  for name in names:
    lines.append(f'  {node_id(name)}["{name}"]')
  for source in sorted(edges):
    for sink in sorted(edges[source]):
      lines.append(f"  {node_id(source)} --> {node_id(sink)}")
  return fence("mermaid", "\n".join(lines))


def class_diagram() -> str:
  """Emit pyreverse's own mermaid for the coupling package."""
  with tempfile.TemporaryDirectory() as tmp:
    return fence("mermaid", ex.class_diagram("actions", Path(tmp)))


def binding_imports() -> str:
  """Which modules reach mc_rtc, and how the import is written."""
  rows = []
  for module, names in sorted(ex.binding_importers().items()):
    path = ex.SRC / Path(*module.split(".")).with_suffix(".py")
    style = ex.import_style(path, names[0])
    rows.append((f"`{module}`", ", ".join(f"`{n}`" for n in names), style))
  return table(("Module", "Imports", "Written as"), rows)


def entry_points() -> str:
  """The declared entry points and the pinned interpreter."""
  rows = [(f"`{k}`", f"`{v}`") for k, v in sorted(ex.entry_points().items())]
  return table(("Entry point", "Target"), rows) + (
    f"\n\nThe bindings are interpreter-specific, so `requires-python` pins "
    f"`{ex.requires_python()}` and moves with whatever the workspace builds for."
  )


def context_diagram() -> str:
  """Draw the repo against mjlab and the workspace outside it."""
  lines = [
    "flowchart TB",
    '  subgraph cli["command line"]',
    '    T["train"]',
    '    P["play"]',
    '    L["list-envs"]',
    "  end",
    '  subgraph mjlab["mjlab and rsl_rl"]',
    '    IMP["entry-point loader"]',
    '    REG["mjlab.tasks.registry"]',
    "  end",
    '  subgraph repo["this repo"]',
  ]
  for target in sorted(ex.entry_points().values()):
    lines.append(f'    EP["{target}"]')
  for package in ("actions", "robots", "tasks"):
    lines.append(f'    {package.upper()}["{package}/"]')
  for dotted, _ in EXTERNAL_PATHS:
    short = dotted.rsplit(".", 1)[-1]
    lines.append(f'    {node_id(short)}[("{short}")]')
  lines += ["  end", '  subgraph outside["outside the repo"]']
  for names in sorted(ex.binding_importers().values()):
    for name in names:
      lines.append(f'    {node_id(name)}["{name}"]')
  lines += [
    '    WS["sourced ROS workspace"]',
    "  end",
    "  T --> IMP",
    "  P --> IMP",
    "  L --> IMP",
    '  IMP -.->|"exceptions become a warning line"| EP',
    "  EP --> TASKS",
    "  TASKS --> REG",
    "  TASKS --> ACTIONS",
    "  TASKS --> ROBOTS",
  ]
  for name in sorted({n for names in ex.binding_importers().values() for n in names}):
    lines.append(f"  WS --> {node_id(name)}")
  for module, names in sorted(ex.binding_importers().items()):
    target = "ACTIONS" if ".actions." in module else "ROBOTS"
    for name in names:
      lines.append(f"  {node_id(name)} ==> {target}")
  for dotted, _ in EXTERNAL_PATHS:
    lines.append(f"  {node_id(dotted.rsplit('.', 1)[-1])} --> TASKS")
  return fence("mermaid", "\n".join(lines))


def external_inputs() -> str:
  """Paths outside the repo, as the source writes them."""
  rows = [
    (f"`{dotted.rsplit('.', 1)[-1]}`", f"`{ex.attribute(dotted)}`", purpose)
    for dotted, purpose in EXTERNAL_PATHS
  ]
  return table(("Constant", "Value as written", "Supplies"), rows)


def topology_diagram() -> str:
  """Draw the two sides and everything that crosses between them."""
  require(MAIN_SIDE + WORKER_SIDE)
  protocol = ex.pipe_protocol(HOST, POOL)
  channels = ", ".join(
    ex.dict_keys("actions.mc_rtc_controller_host.MBC_ATTR_BY_CHANNEL")
  )
  vectors = ", ".join(ex.dict_keys("actions.mc_rtc_controller_host.VECTOR_OUTPUTS"))
  lines = ["flowchart LR", '  subgraph main["main process"]']
  lines += [f'    {node_id(n)}["{n}"]' for n in MAIN_SIDE]
  lines += [
    "  end",
    '  IN[("in_np")]',
    '  OUT[("out_np")]',
    '  subgraph workers["worker processes"]',
    '    WM["worker_main"]',
  ]
  lines += [f'    {node_id(n)}["{n}"]' for n in WORKER_SIDE]
  lines += [
    '    MC["MCGlobalController, one per env"]',
    "  end",
    "  McRtcResidualActionBase --> ControllerIoBinding",
    "  McRtcResidualActionBase --> ControllerPool",
    "  McRtcResidualActionBase --> RecoveryAuthority",
    '  ControllerIoBinding ==>|"encoders, root, IMU, wrenches"| IN',
    "  IN ==> ControllerHost",
    f'  ControllerPool -.->|"pipe: {", ".join(protocol["sent"])}"| WM',
    f'  WM -.->|"pipe: {", ".join(protocol["replies"])}"| ControllerPool',
    "  WM --> ControllerHost",
    "  ControllerHost --> MC",
    "  MC --> ControllerHost",
    f'  ControllerHost ==>|"{channels} + status + {vectors}"| OUT',
    "  OUT ==> ControllerIoBinding",
  ]
  return fence("mermaid", "\n".join(lines))


def pipe_protocol() -> str:
  """Both ends of the worker command vocabulary, cross-checked."""
  protocol = ex.pipe_protocol(HOST, POOL)
  if protocol["handled"] != protocol["sent"]:
    raise SystemExit(
      f"pipe protocol disagrees: worker handles {protocol['handled']}, "
      f"pool sends {protocol['sent']}"
    )
  rows = [(f"`{c}`", "worker handles it, pool sends it") for c in protocol["handled"]]
  rows += [(f"`{r}`", "reply tag") for r in protocol["replies"]]
  return table(("Word", "Role"), rows)


def io_layout() -> str:
  """The symbolic column map, resolved in dependency order."""
  columns = ex.io_layout_columns(HOST)
  out_side = {"status_off", "vector_off", "scalar_command_output_off", "out_width"}
  rows = [
    (f"`{c.name}`", "output" if c.name in out_side else "input", f"`{c.expr}`")
    for c in sorted(columns, key=lambda c: (c.name in out_side, columns.index(c)))
  ]
  modes = []
  for path, label in ((POSITION, "position"), (TORQUE, "torque")):
    channels = ex.class_channels(path)
    modes.append((label, f"`{channels}`"))
  return (
    table(("Offset", "Block", "Expression"), rows)
    + "\n\n"
    + table(("Control mode", "output_channels"), modes)
    + "\n\n"
    + md.SYMBOLIC
  )


def status_column() -> str:
  """The three values of the status column, and who writes each."""
  names = ex.module_constants("actions.mc_rtc_controller_host", "STATUS_")
  rows = [
    (f"`{name}`", f"`{ex.attribute(f'actions.mc_rtc_controller_host.{name}')}`")
    for name in names
  ]
  return table(("Constant", "Value"), rows)


def failure_diagram() -> str:
  """Draw the two failure paths that must not be scored alike."""
  lines = [
    "flowchart LR",
    '  QP["run() returned false"] --> SQP["STATUS_QP_FAILED"]',
    '  SQP --> LQP["controller_failed"]',
    '  LQP --> TQP["a normal fall: pays termination_penalty"]',
    '  DEAD["worker died or timed out"] --> REV["_revive_worker"]',
    '  REV --> MARK["_mark_failed"]',
    '  MARK --> SWF["STATUS_WORKER_FAILED"]',
    '  SWF --> LWF["controller_worker_failed"]',
    '  LWF --> TWF["time_out, so the value bootstraps"]',
  ]
  return fence("mermaid", "\n".join(lines))


def step_sequence() -> str:
  """Draw the control step in the order the source runs it."""
  beats = ex.call_sequence(ACTION, "apply_actions", ("_io", "_pool"))
  lines = [
    "sequenceDiagram",
    "  participant A as apply_actions",
    "  participant B as ControllerIoBinding",
    "  participant M as in_np and out_np",
    "  participant P as ControllerPool",
    "  participant S as MuJoCo actuators",
    "  Note over A: first substep of the period",
  ]
  wrote = False
  for beat in beats:
    if beat.startswith("_io."):
      lines.append(f"  A->>B: {beat.split('.', 1)[1]}")
      if not wrote:
        lines.append("  B->>M: fills this env's input row")
        wrote = True
    elif beat.startswith("_pool."):
      lines.append(f"  A->>P: {beat.split('.', 1)[1]}")
    elif beat == "_collect_controller_output":
      lines.append(f"  A->>P: {beat}")
      lines.append("  P-->>A: the step dispatched one period ago")
    elif beat == "_apply_control":
      lines.append(f"  A->>S: {beat}")
    else:
      lines.append(f"  A->>A: {beat}")
  return fence("mermaid", "\n".join(lines)) + (
    "\n\nRead top to bottom, that is the source order of "
    f"`{ACTION.relative_to(ex.ROOT)}`'s `apply_actions`: collecting the previous "
    "period's solve happens before anything refills the input block."
  )


def rate_stack() -> str:
  """The three rates, from the literals in the env cfg."""
  cfg = RB / "residual_balance_env_cfg.py"
  env = ex.literal_kwargs(cfg, "ManagerBasedRlEnvCfg", ("decimation",))
  sim = ex.literal_kwargs(cfg, "MujocoCfg", ("timestep",))
  frameskip_value = ex.kwarg_literal(cfg, "frameskip")
  defaults = ex.param_defaults(cfg, "_make_env_cfg")
  decimation = int(env["decimation"])
  frameskip = int(frameskip_value or "1")
  timestep = float(sim["timestep"])
  rows = [
    ("sim", f"`timestep={sim['timestep']}`", f"{1 / timestep:.0f} Hz"),
    (
      "controller",
      f"`frameskip={frameskip_value}`",
      f"{1 / (timestep * frameskip):.0f} Hz",
    ),
    (
      "policy",
      f"`decimation={env['decimation']}`",
      f"{1 / (timestep * decimation):.0f} Hz",
    ),
  ]
  return table(("Rate", "Set by", "Runs at"), rows) + (
    f"\n\nSo one policy step spans {decimation // frameskip} controller periods, "
    f"and the episode runs `episode_length_s="
    f"{defaults.get('episode_length_s', '?')}` by default."
  )


def reset_sequence() -> str:
  """The reset path, in source order."""
  beats = ex.call_sequence(ACTION, "reset", ("_io", "_pool"))
  rows = [(str(index + 1), f"`{beat}`") for index, beat in enumerate(beats)]
  return table(("Step", "Call"), rows)


def wiring_diagram() -> str:
  """Draw yaml to ids to cfgs to managers to the runner."""
  registrations = ex.registrations(RB / "__init__.py")
  zero = ex.registrations(ZR / "__init__.py")
  runners = sorted({r.runner_cls for r in registrations if r.runner_cls})
  lines = [
    "flowchart LR",
    '  YAML[("etc/mc_rtc.yaml")] --> CFG["utils/mc_rtc_config"]',
    '  CFG --> NAME["get_task_name"]',
    f'  NAME --> RB["residual_balance: {len(registrations)} ids"]',
    f'  NAME --> ZR["zero_residual: {len(zero)} ids, play only"]',
    '  RB --> REG["register_mjlab_task"]',
    "  ZR --> REG",
    '  RB --> MAKE["_make_env_cfg"]',
    '  MAKE --> MGR["manager terms from tasks/mdp.py"]',
  ]
  for runner in runners:
    lines.append(f'  RB --> {node_id(runner)}["{runner}"]')
  for dotted in ex.dotted_class_paths(RB / "residual_balance_ppo_cfg.py"):
    name = dotted.rsplit(":", 1)[-1]
    lines.append(f'  RB -.->|"by dotted string"| {node_id(name)}["{name}"]')
  return fence("mermaid", "\n".join(lines))


def task_ids() -> str:
  """The ids as the repo's own naming helper builds them."""
  from mc_mjlab.utils.mc_rtc_config import get_controller_name, get_main_robot_name
  from mc_mjlab.utils.task_naming import get_task_name

  yaml = ex.ROOT / "etc" / "mc_rtc.yaml"
  ids = []
  for directory, path in (("residual_balance", RB), ("zero_residual", ZR)):
    for suffix in sorted(suffixes(path)):
      ids.append((f"`{get_task_name(directory, suffix)}`", directory))
  head = (
    f"Built from `MainRobot: {get_main_robot_name(yaml)}` and "
    f"`Enabled: {get_controller_name(yaml)}`.\n\n"
  )
  return head + table(("Task id", "Package"), ids)


def suffixes(package: Path) -> set[str]:
  """Task-id suffixes a package registers, from its get_task_name calls."""
  found: set[str] = set()
  for node in ast.walk(ex.tree(package / "__init__.py")):
    if isinstance(node, ast.Call) and ast.unparse(node.func) == "get_task_name":
      if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
        value = node.args[1].value
        if isinstance(value, str):
          found.add(value)
  return found


def registration_table() -> str:
  """Every register_mjlab_task call, as written."""
  rows = []
  for package, path in (("residual_balance", RB), ("zero_residual", ZR)):
    for reg in ex.registrations(path / "__init__.py"):
      rows.append(
        (
          package,
          f"`{reg.task_id}`",
          f"`{reg.env_cfg}`",
          f"`{reg.runner_cls}`" if reg.runner_cls else "mjlab default",
        )
      )
  return table(("Package", "Id constant", "Env cfg", "Runner"), rows)


def term_census() -> str:
  """Manager terms by constructor, plus the ones bound under two managers."""
  rows = []
  for package, path in (
    ("residual_balance", RB / "residual_balance_env_cfg.py"),
    ("zero_residual", ZR / "zero_residual_env_cfg.py"),
  ):
    counts: dict[str, int] = {}
    for binding in ex.term_bindings(path):
      counts[binding.kind] = counts.get(binding.kind, 0) + 1
    for kind in ex.TERM_KINDS:
      if counts.get(kind):
        rows.append((package, f"`{kind}`", str(counts[kind])))
  dual = ex.dual_role_terms(RB / "residual_balance_env_cfg.py")
  tail = ""
  if dual:
    listed = ", ".join(f"`{name}`" for name in sorted(dual))
    tail = (
      f"\n\nBound under more than one manager, and therefore drawn twice: {listed}."
    )
  return table(("Package", "Constructor", "Constructions"), rows) + tail


def injected_classes() -> str:
  """Classes rsl_rl receives as strings, which no import edge records."""
  rows = [
    (f"`{dotted}`", "no import edge exists")
    for dotted in ex.dotted_class_paths(RB / "residual_balance_ppo_cfg.py")
  ]
  return table(("Dotted path", "Visibility"), rows)


def robot_registry() -> str:
  """The registry keys and the fields each spec carries."""
  keys = ex.dict_keys("robots.robots_registry.ROBOTS")
  fields = ex.class_attributes("robots.robots_registry.RobotSpec")
  rows = [
    (f"`{key}`", "yes" if (ex.SRC / "mc_mjlab" / "robots" / key).is_dir() else "no")
    for key in keys
  ]
  return (
    table(("MainRobot key", "Directory of the same name"), rows)
    + "\n\n`RobotSpec` carries "
    + ", ".join(f"`{name}`" for name in fields)
    + "."
  )


def asset_diagram() -> str:
  """Draw MainRobot to a loaded spec with assets, gains and sensors."""
  keys = ex.dict_keys("robots.robots_registry.ROBOTS")
  lines = [
    "flowchart TB",
    '  YAML[("etc/mc_rtc.yaml — MainRobot")] --> GET["get_main_robot_spec"]',
    f'  GET --> ROBOTS["ROBOTS: {", ".join(keys)}"]',
    '  ROBOTS --> SPEC["RobotSpec"]',
    '  SPEC --> CFGFN["get_robot_cfg"]',
    '  CFGFN --> GS["get_spec"]',
    '  GS --> ENS["ensure_assets"]',
    '  ENS --> SYM["ensure_asset_symlink"]',
    '  SYM --> SRC[("share/mc_mujoco/ROBOT")]',
    '  GS --> COL["collision_configuration"]',
    '  GS --> SENS["add_locomotion_sensors"]',
    '  GS --> PDA["get_pd_actuator_cfgs"]',
    '  CFGFN --> PREP["prepare_cfg_for_mc_rtc"]',
    '  SRC --> GAINS[("PDgains_sim.dat")]',
    '  META["await_ready returns ref_joint_order"] ==> APPLY["apply_reference_pd_gains"]',
    "  GAINS ==> APPLY",
    '  PDA -.->|"overwritten at action-term init"| APPLY',
  ]
  return fence("mermaid", "\n".join(lines))


def gain_ordering() -> str:
  """Where the gains file lives for each robot."""
  rows = [
    (f"`{key}`", f"`{kwargs.get('pd_gains_path', '—')}`")
    for key, kwargs in ex.dict_call_kwargs("robots.robots_registry.ROBOTS").items()
  ]
  return table(("Robot", "pd_gains_path, as written"), rows)


BUILDERS = {
  name: value
  for name, value in list(globals().items())
  if callable(value) and name in md.SECTION_TITLES
}


# --- rendering --------------------------------------------------------------


def render(layer: md.Layer) -> str:
  """One generated document."""
  parts = [md.BANNER, "", f"# {layer.title}", "", layer.claim, ""]
  if layer.intro:
    parts += [layer.intro, ""]
  for section in layer.sections:
    parts += [f"## {md.SECTION_TITLES[section]}", ""]
    prose = md.BUILDER_PROSE.get(section, "")
    if prose:
      parts += [prose, ""]
    parts += [BUILDERS[section](), ""]
  if layer.notes:
    parts += ["## Notes", ""]
    for head, body in layer.notes:
      parts += [f"**{head}.** {body}", ""]
  if layer.hazard:
    parts += ["## What bites", "", layer.hazard, ""]
  if layer.links:
    parts += ["## Where the numbers live", ""]
    for text, target in layer.links:
      parts.append(f"- [{text}]({target})")
    parts.append("")
  return wrap("\n".join(parts)).rstrip() + "\n"


def render_index() -> str:
  """The directory README."""
  rows = [(f"[{layer.title}]({layer.slug}.md)", layer.summary) for layer in md.LAYERS]
  body = "\n".join(
    [
      md.BANNER,
      "",
      "# How the system fits together",
      "",
      md.INDEX_INTRO,
      table(("View", "Answers"), rows),
      "",
      md.INDEX_HOW,
      "## What is not here",
      "",
      md.NOT_IN_SOURCE,
    ]
  )
  return wrap(body).rstrip() + "\n"


def build() -> dict[Path, str]:
  """Every generated file, in memory."""
  files = {OUT / "README.md": render_index()}
  for layer in md.LAYERS:
    files[OUT / f"{layer.slug}.md"] = render(layer)
  check_links(files)
  return files


def check_links(files: dict[Path, str]) -> None:
  """Resolve every cross-link so a renamed heading fails the build."""
  anchors = ex.doc_anchors()
  generated = {str(path.resolve()): text for path, text in files.items()}
  for name, text in generated.items():
    anchors[name] = heading_anchors(text)
  pattern = re.compile(r"\]\(([\w./-]+\.md)(?:#([\w-]+))?\)")
  for path, text in files.items():
    for target, anchor in pattern.findall(text):
      destination = str((path.parent / target).resolve())
      if destination not in generated and not Path(destination).exists():
        raise SystemExit(f"{path.name}: dead link {target}")
      if anchor and anchor not in anchors.get(destination, set()):
        raise SystemExit(f"{path.name}: dead anchor {target}#{anchor}")


def heading_anchors(text: str) -> set[str]:
  """The anchors a generated document defines, slugged as GitHub does."""
  found = set()
  for line in text.splitlines():
    if line.startswith("#"):
      head = line.lstrip("#").strip()
      found.add(re.sub(r"[^\w\- ]", "", head).strip().lower().replace(" ", "-"))
  return found


def listed(names) -> str:
  """A comma-separated backticked list."""
  return ", ".join(f"`{name}`" for name in sorted(names))


def live_inventory() -> str:
  """What only a running mjlab can report, for the machine it ran on."""
  from mjlab.tasks.registry import list_tasks, load_env_cfg

  ours = sorted(task for task in list_tasks() if task.startswith("Mc-Mjlab-"))
  parts = [
    md.BANNER,
    "",
    "# Live inventory",
    "",
    md.LIVE_INTRO,
    f"## Registered task ids ({len(ours)})",
    "",
    table(("Task id",), [(f"`{task}`",) for task in ours]),
    "",
  ]
  if ours:
    task = ours[0]
    cfg = load_env_cfg(task)
    parts += [f"## Resolved manager terms of `{task}`", "", md.LIVE_TERMS]
    rows = []
    for manager in ("observations", "rewards", "terminations", "events", "metrics"):
      group = getattr(cfg, manager, None) or {}
      for name, value in sorted(group.items()):
        terms = getattr(value, "terms", None)
        if isinstance(terms, dict):
          rows.append((f"{manager}/{name}", str(len(terms)), listed(terms)))
        else:
          rows.append((manager, "", f"`{name}`"))
    merged: dict[str, list[str]] = {}
    for label, count, text in rows:
      merged.setdefault(f"{label}\u0000{count}", []).append(text)
    parts += [
      table(
        ("Manager", "Count", "Terms"),
        [
          (
            key.split("\u0000")[0],
            key.split("\u0000")[1] or str(len(values)),
            ", ".join(values),
          )
          for key, values in merged.items()
        ],
      ),
      "",
    ]
  return wrap("\n".join(parts)).rstrip() + "\n"


def rate_strip() -> str:
  """A to-scale tick strip of the three rates, drawn from the extracted values."""
  cfg = RB / "residual_balance_env_cfg.py"
  decimation = int(
    ex.literal_kwargs(cfg, "ManagerBasedRlEnvCfg", ("decimation",))["decimation"]
  )
  frameskip = int(ex.kwarg_literal(cfg, "frameskip") or "1")
  timestep = float(ex.literal_kwargs(cfg, "MujocoCfg", ("timestep",))["timestep"])
  x0, x1 = 112.0, 640.0
  span = x1 - x0
  rows = (
    (f"{1 / timestep:.0f} Hz", "sim", decimation),
    (f"{1 / (timestep * frameskip):.0f} Hz", "controller", decimation // frameskip),
    (f"{1 / (timestep * decimation):.0f} Hz", "policy", 1),
  )
  parts = []
  for index, (label, sub, ticks) in enumerate(rows):
    top = 16 + index * 34
    parts.append(
      f'<text x="0" y="{top + 4}" class="sv-l">{label}</text>'
      f'<text x="66" y="{top + 4}" class="sv-s">{sub}</text>'
      f'<line x1="{x0}" y1="{top}" x2="{x1}" y2="{top}" class="sv-a"/>'
    )
    for k in range(ticks + 1):
      px = x0 + span * k / ticks
      height = 8 if k in (0, ticks) else 4.5
      parts.append(
        f'<line x1="{px:.1f}" y1="{top - height:.1f}" '
        f'x2="{px:.1f}" y2="{top + height:.1f}" class="sv-t"/>'
      )
  parts.append(f'<line x1="{x0}" y1="4" x2="{x0}" y2="100" class="sv-b"/>')
  parts.append(f'<line x1="{x1}" y1="4" x2="{x1}" y2="100" class="sv-b"/>')
  parts.append(
    f'<text x="{x1}" y="116" class="sv-s" text-anchor="end">one policy step '
    f"&#183; {decimation} sim steps &#183; {decimation // frameskip} controller "
    f"periods</text>"
  )
  return "\n        ".join(parts)


def page_html(files: dict[Path, str]) -> str:
  """Render the generated markdown as one self-contained page."""
  import markdown as markdown_lib

  converter = markdown_lib.Markdown(extensions=["tables", "fenced_code"])
  rail, body = [], []
  for index, layer in enumerate(md.LAYERS):
    converter.reset()
    text = files[OUT / f"{layer.slug}.md"]
    kept = [row for row in text.splitlines() if not row.startswith("<!--")]
    text = "\n".join(kept)
    html = converter.convert(text)
    html = re.sub(
      r'<pre><code class="language-mermaid">(.*?)</code></pre>',
      r'<pre class="mermaid">\1</pre>',
      html,
      flags=re.S,
    )
    for level in (3, 2, 1):
      html = html.replace(f"<h{level}>", f"<h{level + 1}>")
      html = html.replace(f"</h{level}>", f"</h{level + 1}>")
    html = html.replace("<table>", '<div class="tablewrap"><table>')
    html = html.replace("</table>", "</table></div>")
    rail.append(
      f'<li><a href="#{layer.slug}"><span>L{index}</span>'
      f"<span>{layer.title}</span></a></li>"
    )
    body.append(
      f'<section class="layer" id="{layer.slug}">'
      f'<p class="tag">L{index}</p>{html}</section>'
    )
  return "\n".join(
    [
      "<title>mc_mjlab Architecture</title>",
      '<meta name="viewport" content="width=device-width, initial-scale=1">',
      '<link rel="preconnect" href="https://fonts.googleapis.com">',
      '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
      '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      "family=Archivo:wght@600;700&family=IBM+Plex+Mono:wght@400;500&"
      'family=IBM+Plex+Sans:wght@400;500;600&display=swap">',
      f"<style>{md.PAGE_CSS}</style>",
      '<header class="masthead"><div class="wrap"><div>',
      '<p class="eyebrow">generated from the source</p>',
      "<h1>mc_mjlab Architecture</h1>",
      f'<p class="lede">{md.LAYERS[0].claim}</p>',
      "</div>",
      '<figure class="strip">',
      '<svg viewBox="0 -6 660 126" role="img" aria-label="Three nested rates over '
      "one policy step: the sim ticks many times, the controller fewer, the policy "
      'once.">',
      f"        {rate_strip()}",
      "</svg>",
      "<figcaption>The rate stack, to scale, from the literals in the env "
      "cfg.</figcaption></figure>",
      "</div></header>",
      '<div class="wrap shell">',
      '<nav class="rail" aria-label="Views"><h2>Views</h2><ol>',
      "".join(rail),
      "</ol>",
      '<p class="source">Source of truth: <span class="mono">'
      "scripts/generate_architecture_docs.py</span>. This page and "
      '<span class="mono">docs/architecture/</span> are both its output.</p>',
      "</nav><main>",
      "".join(body),
      "</main></div>",
      '<footer><div class="wrap"><p>Every fact here is read out of the source at '
      'generation time by griffe, grimp, pyreverse and <span class="mono">ast'
      '</span>. Running the generator with <span class="mono">--check</span> '
      "turns any drift between these pages and the code into a build failure.</p>"
      "<p>Measurements deliberately live elsewhere. A diagram states structure; it "
      "never restates a number.</p></div></footer>",
    ]
  )


def main() -> int:
  """Write, or check, the generated architecture docs."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--check", action="store_true", help="fail if files differ")
  parser.add_argument(
    "--live", action="store_true", help="also write live-inventory.md"
  )
  parser.add_argument("--html", type=Path, help="also render one shareable page")
  args = parser.parse_args()

  files = build()
  if args.check:
    problems = []
    for path, text in sorted(files.items()):
      current = path.read_text() if path.exists() else ""
      if current != text:
        diff = difflib.unified_diff(
          current.splitlines(), text.splitlines(), "on disk", "generated", lineterm=""
        )
        problems.append(f"{path.relative_to(ex.ROOT)}\n" + "\n".join(list(diff)[:40]))
    if problems:
      print("\n\n".join(problems))
      print("\narchitecture docs are stale -- rerun without --check")
      return 1
    print(f"architecture docs match the source ({len(files)} files)")
    return 0

  OUT.mkdir(parents=True, exist_ok=True)
  for path, text in sorted(files.items()):
    path.write_text(text)
  if args.live:
    (OUT / "live-inventory.md").write_text(live_inventory())
  if args.html:
    args.html.write_text(page_html(files))
    print(f"rendered {args.html}")
  print(
    f"wrote {len(files) + (1 if args.live else 0)} files to {OUT.relative_to(ex.ROOT)}"
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
