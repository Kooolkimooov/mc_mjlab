#!/usr/bin/env python3
"""Write docs/architecture from the source; --check turns drift into a failure."""

import argparse
import ast
import difflib
import re
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# Sibling script, resolved by the interpreter's script directory at runtime.
import architecture_extract as ex  # ty: ignore[unresolved-import]

OUT = ex.ROOT / "docs" / "architecture"
CSS = Path(__file__).with_name("architecture_page.css")
ACTIONS = ex.SRC / "mc_mjlab" / "actions"
ROBOTS = ex.SRC / "mc_mjlab" / "robots"
TASKS = ex.SRC / "mc_mjlab" / "tasks"
HOST = ACTIONS / "mc_rtc_controller_host.py"
POOL = ACTIONS / "mc_rtc_controller_pool.py"
BINDING = ACTIONS / "mc_rtc_controller_io_binding.py"
ACTION = ACTIONS / "mc_rtc_residual_action.py"
POSITION = ACTIONS / "mc_rtc_residual_joint_position_actions.py"
TORQUE = ACTIONS / "mc_rtc_residual_joint_torque_actions.py"
REGISTRY = ROBOTS / "robots_registry.py"
RB = TASKS / "residual_balance"
ZR = TASKS / "zero_residual"
RB_CFG = RB / "residual_balance_env_cfg.py"
ZR_CFG = ZR / "zero_residual_env_cfg.py"
PPO_CFG = RB / "residual_balance_ppo_cfg.py"

BANNER = (
  "<!-- Generated from the source. Do not edit. -->\n"
  "<!-- Rebuild: uv run python scripts/generate_architecture_docs.py -->"
)
PATH_MODULES = (
  "robots.mc_mujoco_assets",
  "tasks.residual_balance.residual_balance_env_cfg",
)


@dataclass(frozen=True)
class Section:
  """One generated block: its builder, the tool behind it, the files it reads."""

  name: str
  tool: str
  sources: tuple[Path, ...]
  args: tuple = ()


@dataclass(frozen=True)
class Page:
  """One generated document."""

  slug: str
  sections: tuple[Section, ...]


def block(name: str, tool: str, *sources: Path, args: tuple = ()) -> Section:
  """Declare a section, the files it reads, and what to build it for."""
  return Section(name=name, tool=tool, sources=sources, args=args)


def env_cfg_of(package: Path) -> Path | None:
  """The env cfg module of one task package."""
  return next(iter(sorted(package.glob("*_env_cfg.py"))), None)


def module_of(path: Path) -> str:
  """The griffe path of a source file, relative to the package root."""
  parts = path.relative_to(ex.SRC / ex.PKG).with_suffix("").parts
  return ".".join(parts)


def task_page(package: Path) -> Page:
  """One page per task sub-package, since each wires its own managers."""
  sources = tuple(sorted(package.rglob("*.py")))
  cfg = env_cfg_of(package)
  sections = [
    block("task_ids", "ast, mc_mjlab.utils", *sources, args=(package,)),
    block("registration_table", "ast", *sources, args=(package,)),
  ]
  if cfg is not None:
    sections += [
      block("observation_terms", "ast, griffe", cfg, args=(package,)),
      block("reward_terms", "ast, griffe", cfg, args=(package,)),
      block("termination_terms", "ast", cfg, args=(package,)),
      block("event_terms", "ast", cfg, args=(package,)),
      block("curriculum_terms", "ast", cfg, args=(package,)),
      block("metric_terms", "ast", cfg, args=(package,)),
      block("group_composition", "ast", cfg, args=(package,)),
    ]
  sections.append(block("injected_classes", "ast", *sources, args=(package,)))
  return Page(f"task-{package.name.replace('_', '-')}", tuple(sections))


PAGES: tuple[Page, ...] = (
  Page(
    "code-map",
    (
      block("module_census", "grimp, griffe"),
      block("package_graph", "grimp"),
      block("class_diagram", "pyreverse", *sorted(ACTIONS.glob("*.py"))),
      block("binding_imports", "grimp, ast"),
    ),
  ),
  Page(
    "system-context",
    (
      block("entry_points", "tomllib", ex.ROOT / "pyproject.toml"),
      block("external_inputs", "griffe"),
      block("context_diagram", "grimp, griffe, tomllib"),
    ),
  ),
  Page(
    "process-topology",
    (
      block("collaboration_diagram", "ast", ACTION, POOL, HOST),
      block("pipe_protocol", "ast", HOST, POOL),
      block("io_layout", "ast", HOST, POSITION, TORQUE),
      block("layout_inputs", "ast", HOST),
      block("status_column", "griffe, ast", HOST, POOL, BINDING, ACTION),
    ),
  ),
  Page(
    "control-step",
    (
      block("step_sequence", "ast", ACTION),
      block("rate_stack", "ast", RB_CFG),
      block("reset_sequence", "ast", ACTION),
    ),
  ),
  Page(
    "robot-assets",
    (
      block("robot_registry", "griffe", REGISTRY),
      block("asset_call_graph", "ast", *sorted(ROBOTS.glob("*.py"))),
      block("gain_paths", "griffe", REGISTRY),
    ),
  ),
) + tuple(task_page(package) for package in ex.task_packages())


def fence(kind: str, body: str) -> str:
  """One fenced block."""
  return f"```{kind}\n{body.strip()}\n```"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
  """A markdown pipe table."""
  lines = [
    "| " + " | ".join(header) + " |",
    "| " + " | ".join("---" for _ in header) + " |",
  ]
  lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
  return "\n".join(lines)


def node_id(name: str) -> str:
  """A mermaid-safe identifier."""
  return re.sub(r"[^A-Za-z0-9]", "_", name)


def tick(items) -> str:
  """A comma-separated backticked list."""
  return ", ".join(f"`{item}`" for item in items)


# --- builders ---------------------------------------------------------------


def module_census() -> str:
  """Modules per package, with the class and edge totals."""
  counts: dict[str, int] = {}
  for module in ex.internal_modules():
    counts[ex.bucket(module)] = counts.get(ex.bucket(module), 0) + 1
  rows = [(f"`{name}`", str(count)) for name, count in sorted(counts.items())]
  rows.append(
    (
      f"**{len(ex.internal_modules())} modules**",
      f"**{len(ex.classes())} classes, {ex.raw_edge_count()} import edges**",
    )
  )
  return table(("Package", "Modules"), rows)


def package_graph() -> str:
  """Internal imports, collapsed from module to package."""
  edges = ex.package_edges()
  names = sorted(set(edges) | {t for targets in edges.values() for t in targets})
  lines = ["flowchart LR"]
  lines += [f'  {node_id(name)}["{name}"]' for name in names]
  for source in sorted(edges):
    for sink in sorted(edges[source]):
      lines.append(f"  {node_id(source)} --> {node_id(sink)}")
  return fence("mermaid", "\n".join(lines))


def class_diagram() -> str:
  """pyreverse's own mermaid for the coupling package."""
  with tempfile.TemporaryDirectory() as tmp:
    return fence("mermaid", ex.class_diagram("actions", Path(tmp)))


def binding_imports() -> str:
  """Modules reaching the mc_rtc bindings, and how each import is written."""
  rows = []
  for module, names in sorted(ex.binding_importers().items()):
    path = ex.SRC / Path(*module.split(".")).with_suffix(".py")
    rows.append((f"`{module}`", tick(names), ex.import_style(path, names[0])))
  return table(("Module", "Imports", "Written as"), rows)


def entry_points() -> str:
  """Declared entry points and the pinned interpreter."""
  rows = [
    (f"`{key}`", f"`{value}`") for key, value in sorted(ex.entry_points().items())
  ]
  rows.append(("`requires-python`", f"`{ex.requires_python()}`"))
  return table(("Declares", "Value"), rows)


def external_inputs() -> str:
  """Module constants whose value is built from a filesystem path."""
  rows = [
    (f"`{name}`", f"`{module}`", f"`{value}`")
    for name, module, value in ex.path_constants(PATH_MODULES)
  ]
  return table(("Constant", "Module", "Value as written"), rows)


def context_diagram() -> str:
  """Entry point, packages, path constants and bindings, as declared."""
  lines = ["flowchart LR", '  subgraph outside["outside the repo"]']
  bindings = sorted({n for names in ex.binding_importers().values() for n in names})
  lines += [f'    {node_id(name)}["{name}"]' for name in bindings]
  for name, _, _ in ex.path_constants(PATH_MODULES):
    lines.append(f'    {node_id(name)}[("{name}")]')
  lines += ["  end", '  subgraph mjlab["mjlab"]']
  for group in sorted(ex.entry_points()):
    lines.append(f'    {node_id(group)}["{group}"]')
  lines += ["  end", '  subgraph repo["mc_mjlab"]']
  packages = sorted({ex.bucket(module) for module in ex.internal_modules()})
  lines += [f'    {node_id(name)}["{name}"]' for name in packages]
  lines.append("  end")
  for group, target in sorted(ex.entry_points().items()):
    lines.append(f"  {node_id(group)} --> {node_id(target)}")
  for module, names in sorted(ex.binding_importers().items()):
    for name in names:
      lines.append(f"  {node_id(name)} ==> {node_id(ex.bucket(module))}")
  for name, module, _ in ex.path_constants(PATH_MODULES):
    owner = ex.bucket(f"{ex.PKG}.{module}")
    lines.append(f"  {node_id(name)} --> {node_id(owner)}")
  return fence("mermaid", "\n".join(lines))


def collaboration_diagram() -> str:
  """Methods each class calls on the collaborators it constructs."""
  owners = (
    (ACTION, "McRtcResidualActionBase"),
    (POOL, "ControllerPool"),
    (HOST, "ControllerHost"),
  )
  worker_side = {
    name.rsplit(".", 1)[-1]
    for name in ex.classes()
    if name.rsplit(".", 1)[0] in ex.binding_importers()
  }
  grouped: dict[tuple[str, str], list[str]] = {}
  for path, owner in owners:
    for callee, method in ex.collaborations(path, owner):
      grouped.setdefault((owner, callee), []).append(method)
  drawn = {name for pair in grouped for name in pair}
  lines = ["flowchart LR"]
  for name in sorted(drawn):
    side = "worker process" if name in worker_side else "main process"
    lines.append(f'  {node_id(name)}["{name}<br/>{side}"]')
  for (owner, callee), methods in sorted(grouped.items()):
    label = "<br/>".join(sorted(methods))
    lines.append(f'  {node_id(owner)} -->|"{label}"| {node_id(callee)}')
  return fence("mermaid", "\n".join(lines))


def pipe_protocol() -> str:
  """The worker command vocabulary, read from both ends and cross-checked."""
  protocol = ex.pipe_protocol(HOST, POOL)
  if protocol["handled"] != protocol["sent"]:
    raise SystemExit(
      f"pipe protocol disagrees: worker handles {protocol['handled']}, "
      f"pool sends {protocol['sent']}"
    )
  rows = [(f"`{word}`", "command", "handled and sent") for word in protocol["handled"]]
  rows += [
    (f"`{word}`", "reply tag", "sent by the worker") for word in protocol["replies"]
  ]
  return table(("Word", "Kind", "Ends"), rows)


def io_layout() -> str:
  """The offset properties, resolved against each other in dependency order."""
  columns = ex.io_layout_columns(HOST)
  out_side = {"status_off", "vector_off", "scalar_command_output_off", "out_width"}
  ordered = sorted(columns, key=lambda c: (c.name in out_side, columns.index(c)))
  rows = [
    (f"`{c.name}`", "output" if c.name in out_side else "input", f"`{c.expr}`")
    for c in ordered
  ]
  modes = [
    ("position", f"`{ex.class_channels(POSITION)}`"),
    ("torque", f"`{ex.class_channels(TORQUE)}`"),
  ]
  return (
    table(("Offset", "Block", "Expression"), rows)
    + "\n\n"
    + table(("Control mode", "`output_channels`"), modes)
  )


def layout_inputs() -> str:
  """The fields those expressions depend on, with annotation and default."""
  rows = [
    (f"`{name}`", f"`{annotation}`", f"`{default}`" if default else "none")
    for name, annotation, default in ex.layout_inputs(HOST)
  ]
  return table(("Field", "Annotation", "Default"), rows)


def status_column() -> str:
  """The status constants, and every function that names one."""
  names = ex.module_constants("actions.mc_rtc_controller_host", "STATUS_")
  values = table(
    ("Constant", "Value"),
    [
      (f"`{name}`", f"`{ex.attribute(f'actions.mc_rtc_controller_host.{name}')}`")
      for name in names
    ],
  )
  writers = table(
    ("Function", "Constant"),
    [
      (f"`{where}`", f"`{name}`")
      for where, name in ex.status_writers([HOST, POOL, BINDING, ACTION])
    ],
  )
  return values + "\n\n" + writers


def step_sequence() -> str:
  """apply_actions, in the order the source runs it."""
  return call_diagram("apply_actions")


def reset_sequence() -> str:
  """reset, in the order the source runs it."""
  return call_diagram("reset")


def call_diagram(method: str) -> str:
  """One method's calls on its collaborators, as a sequence, in source order."""
  owned = ex.attribute_types(ACTION, "McRtcResidualActionBase")
  aliases = {
    name: node_id(name)[:1].upper() + str(i)
    for i, name in enumerate(sorted(set(owned.values())))
  }
  lines = ["sequenceDiagram", f"  participant A as {method}"]
  lines += [f"  participant {alias} as {name}" for name, alias in aliases.items()]
  for beat in ex.call_sequence(ACTION, method, tuple(owned)):
    attr, _, called = beat.partition(".")
    if called and attr in owned:
      lines.append(f"  A->>{aliases[owned[attr]]}: {called}")
    else:
      lines.append(f"  A->>A: {beat}")
  return fence("mermaid", "\n".join(lines))


def rates() -> tuple[float, int, int]:
  """Timestep, frameskip and decimation, as written in the env cfg."""
  env = ex.literal_kwargs(RB_CFG, "ManagerBasedRlEnvCfg", ("decimation",))
  sim = ex.literal_kwargs(RB_CFG, "MujocoCfg", ("timestep",))
  return (
    float(sim["timestep"]),
    int(ex.kwarg_literal(RB_CFG, "frameskip") or "1"),
    int(env["decimation"]),
  )


def rate_stack() -> str:
  """The three rates, from the literals in the env cfg."""
  timestep, frameskip, decimation = rates()
  rows = [
    ("sim", f"`timestep={timestep}`", f"{1 / timestep:.0f} Hz", "1"),
    (
      "controller",
      f"`frameskip={frameskip}`",
      f"{1 / (timestep * frameskip):.0f} Hz",
      str(frameskip),
    ),
    (
      "policy",
      f"`decimation={decimation}`",
      f"{1 / (timestep * decimation):.0f} Hz",
      str(decimation),
    ),
  ]
  return table(("Rate", "Set by", "Runs at", "Sim steps per period"), rows)


def task_ids(package: Path) -> str:
  """Every id this package builds, resolved through the repo's naming helper."""
  from mc_mjlab.utils.task_naming import get_task_name

  gates = ex.conditional_imports(package / "__init__.py")
  rows = []
  for path in sorted(package.rglob("*.py")):
    for suffix, literal in ex.task_name_calls(path):
      resolved = get_task_name(package.name, suffix) if literal else suffix
      rows.append(
        (
          f"`{resolved}`",
          f"`{path.name}`",
          f"`{gates.get(path.stem, '')}`" if gates.get(path.stem) else "always",
        )
      )
  return table(("Task id", "Built in", "Registered when"), rows) if rows else ""


def registration_table(package: Path) -> str:
  """Every register_mjlab_task call in this package, as written."""
  rows = []
  for path in sorted(package.rglob("*.py")):
    for registration in ex.registrations(path):
      rows.append(
        (
          f"`{registration.task_id}`",
          f"`{registration.env_cfg}`",
          f"`{registration.runner_cls}`" if registration.runner_cls else "not given",
          f"`{path.name}`",
        )
      )
  return table(("Id", "Env cfg", "runner_cls", "Called in"), rows) if rows else ""


def term_census() -> str:
  """Manager terms by the constructor used to build them."""
  rows = []
  for package, path in (("residual_balance", RB_CFG), ("zero_residual", ZR_CFG)):
    counts: dict[str, int] = {}
    for binding in ex.term_bindings(path):
      counts[binding.kind] = counts.get(binding.kind, 0) + 1
    for kind in ex.TERM_KINDS:
      if counts.get(kind):
        rows.append((f"`{package}`", f"`{kind}`", str(counts[kind])))
  duals = table(
    ("Term", "Constructors"),
    [
      (f"`{name}`", tick(kinds))
      for name, kinds in sorted(ex.dual_role_terms(RB_CFG).items())
    ],
  )
  return table(("Package", "Constructor", "Call sites"), rows) + "\n\n" + duals


def manager_table(
  package: Path, kind: str, groups: dict[str, str] | None = None
) -> str:
  """Every term of one manager, with the arguments written at its call site."""
  cfg = env_cfg_of(package)
  if cfg is None:
    return ""
  scalars = ex.module_scalars(module_of(cfg))
  terms = [term for term in ex.manager_terms(cfg) if term.kind == kind]
  if not terms:
    return ""
  keys = {key for term in terms for key in term.params}
  columns = sorted(keys - {"params"}) + (["params"] if "params" in keys else [])
  header = (["Group"] if groups else []) + ["Term", "func"] + columns + ["When"]
  labels: dict[str, list[str]] = {}
  for name, source in (groups or {}).items():
    labels.setdefault(source, []).append(name)
  rows = []
  for term in terms:
    named = ", ".join(labels.get(term.group, [term.group]))
    row = [f"`{named}`"] if groups else []
    row += [f"`{term.key}`", f"`{term.func}`"]
    row += [
      f"`{ex.resolved(term.params[key], scalars)}`" if key in term.params else ""
      for key in columns
    ]
    row.append(f"`{term.condition}`" if term.condition else "always")
    rows.append(row)
  return table(header, rows)


def observation_terms(package: Path) -> str:
  """Observation terms, by the group each is bound into."""
  cfg = env_cfg_of(package)
  groups = ex.group_exposure(cfg) if cfg else {}
  return manager_table(package, "ObservationTermCfg", groups or None)


def reward_terms(package: Path) -> str:
  """Reward terms and their written weights."""
  return manager_table(package, "RewardTermCfg")


def termination_terms(package: Path) -> str:
  """Termination terms, including which one truncates."""
  return manager_table(package, "TerminationTermCfg")


def event_terms(package: Path) -> str:
  """Event terms and the mode each runs in."""
  return manager_table(package, "EventTermCfg")


def curriculum_terms(package: Path) -> str:
  """Curriculum terms."""
  return manager_table(package, "CurriculumTermCfg")


def metric_terms(package: Path) -> str:
  """Metrics terms and how each reduces."""
  return manager_table(package, "MetricsTermCfg")


def group_composition(package: Path) -> str:
  """How each term dict is built beyond its literal entries."""
  cfg = env_cfg_of(package)
  if cfg is None:
    return ""
  owners = {term.group for term in ex.manager_terms(cfg)}
  rows = [
    (f"`{group}`", operation, f"`{source}`")
    for group, operation, source in ex.group_composition(cfg)
    if group in owners
  ]
  return table(("Dict", "Operation", "Source"), rows) if rows else ""


def injected_classes(package: Path) -> str:
  """Classes rsl_rl receives as strings, which no import edge records."""
  rows = []
  for path in sorted(package.rglob("*.py")):
    for dotted in ex.dotted_class_paths(path):
      rows.append(
        (f"`{dotted.split(':')[0]}`", f"`{dotted.split(':')[1]}`", f"`{path.name}`")
      )
  return table(("Module", "Class", "Named in"), rows) if rows else ""


def robot_registry() -> str:
  """The registry keys, and whether each names a directory of the same name."""
  rows = [
    (f"`{key}`", "yes" if (ROBOTS / key).is_dir() else "no")
    for key in ex.dict_keys("robots.robots_registry.ROBOTS")
  ]
  fields = table(
    ("`RobotSpec` field",),
    [
      (f"`{name}`",) for name in ex.class_attributes("robots.robots_registry.RobotSpec")
    ],
  )
  return table(("`MainRobot`", "Directory of that name"), rows) + "\n\n" + fields


def asset_call_graph() -> str:
  """Which of the robot-side functions call which, by source-level name."""
  roots = {}
  paths = sorted(ROBOTS.glob("*.py")) + sorted(ROBOTS.glob("*/*_constants.py"))
  for path in paths:
    for node in ast.walk(ex.tree(path)):
      if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
        roots.setdefault(node.name, path)
  edges = ex.internal_call_graph(roots)
  lines = ["flowchart LR"]
  drawn = {name for edge in edges for name in edge}
  lines += [f'  {node_id(name)}["{name}"]' for name in sorted(drawn)]
  lines += [f"  {node_id(a)} --> {node_id(b)}" for a, b in edges]
  return fence("mermaid", "\n".join(lines))


def gain_paths() -> str:
  """The pd_gains_path each registry entry names."""
  rows = [
    (f"`{key}`", f"`{kwargs.get('pd_gains_path', '')}`")
    for key, kwargs in ex.dict_call_kwargs("robots.robots_registry.ROBOTS").items()
  ]
  return table(("Robot", "`pd_gains_path`"), rows)


BUILDERS = {
  name: value
  for name, value in list(globals().items())
  if callable(value) and name in {b.name for page in PAGES for b in page.sections}
}


# --- rendering --------------------------------------------------------------


def provenance(section: Section) -> str:
  """One line naming the tool and the files a section was read from."""
  paths = list(section.sources)
  if len(paths) > 3:
    parents = {path.parent for path in paths}
    shown = (
      f"`{ex.rel(parents.pop())}/*.py`"
      if len(parents) == 1
      else f"`{ex.rel(paths[0])}` and {len(paths) - 1} more"
    )
  else:
    shown = ", ".join(f"`{ex.rel(path)}`" for path in paths)
  return f"*{section.tool}*" + (f" &middot; {shown}" if shown else "")


def see_also(page: Page) -> str:
  """docs/ links the source files of this page already carry in their comments."""
  paths = sorted({p for section in page.sections for p in section.sources})
  links = ex.doc_links([path for path in paths if path.suffix == ".py"])
  if not links:
    return ""
  rows = [(f"[{target}](../{target})", f"`{where}`") for target, where in links]
  return "\n".join(["## see_also", "", table(("Note", "Linked from"), rows), ""])


def render(page: Page) -> str:
  """One generated document."""
  parts = [BANNER, "", f"# {page.slug}", ""]
  for section in page.sections:
    body = BUILDERS[section.name](*section.args)
    if not body:
      continue
    parts += [f"## {section.name}", "", provenance(section), "", body, ""]
  tail = see_also(page)
  if tail:
    parts.append(tail)
  return "\n".join(parts).rstrip() + "\n"


def render_index() -> str:
  """The directory index: what each page holds and what produced it."""
  pages = table(
    ("Page", "Sections"),
    [
      (f"[{page.slug}]({page.slug}.md)", tick(s.name for s in page.sections))
      for page in PAGES
    ],
  )
  tools = table(
    ("Section", "Extracted with"),
    [(f"`{s.name}`", s.tool) for page in PAGES for s in page.sections],
  )
  rebuild = fence(
    "sh",
    "uv run python scripts/generate_architecture_docs.py\n"
    "uv run python scripts/generate_architecture_docs.py --check\n"
    "uv run python scripts/generate_architecture_docs.py --live",
  )
  parts = [
    BANNER,
    "",
    "# architecture",
    "",
    "## pages",
    "",
    pages,
    "",
    "## provenance",
    "",
    tools,
    "",
    "## rebuild",
    "",
    rebuild,
  ]
  return "\n".join(parts).rstrip() + "\n"


def build() -> dict[Path, str]:
  """Every generated file, in memory."""
  files = {OUT / "README.md": render_index()}
  for page in PAGES:
    files[OUT / f"{page.slug}.md"] = render(page)
  check_links(files)
  return files


def heading_anchors(text: str) -> set[str]:
  """The anchors a document defines, slugged as GitHub does."""
  found = set()
  for line in text.splitlines():
    if line.startswith("#"):
      head = line.lstrip("#").strip()
      found.add(re.sub(r"[^\w\- ]", "", head).strip().lower().replace(" ", "-"))
      # Linked verbatim so grep finds the heading. docs/README.md
      found.add(head)
  return found


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


def live_inventory() -> str:
  """What only a running mjlab can report, for the machine it ran on."""
  from mjlab.tasks.registry import list_tasks, load_env_cfg

  ours = sorted(task for task in list_tasks() if task.startswith("Mc-Mjlab-"))
  parts = [
    BANNER,
    "",
    "# live-inventory",
    "",
    "*mjlab.tasks.registry, on a sourced workspace; excluded from --check*",
    "",
    "## registered_tasks",
    "",
    table(("Task id",), [(f"`{task}`",) for task in ours]),
    "",
  ]
  if ours:
    cfg = load_env_cfg(ours[0])
    rows = []
    for manager in ("observations", "rewards", "terminations", "events", "metrics"):
      group = getattr(cfg, manager, None) or {}
      for name, value in sorted(group.items()):
        terms = getattr(value, "terms", None)
        if isinstance(terms, dict):
          rows.append((f"`{manager}/{name}`", str(len(terms)), tick(sorted(terms))))
        else:
          rows.append((f"`{manager}`", "", f"`{name}`"))
    parts += [
      "## resolved_terms",
      "",
      f"*{ours[0]}*",
      "",
      table(("Manager", "Count", "Terms"), rows),
      "",
    ]
  return "\n".join(parts).rstrip() + "\n"


def rate_strip() -> str:
  """A to-scale tick strip of the three rates, from the extracted values."""
  timestep, frameskip, decimation = rates()
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
    f'<text x="{x1}" y="116" class="sv-s" text-anchor="end">'
    f"{decimation} sim steps &#183; {decimation // frameskip} controller periods "
    f"&#183; 1 policy step</text>"
  )
  return "\n        ".join(parts)


def page_html(files: dict[Path, str]) -> str:
  """Render the generated markdown as one self-contained page."""
  import markdown as markdown_lib

  converter = markdown_lib.Markdown(extensions=["tables", "fenced_code"])
  rail, body = [], []
  for page in PAGES:
    converter.reset()
    kept = [
      row
      for row in files[OUT / f"{page.slug}.md"].splitlines()
      if not row.startswith("<!--")
    ]
    html = converter.convert("\n".join(kept))
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
    rail.append(f'<li><a href="#{page.slug}">{page.slug}</a></li>')
    body.append(f'<section class="layer" id="{page.slug}">{html}</section>')
  return "\n".join(
    [
      "<title>mc_mjlab Architecture</title>",
      '<meta name="viewport" content="width=device-width, initial-scale=1">',
      '<link rel="preconnect" href="https://fonts.googleapis.com">',
      '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
      '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
      "family=Archivo:wght@600;700&family=IBM+Plex+Mono:wght@400;500&"
      'family=IBM+Plex+Sans:wght@400;500;600&display=swap">',
      f"<style>{CSS.read_text()}</style>",
      '<header class="masthead"><div class="wrap"><div>',
      '<p class="eyebrow">generated from the source</p>',
      "<h1>mc_mjlab Architecture</h1>",
      f'<p class="lede">{ex.project_description()}</p>',
      "</div>",
      '<figure class="strip">',
      '<svg viewBox="0 -6 660 126" role="img" aria-label="The sim, controller and '
      'policy rates over one policy step.">',
      f"        {rate_strip()}",
      "</svg>",
      "<figcaption>timestep, frameskip and decimation, to scale</figcaption></figure>",
      "</div></header>",
      '<div class="wrap shell">',
      '<nav class="rail" aria-label="Pages"><h2>pages</h2><ul>',
      "".join(rail),
      "</ul>",
      '<p class="source"><span class="mono">'
      "scripts/generate_architecture_docs.py</span></p>",
      "</nav><main>",
      "".join(body),
      "</main></div>",
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
        problems.append(f"{ex.rel(path)}\n" + "\n".join(list(diff)[:40]))
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
  print(f"wrote {len(files)} files to {OUT.relative_to(ex.ROOT)}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
