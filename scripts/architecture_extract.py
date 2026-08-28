#!/usr/bin/env python3
"""Static facts about this repo, read with griffe, grimp, pyreverse and ast."""

import ast
import functools
import subprocess
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import griffe
import grimp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PKG = "mc_mjlab"
BINDINGS = ("mc_control", "mc_rbdyn", "eigen", "sva")
TERM_KINDS = (
  "ObservationTermCfg",
  "RewardTermCfg",
  "TerminationTermCfg",
  "EventTermCfg",
  "CurriculumTermCfg",
  "MetricsTermCfg",
)


@dataclass
class Binding:
  """One manager term as it is constructed at a call site."""

  kind: str
  func: str
  key: str | None = None


@dataclass
class Registration:
  """One register_mjlab_task call, as written."""

  task_id: str
  env_cfg: str
  rl_cfg: str
  runner_cls: str | None


@dataclass
class Column:
  """One offset property of IoLayout, as a symbolic expression."""

  name: str
  expr: str
  depends: tuple[str, ...] = field(default_factory=tuple)


@functools.cache
def tree(path: Path) -> ast.Module:
  """Parse one source file once."""
  return ast.parse(path.read_text(), filename=str(path))


@functools.cache
def model() -> griffe.Module:
  """The griffe model of the package, loaded without importing it."""
  loaded = griffe.load(PKG, search_paths=[str(SRC)], resolve_aliases=False)
  assert isinstance(loaded, griffe.Module)
  return loaded


@functools.cache
def graph() -> grimp.ImportGraph:
  """The internal import graph, including imports nested in function bodies."""
  return grimp.build_graph(PKG, include_external_packages=True)


def sources() -> list[Path]:
  """Every source file the docs may describe."""
  return sorted(SRC.rglob("*.py"))


def rel(path: Path) -> str:
  """Repo-relative path, for provenance lines."""
  return str(path.relative_to(ROOT))


# --- griffe -----------------------------------------------------------------


def attribute(dotted: str) -> str:
  """The written value of a module-level attribute, as source text."""
  module, _, name = dotted.rpartition(".")
  member = model()[module][name] if module else model()[name]
  return str(member.value)


def dict_keys(dotted: str) -> list[str]:
  """The keys of a module-level dict attribute, in written order."""
  node = ast.parse(attribute(dotted), mode="eval").body
  if not isinstance(node, ast.Dict):
    raise TypeError(f"{dotted} is not a dict literal")
  return [ast.literal_eval(k) for k in node.keys if k is not None]


def dict_call_kwargs(dotted: str) -> dict[str, dict[str, str]]:
  """Keyword arguments of each constructor call inside a dict literal."""
  node = ast.parse(attribute(dotted), mode="eval").body
  if not isinstance(node, ast.Dict):
    raise TypeError(f"{dotted} is not a dict literal")
  found = {}
  for key, value in zip(node.keys, node.values, strict=True):
    if key is None or not isinstance(value, ast.Call):
      continue
    found[ast.literal_eval(key)] = {
      k.arg: ast.unparse(k.value) for k in value.keywords if k.arg
    }
  return found


def class_attributes(dotted: str) -> list[str]:
  """Attribute names declared on one class, in written order."""
  module, _, name = dotted.rpartition(".")
  return [
    member
    for member, obj in model()[module][name].members.items()
    if obj.kind is griffe.Kind.ATTRIBUTE
  ]


def module_constants(dotted: str, prefix: str) -> list[str]:
  """Module-level constants sharing a prefix, sorted."""
  return sorted(n for n in model()[dotted].members if n.startswith(prefix))


def classes() -> dict[str, list[str]]:
  """Every class in the package mapped to its base classes, as written."""
  found: dict[str, list[str]] = {}

  def walk(module: griffe.Module) -> None:
    for cls in module.classes.values():
      found[cls.canonical_path] = [str(base) for base in cls.bases]
    for sub in module.modules.values():
      walk(sub)

  walk(model())
  return found


def subclasses_of(base: str) -> list[str]:
  """Direct subclasses of one base, by its written name."""
  return sorted(n for n, bases in classes().items() if base in bases)


# --- grimp ------------------------------------------------------------------


def internal_modules() -> list[str]:
  """Every module of the package, in import-graph order."""
  return sorted(m for m in graph().modules if m.startswith(PKG))


def bucket(name: str) -> str:
  """The package a module belongs to, with the three robot dirs collapsed."""
  parts = name.split(".")
  if len(parts) > 2 and parts[1] == "robots":
    if (SRC / PKG / "robots" / parts[2]).is_dir():
      return f"{PKG}.robots.<ROBOT>"
  if (SRC / Path(*parts)).is_dir():
    return name
  return ".".join(parts[:-1]) or name


def package_edges() -> dict[str, set[str]]:
  """Internal import edges collapsed to package granularity."""
  edges: dict[str, set[str]] = defaultdict(set)
  for module in internal_modules():
    for target in graph().find_modules_directly_imported_by(module):
      if not target.startswith(PKG):
        continue
      source, sink = bucket(module), bucket(target)
      if source != sink:
        edges[source].add(sink)
  return dict(edges)


def raw_edge_count() -> int:
  """Internal import edges before any collapsing."""
  return sum(
    1
    for module in internal_modules()
    for target in graph().find_modules_directly_imported_by(module)
    if target.startswith(PKG)
  )


def binding_importers() -> dict[str, list[str]]:
  """Modules importing the mc_rtc bindings, mapped to which ones."""
  found: dict[str, list[str]] = {}
  for module in internal_modules():
    used = sorted(
      name
      for name in graph().find_modules_directly_imported_by(module)
      if name in BINDINGS
    )
    if used:
      found[module] = used
  return found


# --- pyreverse --------------------------------------------------------------


def class_diagram(target: str, out_dir: Path) -> str:
  """A mermaid classDiagram for one sub-package, straight out of pyreverse."""
  out_dir.mkdir(parents=True, exist_ok=True)
  subprocess.run(
    [
      sys.executable,
      "-m",
      "pylint.pyreverse.main",
      "-o",
      "mmd",
      "-p",
      "arch",
      "--only-classnames",
      "--output-directory",
      str(out_dir),
      str(SRC / PKG / target),
    ],
    check=True,
    capture_output=True,
    cwd=ROOT,
  )
  return (out_dir / "classes_arch.mmd").read_text().strip()


# --- ast: the four extractions no tool covers -------------------------------


def term_bindings(path: Path) -> list[Binding]:
  """Manager terms by construction site, since the dicts are mutated after."""
  found = []
  for node in ast.walk(tree(path)):
    if not isinstance(node, ast.Call):
      continue
    if not isinstance(node.func, ast.Name) or node.func.id not in TERM_KINDS:
      continue
    func = next((ast.unparse(k.value) for k in node.keywords if k.arg == "func"), None)
    if func:
      found.append(Binding(kind=node.func.id, func=func))
  return found


def dual_role_terms(path: Path) -> dict[str, list[str]]:
  """Terms bound under more than one manager, which must be drawn twice."""
  roles: dict[str, set[str]] = defaultdict(set)
  for binding in term_bindings(path):
    roles[binding.func].add(binding.kind)
  return {f: sorted(k) for f, k in sorted(roles.items()) if len(k) > 1}


def io_layout_columns(path: Path) -> list[Column]:
  """IoLayout's offsets as symbolic expressions, ordered by dependency."""
  node = next(
    n
    for n in ast.walk(tree(path))
    if isinstance(n, ast.ClassDef) and n.name == "IoLayout"
  )
  columns = {}
  for item in node.body:
    if not isinstance(item, ast.FunctionDef):
      continue
    if not any(ast.unparse(d) == "property" for d in item.decorator_list):
      continue
    body = item.body[0]
    if len(item.body) != 1 or not isinstance(body, ast.Return) or body.value is None:
      continue
    expr = ast.unparse(body.value).replace("self.", "")
    columns[item.name] = Column(name=item.name, expr=expr)

  for column in columns.values():
    column.depends = tuple(n for n in columns if n != column.name and n in column.expr)

  ordered: list[Column] = []
  remaining = dict(columns)
  while remaining:
    ready = [
      c for c in remaining.values() if all(d not in remaining for d in c.depends)
    ]
    if not ready:
      raise RuntimeError(f"cycle among IoLayout offsets: {sorted(remaining)}")
    for column in sorted(ready, key=lambda c: c.name):
      ordered.append(column)
      del remaining[column.name]
  return ordered


def pipe_protocol(host: Path, pool: Path) -> dict[str, list[str]]:
  """The worker command vocabulary, read from both ends so they can disagree."""
  handled = []
  worker = next(
    n
    for n in ast.walk(tree(host))
    if isinstance(n, ast.FunctionDef) and n.name == "worker_main"
  )
  for node in ast.walk(worker):
    if not isinstance(node, ast.Compare) or not isinstance(node.left, ast.Name):
      continue
    if node.left.id != "cmd":
      continue
    for other in node.comparators:
      if isinstance(other, ast.Constant) and isinstance(other.value, str):
        handled.append(other.value)

  sent, replies = [], []
  for source, sink in ((pool, sent), (host, replies)):
    for node in ast.walk(tree(source)):
      if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        continue
      if node.func.attr != "send" or not node.args:
        continue
      first = node.args[0]
      if isinstance(first, ast.Tuple) and first.elts:
        head = first.elts[0]
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
          sink.append(head.value)
  return {
    "handled": sorted(set(handled)),
    "sent": sorted(set(sent)),
    "replies": sorted(set(replies)),
  }


def call_sequence(path: Path, method: str, receivers: tuple[str, ...]) -> list[str]:
  """Calls in source order inside one method: the control step's real ordering."""
  node = next(
    n
    for n in ast.walk(tree(path))
    if isinstance(n, ast.FunctionDef) and n.name == method
  )
  beats = []
  for child in ast.walk(node):
    if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
      continue
    target = ast.unparse(child.func)
    if any(target.startswith(f"self.{r}.") for r in receivers):
      beats.append((child.lineno, target.replace("self.", "")))
    elif target.startswith("self.") and target.count(".") == 1:
      beats.append((child.lineno, target.replace("self.", "")))
  seen, ordered = set(), []
  for _, name in sorted(beats):
    if name not in seen:
      seen.add(name)
      ordered.append(name)
  return ordered


def import_style(path: Path, name: str) -> str:
  """How one import is written: guarded, deferred into a body, or plain."""
  root = tree(path)
  for node in ast.walk(root):
    if not isinstance(node, (ast.Import, ast.ImportFrom)):
      continue
    if not any(alias.name.split(".")[0] == name for alias in node.names):
      continue
    for parent in ast.walk(root):
      if isinstance(parent, ast.Try) and node in ast.walk(parent):
        return "guarded by try/except ImportError"
      if isinstance(parent, ast.FunctionDef) and node in ast.walk(parent):
        return f"deferred into {parent.name}()"
    return "module level"
  return "absent"


def kwarg_literal(path: Path, name: str) -> str | None:
  """The first literal value given to one keyword anywhere in a module."""
  for node in ast.walk(tree(path)):
    if not isinstance(node, ast.Call):
      continue
    for keyword in node.keywords:
      if keyword.arg == name and isinstance(keyword.value, ast.Constant):
        return ast.unparse(keyword.value)
  return None


def class_channels(path: Path) -> str:
  """The output_channels a concrete action subclass declares, as written."""
  for node in ast.walk(tree(path)):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
      target = node.targets[0]
      if isinstance(target, ast.Name) and target.id == "output_channels":
        return ast.unparse(node.value)
  return "?"


def param_defaults(path: Path, function: str) -> dict[str, str]:
  """Parameter defaults of one function, for values forwarded rather than written."""
  node = next(
    (
      n
      for n in ast.walk(tree(path))
      if isinstance(n, ast.FunctionDef) and n.name == function
    ),
    None,
  )
  if node is None:
    return {}
  args = node.args
  found = {}
  for name, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
    if default is not None:
      found[name.arg] = ast.unparse(default)
  positional = args.posonlyargs + args.args
  for name, default in zip(
    positional[len(positional) - len(args.defaults) :], args.defaults, strict=True
  ):
    found[name.arg] = ast.unparse(default)
  return found


def registrations(path: Path) -> list[Registration]:
  """Every register_mjlab_task call in one task package, as written."""
  found = []
  for node in ast.walk(tree(path)):
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
      continue
    if node.func.id != "register_mjlab_task":
      continue
    kwargs = {k.arg: ast.unparse(k.value) for k in node.keywords if k.arg}
    found.append(
      Registration(
        task_id=kwargs.get("task_id", "?"),
        env_cfg=kwargs.get("env_cfg", "?"),
        rl_cfg=kwargs.get("rl_cfg", "?"),
        runner_cls=kwargs.get("runner_cls"),
      )
    )
  return found


def dotted_class_paths(path: Path) -> list[str]:
  """ "module:Class" strings, the wiring no import graph can see."""
  found = []
  for node in ast.walk(tree(path)):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
      if ":" in node.value and node.value.startswith(PKG):
        found.append(node.value)
  return sorted(set(found))


def literal_kwargs(path: Path, call: str, names: tuple[str, ...]) -> dict[str, str]:
  """Literal keyword arguments of one constructor call, for the rate stack."""
  found: dict[str, str] = {}
  for node in ast.walk(tree(path)):
    if not isinstance(node, ast.Call):
      continue
    if ast.unparse(node.func) != call:
      continue
    for keyword in node.keywords:
      if keyword.arg in names and keyword.arg not in found:
        found[keyword.arg] = ast.unparse(keyword.value)
  return found


# --- pyproject / config -----------------------------------------------------


def entry_points() -> dict[str, str]:
  """The entry points this package declares."""
  data = tomllib.loads((ROOT / "pyproject.toml").read_text())
  groups = data["project"].get("entry-points", {})
  return {f"{g}:{k}": v for g, items in groups.items() for k, v in items.items()}


def requires_python() -> str:
  """The interpreter this package pins, which the bindings must match."""
  return tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
    "requires-python"
  ]


def doc_anchors() -> dict[str, set[str]]:
  """Anchors defined by the hand-written docs, so generated links can be checked."""
  found: dict[str, set[str]] = {}
  import re

  for md in sorted((ROOT / "docs").rglob("*.md")):
    anchors = set()
    for line in md.read_text().splitlines():
      if line.startswith("#"):
        text = line.lstrip("#").strip()
        anchors.add(re.sub(r"[^\w\- ]", "", text).strip().lower().replace(" ", "-"))
    found[str(md.resolve())] = anchors
  return found
