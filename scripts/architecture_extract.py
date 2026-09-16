#!/usr/bin/env python3
"""Static facts about this repo, read with griffe, grimp, pyreverse and ast."""

from __future__ import annotations

import ast
import functools
import re
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


def project_description() -> str:
  """The one-line description the package declares."""
  data = tomllib.loads((ROOT / "pyproject.toml").read_text())
  return data["project"]["description"]


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
        # `## IDENTIFIER` headings are linked verbatim so grep finds them; the
        # GitHub slug lowercases that away. docs/README.md
        anchors.add(text)
    found[str(md.resolve())] = anchors
  return found


def attribute_types(path: Path, class_name: str) -> dict[str, str]:
  """Instance attributes assigned a constructor call, mapped to that class."""
  node = next(
    (
      n
      for n in ast.walk(tree(path))
      if isinstance(n, ast.ClassDef) and n.name == class_name
    ),
    None,
  )
  if node is None:
    return {}
  found = {}
  for child in ast.walk(node):
    if not isinstance(child, ast.Assign) or len(child.targets) != 1:
      continue
    target = child.targets[0]
    value = child.value
    if not isinstance(target, ast.Attribute) or not isinstance(target.value, ast.Name):
      continue
    if target.value.id != "self" or not isinstance(value, ast.Call):
      continue
    callee = ast.unparse(value.func).rsplit(".", 1)[-1]
    if callee[:1].isupper():
      found[target.attr] = callee
  return found


def collaborations(path: Path, class_name: str) -> list[tuple[str, str]]:
  """Methods this class calls on the collaborators it constructs."""
  owned = attribute_types(path, class_name)
  node = next(
    n
    for n in ast.walk(tree(path))
    if isinstance(n, ast.ClassDef) and n.name == class_name
  )
  found = set()
  for child in ast.walk(node):
    if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
      continue
    receiver = child.func.value
    if not isinstance(receiver, ast.Attribute) or not isinstance(
      receiver.value, ast.Name
    ):
      continue
    if receiver.value.id == "self" and receiver.attr in owned:
      found.add((owned[receiver.attr], child.func.attr))
  return sorted(found)


def internal_call_graph(
  roots: dict[str, Path], depth: int = 3
) -> list[tuple[str, str]]:
  """Which of the named functions call which others, by source-level name."""
  bodies = {}
  for name, path in roots.items():
    node = next(
      (
        n
        for n in ast.walk(tree(path))
        if isinstance(n, ast.FunctionDef) and n.name == name
      ),
      None,
    )
    if node is not None:
      bodies[name] = node
  edges = set()
  for name, node in bodies.items():
    for child in ast.walk(node):
      if not isinstance(child, ast.Call):
        continue
      callee = ast.unparse(child.func).rsplit(".", 1)[-1]
      if callee in bodies and callee != name:
        edges.add((name, callee))
  return sorted(edges)


def layout_inputs(path: Path) -> list[tuple[str, str, str]]:
  """IoLayout fields the offset expressions depend on, with their annotations."""
  node = next(
    n
    for n in ast.walk(tree(path))
    if isinstance(n, ast.ClassDef) and n.name == "IoLayout"
  )
  fields = {}
  for item in node.body:
    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
      default = ast.unparse(item.value) if item.value else ""
      fields[item.target.id] = (ast.unparse(item.annotation), default)
  used = set()
  for column in io_layout_columns(path):
    for token in ast.walk(ast.parse(column.expr, mode="eval")):
      if isinstance(token, ast.Name) and token.id in fields:
        used.add(token.id)
  return [(name, *fields[name]) for name in sorted(used)]


def doc_links(paths: list[Path]) -> list[tuple[str, str]]:
  """docs/ links the source comments already carry, with the file that carries them."""
  pattern = re.compile(r"docs/([\w./-]+\.md)(?:#([\w-]+))?")
  found = set()
  for path in paths:
    for target, anchor in pattern.findall(path.read_text()):
      found.add((f"{target}#{anchor}" if anchor else target, path.stem))
  return sorted(found)


def path_constants(dotted_modules: tuple[str, ...]) -> list[tuple[str, str, str]]:
  """Module constants whose value is built from a filesystem path."""
  markers = ("Path.home()", "REPO_ROOT", "parents[", "__file__")
  found = []
  for module in dotted_modules:
    for name, member in model()[module].members.items():
      if member.kind is not griffe.Kind.ATTRIBUTE or not name.isupper():
        continue
      value = str(member.value)
      if any(marker in value for marker in markers):
        found.append((name, module, value))
  return sorted(found)


@dataclass
class Term:
  """One manager term as it is written at its construction site."""

  group: str
  key: str
  kind: str
  func: str
  params: dict[str, str]
  condition: str = ""


def _condition_map(root: ast.Module) -> dict[int, str]:
  """Map each node id to the `if` test that guards it, where there is one."""
  guards: dict[int, str] = {}
  for node in ast.walk(root):
    if not isinstance(node, ast.If):
      continue
    test = ast.unparse(node.test)
    for branch, label in ((node.body, test), (node.orelse, f"not ({test})")):
      for statement in branch:
        for child in ast.walk(statement):
          guards.setdefault(id(child), label)
  return guards


def _terms_from_dict(
  node: ast.Dict, group: str, condition: str, constants: dict[str, list[str]]
) -> list[Term]:
  """Terms written in one dict display, expanding a starred comprehension."""
  found = []
  for key, value in zip(node.keys, node.values, strict=True):
    if key is None and isinstance(value, ast.DictComp):
      call = value.value
      source = ast.unparse(value.generators[0].iter)
      if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
        for name in constants.get(source, [source]):
          found += _one(group, name, call, condition)
      continue
    if isinstance(key, ast.Constant) and isinstance(value, ast.Call):
      found += _one(group, str(key.value), value, condition)
  return found


def _one(group: str, key: str, call: ast.Call, condition: str) -> list[Term]:
  """The term this call builds, or nothing when it is not a term constructor."""
  built = _term(group, key, call, condition)
  return [built] if built is not None else []


def _term(group: str, key: str, call: ast.Call, condition: str) -> Term | None:
  """One term, or None when the call is not a manager term constructor."""
  if not isinstance(call.func, ast.Name) or call.func.id not in TERM_KINDS:
    return None
  params = {k.arg: ast.unparse(k.value) for k in call.keywords if k.arg}
  return Term(
    group=group,
    key=str(key),
    kind=call.func.id,
    func=params.pop("func", ""),
    params=params,
    condition=condition,
  )


def sequence_constants(path: Path) -> dict[str, list[str]]:
  """Module-level tuples and lists of strings, for expanding comprehensions."""
  found = {}
  for node in tree(path).body:
    if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
      targets = node.targets if isinstance(node, ast.Assign) else [node.target]
      if len(targets) != 1 or not isinstance(targets[0], ast.Name):
        continue
      if isinstance(node.value, (ast.Tuple, ast.List)):
        items = [
          element.value
          for element in node.value.elts
          if isinstance(element, ast.Constant) and isinstance(element.value, str)
        ]
        if len(items) == len(node.value.elts) and items:
          found[targets[0].id] = items
  return found


def manager_terms(path: Path) -> list[Term]:
  """Every manager term written in a module, with the group and guard it carries."""
  root = tree(path)
  guards = _condition_map(root)
  constants = sequence_constants(path)
  found: list[Term] = []
  for node in ast.walk(root):
    condition = guards.get(id(node), "")
    for target, value in _bindings(node):
      if isinstance(target, ast.Name) and isinstance(value, ast.Dict):
        found += _terms_from_dict(value, target.id, condition, constants)
      elif isinstance(target, ast.Subscript) and isinstance(value, ast.Call):
        key = target.slice
        if isinstance(key, ast.Constant):
          found += _one(ast.unparse(target.value), str(key.value), value, condition)
  return found


def _bindings(node: ast.AST) -> list[tuple[ast.expr, ast.expr]]:
  """The (target, value) pairs of any assignment form that binds a dict."""
  if isinstance(node, ast.Assign) and len(node.targets) == 1 and node.value:
    return [(node.targets[0], node.value)]
  if isinstance(node, (ast.AnnAssign, ast.AugAssign)) and node.value:
    return [(node.target, node.value)]
  return []


def group_exposure(path: Path) -> dict[str, str]:
  """Which local dict each observation group is built from."""
  found = {}
  for node in ast.walk(tree(path)):
    pairs = _bindings(node)
    if not pairs or not isinstance(pairs[0][1], ast.Dict):
      continue
    holder = pairs[0][1]
    for key, value in zip(holder.keys, holder.values, strict=True):
      if not isinstance(key, ast.Constant) or not isinstance(value, ast.Call):
        continue
      terms = next((k.value for k in value.keywords if k.arg == "terms"), None)
      if isinstance(terms, ast.Call) and len(terms.args) == 1:
        terms = terms.args[0]
      if terms is not None:
        found[str(key.value)] = ast.unparse(terms)
  return found


def group_composition(path: Path) -> list[tuple[str, str, str]]:
  """How each term dict is built beyond its literal entries."""
  root = tree(path)
  guards = _condition_map(root)
  loops = {
    node.target.id: ast.unparse(node.iter)
    for node in ast.walk(root)
    if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
  }
  found = []
  for node in ast.walk(root):
    condition = guards.get(id(node), "")
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
      target = node.targets[0]
      if isinstance(target, ast.Name) and isinstance(node.value, ast.DictComp):
        found.append(
          (target.id, "comprehension over", ast.unparse(node.value.generators[0].iter))
        )
      if isinstance(target, ast.Subscript):
        owner = ast.unparse(target.value)
        if (
          isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "replace"
        ):
          found.append((owner, "replace", ast.unparse(target.slice)))
    if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
      if condition:
        found.append((node.target.id, "merged when", condition))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
      if node.func.attr == "pop" and node.args:
        found.append(
          (
            ast.unparse(node.func.value),
            "pop",
            loops.get(ast.unparse(node.args[0]), ast.unparse(node.args[0])),
          )
        )
  return sorted(set(found))


def module_scalars(dotted: str) -> dict[str, str]:
  """Module-level constants that hold a written scalar value."""
  found = {}
  for name, member in model()[dotted].members.items():
    if member.kind is griffe.Kind.ATTRIBUTE and name.isupper():
      found[name] = str(member.value)
  return found


def resolved(expression: str, scalars: dict[str, str]) -> str:
  """An expression with any bare module constant followed by its value."""
  if expression in scalars and scalars[expression] != expression:
    return f"{expression} = {scalars[expression]}"
  return expression


def task_packages() -> list[Path]:
  """Task sub-packages, the only thing mjlab's importer walks."""
  tasks = SRC / PKG / "tasks"
  return sorted(p for p in tasks.iterdir() if (p / "__init__.py").is_file())


def _enclosing_loops(root: ast.Module) -> dict[int, list[ast.For]]:
  """Map each node id to the `for` statements that enclose it."""
  found: dict[int, list[ast.For]] = {}

  def walk(node: ast.AST, loops: list[ast.For]) -> None:
    found[id(node)] = loops
    for child in ast.iter_child_nodes(node):
      walk(child, loops + [node] if isinstance(node, ast.For) else loops)

  walk(root, [])
  return found


def task_name_calls(path: Path) -> list[tuple[str, str, bool]]:
  """Config path and suffix of each get_task_name call, f-strings expanded over loops."""
  root = tree(path)
  loops = _enclosing_loops(root)
  found: list[tuple[str, str, bool]] = []
  for node in ast.walk(root):
    if not isinstance(node, ast.Call) or ast.unparse(node.func) != "get_task_name":
      continue
    if len(node.args) < 3:
      continue
    config = ast.unparse(node.args[1])
    suffix = node.args[2]
    if isinstance(suffix, ast.Constant) and isinstance(suffix.value, str):
      found.append((config, suffix.value, True))
      continue
    if not isinstance(suffix, ast.JoinedStr):
      found.append((config, ast.unparse(suffix), False))
      continue
    template = ast.unparse(suffix)[2:-1]
    options: list[list[str]] = [[template]]
    for loop in loops.get(id(node), []):
      if not isinstance(loop.target, ast.Name):
        continue
      values = [
        ast.unparse(item)
        for item in getattr(loop.iter, "elts", [])
        if isinstance(item, ast.Constant)
      ]
      if not values:
        continue
      variable = "{" + loop.target.id + "}"
      options = [
        [text.replace(variable, value.strip("'\"")) for text in group]
        for group in options
        for value in values
      ]
    expanded = [text for group in options for text in group]
    found += [(config, text, "{" not in text) for text in expanded]
  return sorted(set(found))


def conditional_imports(path: Path) -> dict[str, str]:
  """Modules this file imports only under an `if`, mapped to that condition."""
  root = tree(path)
  guards = _condition_map(root)
  found = {}
  for node in ast.walk(root):
    condition = guards.get(id(node), "")
    if not condition:
      continue
    if isinstance(node, ast.ImportFrom) and node.module:
      found[node.module.rsplit(".", 1)[-1]] = condition
    elif isinstance(node, ast.Import):
      for alias in node.names:
        found[alias.name.rsplit(".", 1)[-1]] = condition
  return found
