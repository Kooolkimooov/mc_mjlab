#!/usr/bin/env python3
"""Keep notes out of the code: one-line docstrings, prose under 10%, live doc links."""

from __future__ import annotations

import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path

MAX_PROSE = 0.10
MIN_LINES = 30  # Below this a lone module docstring is unavoidably most of the file.
ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ("src", "scripts")
# Docs name C++ identifiers and files too; the native tree lives under src/.
CODE_SUFFIXES = (".py", ".cpp", ".hpp")
LINK = re.compile(r"docs/([\w./-]+\.md)#([\w-]+)")
# A heading naming code: snake_case, CONST_CASE or CamelCase. A plain English
# word like "Assets" is a section title, not something to look for in the code.
IDENTIFIER = re.compile(r"^(?=.*(_|[a-z][A-Z]))[A-Za-z_][\w.]*$")


def measure(path: Path) -> tuple[int, int, int, list[int]]:
  """Return (total lines, comment lines, docstring lines, multi-line docstring rows)."""
  src = path.read_text()
  total = src.count("\n") + (0 if src.endswith("\n") else 1)
  comment = doc = 0
  prev = None
  for tok in tokenize.generate_tokens(io.StringIO(src).readline):
    if tok.type == tokenize.COMMENT:
      comment += 1
    elif tok.type == tokenize.STRING and prev in (
      tokenize.INDENT,
      tokenize.NEWLINE,
      tokenize.DEDENT,
      None,
    ):
      doc += tok.string.count("\n") + 1
    if tok.type != tokenize.NL:
      prev = tok.type

  multiline = []
  for node in ast.walk(ast.parse(src)):
    if not isinstance(
      node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    ):
      continue
    if ast.get_docstring(node) is None:
      continue
    expr = node.body[0]
    if expr.end_lineno is not None and expr.end_lineno > expr.lineno:
      multiline.append(expr.lineno)
  return total, comment, doc, multiline


def headings() -> dict[str, set[str]]:
  """Map each docs/*.md file to the GitHub-style anchors it defines."""
  found: dict[str, set[str]] = {}
  for md in sorted((ROOT / "docs").rglob("*.md")):
    anchors = set()
    for line in md.read_text().splitlines():
      if line.startswith("#"):
        text = line.lstrip("#").strip()
        anchors.add(re.sub(r"[^\w\- ]", "", text).strip().lower().replace(" ", "-"))
    found[str(md.relative_to(ROOT / "docs"))] = anchors
  return found


def check(paths: list[Path], link_targets: dict[str, set[str]]) -> list[str]:
  problems = []
  for path in paths:
    rel = path.relative_to(ROOT) if path.is_absolute() else path
    try:
      total, comment, doc, multiline = measure(path)
    except (SyntaxError, tokenize.TokenError) as exc:
      problems.append(f"{rel}: could not parse ({exc})")
      continue

    if multiline:
      rows = ", ".join(f"L{n}" for n in multiline)
      problems.append(
        f"{rel}: {len(multiline)} multi-line docstring(s) at {rows} "
        f"-- docstrings are one line; the body belongs in docs/"
      )

    # The 10% is on *comments*. One-line docstrings are always allowed: the
    # one-line rule already bounds them at one per definition, and capping them
    # too would mean deleting API documentation to hit a ratio.
    if total >= MIN_LINES and comment > MAX_PROSE * total:
      budget = int(MAX_PROSE * total)
      problems.append(
        f"{rel}: {100 * comment / total:.0f}% comments ({comment}/{total} lines, "
        f"budget {budget}; {doc} docstring lines not counted) "
        f"-- move notes to docs/ or split the file"
      )

    for target, anchor in LINK.findall(path.read_text()):
      # Case-insensitive: a heading is the identifier it documents, so a
      # CONST_CASE one anchors lowercased while the link keeps the spelling.
      if anchor.lower() not in link_targets.get(target, set()):
        problems.append(f"{rel}: dead link docs/{target}#{anchor}")
  return problems


def orphans(link_targets: dict[str, set[str]], code: str) -> list[str]:
  """Docs headings naming an identifier the code no longer defines."""
  stale = []
  for md in sorted((ROOT / "docs").rglob("*.md")):
    if md.name == "README.md":
      continue  # The conventions doc; its headings are examples, not identifiers.
    for line in md.read_text().splitlines():
      if not line.startswith("## "):
        continue
      name = line[3:].strip().strip("`")
      if name.endswith(CODE_SUFFIXES):
        # rglob returns a generator, which is always truthy -- consume it.
        if not any(any((ROOT / d).rglob(name)) for d in CODE_DIRS):
          stale.append(f"docs/{md.relative_to(ROOT / 'docs')}: no such file '{name}'")
      elif IDENTIFIER.match(name) and name.split(".")[0] not in code:
        stale.append(
          f"docs/{md.relative_to(ROOT / 'docs')}: '{name}' is not in the code"
        )
  return stale


def source_text() -> str:
  """Every source file the docs headings may name, Python and C++ alike."""
  files = [
    path
    for directory in CODE_DIRS
    for path in sorted((ROOT / directory).rglob("*"))
    if path.suffix in CODE_SUFFIXES
  ]
  return "\n".join(path.read_text() for path in files)


def python_files(targets: list[str]) -> list[Path]:
  out: list[Path] = []
  for t in targets:
    p = Path(t)
    if not p.is_absolute():
      p = ROOT / p
    out.extend(sorted(p.rglob("*.py")) if p.is_dir() else [p])
  return [p for p in out if p.suffix == ".py" and p.exists()]


def main() -> int:
  argv = sys.argv[1:]
  # Pre-commit runs the full check strictly; the edit-time hook stays advisory.
  strict = "--strict" in argv
  argv = [arg for arg in argv if arg != "--strict"]
  if argv:
    files, hook = python_files(argv), False
  else:
    # Hook mode: the PostToolUse payload names the file that was just written.
    try:
      payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
      return 0
    raw = (payload.get("tool_input") or {}).get("file_path")
    if not raw or not raw.endswith(".py"):
      return 0
    path = Path(raw)
    if not path.exists() or not any(
      str(path).startswith(str(ROOT / d)) for d in CODE_DIRS
    ):
      return 0
    files, hook = [path], True

  link_targets = headings()
  problems = check(files, link_targets)

  stale: list[str] = []
  if not hook:
    stale = orphans(link_targets, source_text())
    for warning in stale:
      print(
        f"{'error' if strict else 'warning'}: {warning}",
        file=sys.stderr if strict else sys.stdout,
      )

  for problem in problems:
    print(problem, file=sys.stderr)
  if problems and hook:
    # Advisory: surface it in the transcript without failing the edit, since a
    # file can legitimately be over budget partway through a refactor.
    return 2
  return 1 if problems or (strict and stale) else 0


if __name__ == "__main__":
  sys.exit(main())
