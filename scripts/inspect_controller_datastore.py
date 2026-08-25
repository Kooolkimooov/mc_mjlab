"""List live controller datastore entries without invoking their callbacks."""

from __future__ import annotations

import argparse
import contextlib
import os
import re
from pathlib import Path

import mc_control

DEFAULT_CONFIG = Path(__file__).parents[1] / "etc" / "mc_rtc.yaml"
FUNCTION_TYPE = re.compile(r"std::function<(.+) \((.*)\)>")


@contextlib.contextmanager
def suppress_native_output():
  """Temporarily redirect native stdout and stderr to ``/dev/null``."""
  saved = (os.dup(1), os.dup(2))
  devnull = os.open(os.devnull, os.O_WRONLY)
  try:
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    yield
  finally:
    os.dup2(saved[0], 1)
    os.dup2(saved[1], 2)
    os.close(saved[0])
    os.close(saved[1])
    os.close(devnull)


def callback_kind(type_name: str) -> str:
  """Classify a demangled datastore type without invoking it."""
  match = FUNCTION_TYPE.fullmatch(type_name)
  if match is None:
    return "value"
  result, arguments = match.groups()
  if result == "void" and not arguments:
    return "command"
  if result == "void":
    return "setter"
  if not arguments:
    return "getter"
  return "function"


def main() -> None:
  """Build one disposable controller and print its datastore inventory as CSV."""
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
  parser.add_argument("--prefix", default="ismpc_walking::")
  parser.add_argument("--all", action="store_true", dest="show_all")
  args = parser.parse_args()

  with suppress_native_output():
    controller = mc_control.MCGlobalController(  # ty: ignore[unresolved-attribute]
      str(args.config)
    )
    datastore = controller.controller().datastore()
    rows = []
    for raw_key in sorted(datastore.keys()):
      key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
      if args.show_all or key.startswith(args.prefix):
        rows.append((key, datastore.type(raw_key)))
    del datastore
    del controller
  print("key,kind,type")
  for key, type_name in rows:
    print(f'"{key}",{callback_kind(type_name)},"{type_name}"')


if __name__ == "__main__":
  main()
