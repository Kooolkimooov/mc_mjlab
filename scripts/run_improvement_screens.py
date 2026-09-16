"""Run the retained standard-versus-ankle comparison resumably."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from mc_mjlab.tasks.residual_balance import ANKLE_TASK_ID, POSITION_TASK_ID


@dataclass(frozen=True)
class Screen:
  """One supported task in the short comparison."""

  name: str
  task_id: str


SCREENS = (
  Screen("standard", POSITION_TASK_ID),
  Screen("authority-ankle", ANKLE_TASK_ID),
)


def read_state(path: Path) -> dict[str, dict]:
  """Load completed-arm state, tolerating the first run."""
  return json.loads(path.read_text()) if path.is_file() else {}


def write_state(path: Path, state: dict[str, dict]) -> None:
  """Persist completion after each arm so an interrupted matrix can resume."""
  path.write_text(json.dumps(state, indent=2) + "\n")


def main() -> None:
  """Run requested unfinished arms with the roadmap's fixed screen budget."""
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--only", action="append", choices=[arm.name for arm in SCREENS])
  parser.add_argument("--iterations", type=int, default=188)
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--num-workers", type=int, default=30)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--log-root", type=Path, default=Path("logs/critique_screens"))
  parser.add_argument("--rerun", action="store_true")
  parser.add_argument("--list", action="store_true")
  args = parser.parse_args()
  selected = [arm for arm in SCREENS if args.only is None or arm.name in args.only]
  if args.list:
    for arm in selected:
      print(f"{arm.name:<24} {arm.task_id}")
    return

  args.log_root.mkdir(parents=True, exist_ok=True)
  state_path = args.log_root / "screen_state.json"
  state = read_state(state_path)

  for arm in selected:
    if not args.rerun and state.get(arm.name, {}).get("exit_code") == 0:
      print(f"[screen] skip completed {arm.name}", flush=True)
      continue

    command = [
      "uv",
      "run",
      "train",
      arm.task_id,
      "--env.scene.num-envs",
      str(args.num_envs),
      "--env.actions.mc-rtc-residual.num-workers",
      str(args.num_workers),
      "--agent.seed",
      str(args.seed),
      "--agent.max-iterations",
      str(args.iterations),
      "--agent.save-interval",
      "20",
      "--agent.run-name",
      f"critique-screen-{arm.name}",
      "--agent.logger",
      "tensorboard",
      "--agent.upload-model",
      "False",
      "--log-root",
      str(args.log_root),
    ]

    log_path = args.log_root / f"{arm.name}.log"
    print(f"[screen] start {arm.name}; log {log_path}", flush=True)
    with log_path.open("w") as stream:
      result = subprocess.run(
        command,
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
      )

    state[arm.name] = {
      "exit_code": result.returncode,
      "task_id": arm.task_id,
      "iterations": args.iterations,
      "seed": args.seed,
      "log": str(log_path),
    }

    write_state(state_path, state)
    if result.returncode:
      raise SystemExit(f"screen {arm.name} failed; see {log_path}")
    print(f"[screen] complete {arm.name}", flush=True)


if __name__ == "__main__":
  main()
