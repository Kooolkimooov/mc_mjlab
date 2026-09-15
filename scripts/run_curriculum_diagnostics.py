"""Run frozen-versus-gradual curriculum smoke tests and diagnostics."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from mc_mjlab.tasks.naming import get_task_name

TASK_DIR = "residual_balance"


@dataclass(frozen=True)
class Arm:
  """One impulse-curriculum diagnostic arm."""

  name: str
  task_id: str


def task_id(schedule: str) -> str:
  """Resolve a curriculum task id through the repository naming utility."""
  return get_task_name(TASK_DIR, f"position-ankle-curriculum-{schedule}")


ARMS = (
  Arm("frozen", task_id("frozen")),
  Arm("gradual", task_id("gradual")),
)


def read_state(path: Path) -> dict[str, dict]:
  """Load completed-run state, tolerating the first invocation."""
  return json.loads(path.read_text()) if path.is_file() else {}


def write_state(path: Path, state: dict[str, dict]) -> None:
  """Persist completion after each run so the sequence can resume."""
  path.write_text(json.dumps(state, indent=2) + "\n")


def train_command(
  arm: Arm,
  phase: str,
  iterations: int,
  num_envs: int,
  num_workers: int,
  seed: int,
  logger: str,
  log_root: Path,
) -> list[str]:
  """Build one explicit mjlab training command."""
  return [
    "uv",
    "run",
    "train",
    arm.task_id,
    "--env.scene.num-envs",
    str(num_envs),
    "--env.actions.mc-rtc-residual.num-workers",
    str(num_workers),
    "--agent.seed",
    str(seed),
    "--agent.max-iterations",
    str(iterations),
    "--agent.save-interval",
    "20",
    "--agent.run-name",
    f"curriculum-{phase}-{arm.name}",
    "--agent.logger",
    logger,
    "--agent.upload-model",
    "False",
    "--enable-nan-guard",
    "True",
    "--log-root",
    str(log_root / phase),
  ]


def main() -> None:
  """Run smoke arms first and diagnostics only after both pass."""
  parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument("--only", action="append", choices=[arm.name for arm in ARMS])
  parser.add_argument("--phase", action="append", choices=("smoke", "diagnostic"))
  parser.add_argument("--smoke-iterations", type=int, default=2)
  parser.add_argument("--diagnostic-iterations", type=int, default=220)
  parser.add_argument("--smoke-num-envs", type=int, default=8)
  parser.add_argument("--smoke-num-workers", type=int, default=6)
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--num-workers", type=int, default=30)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--logger", choices=("wandb", "tensorboard"), default="wandb")
  parser.add_argument(
    "--log-root", type=Path, default=Path("logs/curriculum_diagnostics")
  )
  parser.add_argument("--rerun", action="store_true")
  parser.add_argument("--list", action="store_true")
  args = parser.parse_args()
  arms = [arm for arm in ARMS if args.only is None or arm.name in args.only]
  phases = args.phase or ["smoke", "diagnostic"]
  if args.list:
    for arm in arms:
      print(f"{arm.name:<12} {arm.task_id}")
    return

  args.log_root.mkdir(parents=True, exist_ok=True)
  state_path = args.log_root / "run_state.json"
  state = read_state(state_path)
  for phase in phases:
    smoke = phase == "smoke"
    iterations = args.smoke_iterations if smoke else args.diagnostic_iterations
    num_envs = args.smoke_num_envs if smoke else args.num_envs
    num_workers = args.smoke_num_workers if smoke else args.num_workers
    logger = "tensorboard" if smoke else args.logger
    for arm in arms:
      key = f"{phase}:{arm.name}"
      if not args.rerun and state.get(key, {}).get("exit_code") == 0:
        print(f"[curriculum] skip completed {key}", flush=True)
        continue
      command = train_command(
        arm,
        phase,
        iterations,
        num_envs,
        num_workers,
        args.seed,
        logger,
        args.log_root,
      )
      log_path = args.log_root / f"{phase}-{arm.name}.log"
      print(f"[curriculum] start {key}; log {log_path}", flush=True)
      with log_path.open("w") as stream:
        result = subprocess.run(
          command,
          stdout=stream,
          stderr=subprocess.STDOUT,
          text=True,
          check=False,
        )
      state[key] = {
        "exit_code": result.returncode,
        "task_id": arm.task_id,
        "iterations": iterations,
        "num_envs": num_envs,
        "num_workers": num_workers,
        "seed": args.seed,
        "logger": logger,
        "log": str(log_path),
      }
      write_state(state_path, state)
      if result.returncode:
        raise SystemExit(f"curriculum run {key} failed; see {log_path}")
      print(f"[curriculum] complete {key}", flush=True)


if __name__ == "__main__":
  main()
