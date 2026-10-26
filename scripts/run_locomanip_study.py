"""Run one bounded study training job and validate each scheduled checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mjlab  # noqa: F401
from evaluation.study_io import write_json
from freeze_locomanip_references import file_identity, revision

from mc_mjlab.tasks.locomanip_study import CONTROL_TASK_ID, TASK_IDS

EVALUATION_ITERATIONS = (*range(200, 2000, 200), 1999)


def main() -> None:
  """Keep training and validation owned by the tmux-launched coordinator."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--variant", choices=tuple(TASK_IDS), default="control")
  parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
  parser.add_argument("--reference", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--eval-workers", type=int, default=4)
  parser.add_argument("--eval-batch-size", type=int, default=32)
  args = parser.parse_args()
  if args.eval_workers < 1 or args.eval_batch_size < 1:
    parser.error("positive evaluation worker and batch counts required")
  dirty = subprocess.check_output(
    ["git", "diff", "HEAD", "--name-only"], text=True
  ).splitlines()
  if set(dirty) - {"pyproject.toml"}:
    parser.error("commit source/configuration changes before launching training")
  reference = file_identity(args.reference)
  if not reference["present"]:
    parser.error("reference checkpoint is missing")
  args.output = args.output.resolve()
  args.output.mkdir(parents=True, exist_ok=False)
  task = TASK_IDS[args.variant]
  command = [
    "uv",
    "run",
    "--no-sync",
    "train",
    task,
    "--log-root",
    str(args.output / "training"),
    "--agent.run-name",
    f"study-{args.variant}-s{args.seed}",
    "--agent.seed",
    str(args.seed),
    "--agent.logger",
    "tensorboard",
    "--env.scene.num-envs",
    "256",
    "--agent.num-steps-per-env",
    "64",
    "--agent.max-iterations",
    "2000",
    "--agent.save-interval",
    "20",
  ]
  state: dict[str, Any] = {
    "source": revision(Path.cwd()),
    "mjlab": revision(Path(mjlab.__file__).resolve().parents[2]),
    "command": command,
    "reference": reference,
    "seed": args.seed,
    "task": task,
    "budget": {"envs": 256, "rollout_steps": 64, "iterations": 2000},
    "evaluation_iterations": EVALUATION_ITERATIONS,
    "evaluations": [],
    "status": "starting",
  }
  save_status(args.output, state)
  with (args.output / "train.log").open("w") as log:
    trainer = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    state.update(training_pid=trainer.pid, status="training")
    save_status(args.output, state)
    try:
      monitor(trainer, args, state)
    finally:
      if trainer.poll() is None:
        trainer.terminate()
        trainer.wait()
      state["training_exit_code"] = trainer.returncode
      save_status(args.output, state)


def monitor(trainer: subprocess.Popen, args: argparse.Namespace, state: dict) -> None:
  """Validate saved milestones serially while training proceeds independently."""
  evaluated: set[int] = set()
  while True:
    runs = list((args.output / "training" / state["task"]).glob("*"))
    runs = [path for path in runs if path.is_dir()]
    if len(runs) > 1:
      raise RuntimeError("campaign output contains more than one training run")
    run = runs[0] if runs else None
    if run is not None:
      state["run_directory"] = str(run)
      for iteration in EVALUATION_ITERATIONS:
        checkpoint = run / f"model_{iteration}.pt"
        if iteration in evaluated or not checkpoint.is_file():
          continue
        size = checkpoint.stat().st_size
        time.sleep(2)
        if checkpoint.stat().st_size != size or not size:
          continue
        evaluate(checkpoint, iteration, args, state)
        evaluated.add(iteration)
    code = trainer.poll()
    state["training_exit_code"] = code
    if code is not None:
      missing = sorted(set(EVALUATION_ITERATIONS) - evaluated)
      state["missing_evaluations"] = missing
      failed = any(e["exit_code"] != 0 for e in state["evaluations"])
      state["status"] = (
        "run_finished"
        if code == 0 and not missing and not failed
        else "needs_attention"
      )
      save_status(args.output, state)
      return
    state["status"] = "training"
    save_status(args.output, state)
    time.sleep(10)


def evaluate(
  checkpoint: Path, iteration: int, args: argparse.Namespace, state: dict
) -> None:
  """Record checkpoint identities and compare on the fixed validation cohort."""
  output = args.output / f"validation-{iteration}"
  command = [
    "uv",
    "run",
    "--no-sync",
    "python",
    "scripts/evaluate_locomanip_study.py",
    "--candidate-task",
    state["task"],
    "--candidate-checkpoint",
    str(checkpoint),
    "--reference-task",
    CONTROL_TASK_ID,
    "--reference-checkpoint",
    str(args.reference.resolve()),
    "--batch-size",
    str(args.eval_batch_size),
    "--workers",
    str(args.eval_workers),
    "--output",
    str(output),
  ]
  entry = {
    "iteration": iteration,
    "checkpoint": file_identity(checkpoint),
    "command": command,
    "output": str(output),
    "started_at": datetime.now().astimezone().isoformat(),
    "exit_code": None,
  }
  state["evaluations"].append(entry)
  state["status"] = "validating"
  save_status(args.output, state)
  with (args.output / f"validation-{iteration}.log").open("w") as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
  entry["exit_code"] = result.returncode
  entry["finished_at"] = datetime.now().astimezone().isoformat()
  report = output / "report.json"
  if report.is_file():
    summary = json.loads(report.read_text())
    entry["report"] = file_identity(report)
    entry["complete_matched_data"] = summary["complete_matched_data"]
    entry["full_range_standardized"] = summary["full_range_standardized"]
  save_status(args.output, state)


def save_status(output: Path, state: dict) -> None:
  """Atomically expose progress so an observer never reads a partial record."""
  state["updated_at"] = datetime.now().astimezone().isoformat()
  temporary = output / "status.tmp.json"
  write_json(temporary, state)
  temporary.replace(output / "status.json")


if __name__ == "__main__":
  main()
