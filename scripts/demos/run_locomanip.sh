#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
# The registered id, from the task package itself: it is built from whichever
# mc_rtc profile this branch carries. docs/locomanip.md#per-robot-branches
task_id="$(uv run python -c 'from mc_mjlab.tasks.locomanip import DEMO_TASK_ID; print(DEMO_TASK_ID)')"
exec uv run play "$task_id" --agent zero --viewer native "$@"
