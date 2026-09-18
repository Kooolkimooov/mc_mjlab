#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
robot="${MC_MJLAB_ROBOT:-HRP5P}"
# The registered id, from the task package itself, rather than reassembled here.
# docs/locomanip.md#per-robot-controller-config
task_id="$(uv run python -c "
from mc_mjlab.tasks.locomanip import DEMO_TASK_IDS
print(DEMO_TASK_IDS['$robot'])")"
exec uv run play "$task_id" --agent zero --viewer native "$@"
