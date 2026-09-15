#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."
task_id="$(uv run python -c '
from mc_mjlab import MC_RTC_YAML_PATH
from utils.task_naming import get_task_name
print(get_task_name("locomanip", "position", MC_RTC_YAML_PATH.with_name("mc_rtc_locomanip.yaml")))
')"
exec uv run play "$task_id" --agent zero --viewer viser "$@"
