#!/usr/bin/env bash
# mc_rtc demo. Needs the ROS workspace sourced (mc_rtc bindings + libs).
#
# Runs the zero-residual task through mjlab's `play --agent zero`: the robot
# tracks raw mc_rtc output, so a healthy run holds a steady root height. The
# task itself lives in src/mc_mjlab/tasks/zero_residual/; everything here is
# launcher. Extra arguments are forwarded to `play`, e.g.
#   run_test_mc_rtc.sh --num-envs 8
#   run_test_mc_rtc.sh --viewer native
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."

# The id embeds the robot and controller from etc/mc_rtc.yaml. Resolved through
# mc_mjlab.utils.task_naming, which imports no mjlab, so this costs milliseconds
# rather than a full torch/warp import.
control="${MC_MJLAB_CONTROL:-position}"
task_id="$(uv run python -c "
from mc_mjlab.utils.task_naming import get_task_name
print(get_task_name('zero_residual', '$control'))
")"

# Effective --viewer value ("--viewer X" or "--viewer=X"; viser by default,
# injected below).
viewer="viser"
argstr=" $* "
if [[ "$argstr" == *" --viewer="* ]]; then
  viewer="${argstr##*--viewer=}"; viewer="${viewer%% *}"
elif [[ "$argstr" == *" --viewer "* ]]; then
  viewer="${argstr##*--viewer }"; viewer="${viewer%% *}"
fi

# Open the viser UI once the server answers HTTP (a bare TCP probe would make
# viser log a bad-connection error).
if [[ "$viewer" == "viser" ]] && command -v xdg-open curl >/dev/null 2>&1; then
  (
    set +e
    for _ in $(seq 1 120); do
      curl -sf -o /dev/null http://localhost:8080 && break
      sleep 0.5
    done
    xdg-open http://localhost:8080 >/dev/null 2>&1
  ) &
fi

echo "[mc_rtc] $task_id"
exec uv run play "$task_id" --agent zero --viewer viser "$@"
