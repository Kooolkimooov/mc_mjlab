#!/usr/bin/env bash
# mc_rtc demo. Needs the ROS workspace sourced (mc_rtc bindings + libs).
# Defaults to the viser viewer; pass "--viewer none" for the benchmark.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../.."

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

exec uv run python scripts/demos/test_mc_rtc.py --viewer viser "$@"
