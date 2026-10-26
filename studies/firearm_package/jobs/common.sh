# Sourced by the package jobs: interpreters, repo root, and a retrying runner (each arm resumes from its records).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
PY_VERA=${PY_VERA:-$HOME/vera_research/setup/environment/env/bin/python}
PY4=${PY4:-/local/datasets/sp2117/envs/vlm4/bin/python}
PY5=${PY5:-/local/datasets/sp2117/envs/vlm5/bin/python}
export PYTHONUTF8=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
run() {  # run PY COMMAND ARM [extra...]: up to 3 attempts, then move on
  local py=$1 cmd=$2 arm=$3; shift 3
  for attempt in 1 2 3; do
    echo "$(date -u +%FT%TZ) $cmd $arm attempt $attempt"
    $py -B run_firearm_package.py $cmd --arm "$arm" "$@" && return 0
    sleep 20
  done
  echo "$(date -u +%FT%TZ) GAVE UP $cmd $arm"
}
