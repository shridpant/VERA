#!/bin/bash
# Bounded real-model check of Training 1: 2 training steps, validation after each, 4 val and 4 test videos.
# Usage (on Oz, from launch.sh with an absolute path): bash studies/firearm_training/jobs/smoke.sh SMOKE_ID
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-$HOME/vera_research/setup/environment/env/bin/python}
export PYTHONUTF8=1
CFG=studies/firearm_training/training1.json
R=/local/datasets/sp2117/firearm_visilab/training1/runs/$1
COMMON=(--config "$CFG" --run-dir "$R" --experiment-id "$1" --val-check-interval 1 --eval-limit 4)
$PY -B run_firearm_training.py train "${COMMON[@]}" --max-steps 2
$PY -B run_firearm_training.py test "${COMMON[@]}"
