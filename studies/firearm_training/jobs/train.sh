#!/bin/bash
# Training 1 on one GPU: train (resumed until complete, at most 10 sessions), then test.
# Usage: bash train.sh [CONFIG]  (default studies/firearm_training/training1.json)
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-$HOME/vera_research/setup/environment/env/bin/python}
export PYTHONUTF8=1
CFG=${1:-studies/firearm_training/training1.json}
for session in $(seq 1 10); do
  out=$($PY -B run_firearm_training.py train --config "$CFG" 2>&1 | tee /dev/stderr | tail -1)
  echo "$(date -u +%FT%TZ) train session $session: $out"
  echo "$out" | grep -q '"status": "complete"' && break
  sleep 30
done
$PY -B run_firearm_training.py test --config "$CFG"
