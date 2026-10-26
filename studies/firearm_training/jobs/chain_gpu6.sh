#!/bin/bash
# Overnight chain on one GPU: guard the faithful specs, smoke the first method on real data, then full runs in order.
# Usage (launch.sh, absolute path): bash chain_gpuN.sh
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-$HOME/vera_research/setup/environment/env/bin/python}
export PYTHONUTF8=1
C=studies/firearm_training
R=/local/datasets/sp2117/firearm_visilab/training1/runs
for c in training1.json training1_seed2.json training1_seed3.json; do
  $PY -B run_firearm_training.py check-spec --config $C/$c || echo "WARNING: $c spec differs under this code (resume would be refused)"
done
FIRST=training1c_bank.json
NAME=$(basename "$FIRST" .json)
SMOKE=(--config $C/$FIRST --run-dir $R/smoke_$NAME --experiment-id smoke_$NAME --eval-limit 4)
$PY -B run_firearm_training.py train "${SMOKE[@]}" --max-steps 2 || { echo "smoke train failed"; exit 3; }
$PY -B run_firearm_training.py test "${SMOKE[@]}" || { echo "smoke test failed"; exit 3; }
for c in training1c_bank.json training1c_bank_random.json training1c_bank_noscore.json; do
  echo "$(date -u +%FT%TZ) starting $c"
  bash $C/jobs/train.sh $C/$c
done
