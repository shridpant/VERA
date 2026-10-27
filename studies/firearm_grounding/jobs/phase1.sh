#!/bin/bash
# Phase 1: seed questions on all splits (M1-M4, resumable), then the pre-registered gate.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-$HOME/vera_research/setup/environment/env/bin/python}
export PYTHONUTF8=1
R=/local/datasets/sp2117/firearm_visilab/grounding1
for attempt in 1 2 3; do
  $PY -B run_firearm_grounding.py measure --splits train,val,test --out "$R/phase1_seed_measurements.jsonl" && break
  echo "measure attempt $attempt failed; resuming" >&2; sleep 20
done
# The gate itself refuses (exit 2) when any video has no record.
$PY -B run_firearm_grounding.py gate --measurements "$R/phase1_seed_measurements.jsonl"
