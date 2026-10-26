#!/bin/bash
# Phase 2 on one GPU: the 8 remaining arm runs (Training 1's own run is T1 seed 20261026), each trained
# (resumed until selected.json exists, at most 6 sessions) and then evaluated on the test split.
# Order: G/P for seed 20261026 first (paired with Training 1), then seeds 20261027 and 20261028.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-$HOME/vera_research/setup/environment/env/bin/python}
export PYTHONUTF8=1
R=/local/datasets/sp2117/firearm_visilab/grounding1
T1RUN=/local/datasets/sp2117/firearm_visilab/training1/runs/firearm_vera_training1_v1
train_and_evaluate() {
  arm=$1; seed=$2; dir=$R/arms/${arm}_s${seed}
  for session in 1 2 3 4 5 6; do
    [ -f "$dir/selected.json" ] && break
    $PY -B run_firearm_grounding.py train-arm --arm "$arm" --seed "$seed"
    echo "$(date -u +%FT%TZ) $arm s$seed train session $session exit $?"
    sleep 10
  done
  [ -f "$dir/selected.json" ] || { echo "$arm s$seed did not finish training" >&2; return 1; }
  $PY -B run_firearm_grounding.py evaluate --run-dir "$dir"
  echo "$(date -u +%FT%TZ) $arm s$seed evaluated exit $?"
}
adopt_and_evaluate_t1() {
  dir=$R/arms/T1_s20261026
  [ -f "$dir/selected.json" ] || $PY -B run_firearm_grounding.py adopt-t1 --training-run "$T1RUN" || return 1
  $PY -B run_firearm_grounding.py evaluate --run-dir "$dir"
  echo "$(date -u +%FT%TZ) T1 s20261026 (Training 1) evaluated exit $?"
}
train_and_evaluate G 20261026
train_and_evaluate P 20261026
adopt_and_evaluate_t1
for seed in 20261027 20261028; do
  for arm in T1 G P; do train_and_evaluate "$arm" "$seed"; done
done
adopt_and_evaluate_t1   # in case Training 1 was still running at the first attempt
