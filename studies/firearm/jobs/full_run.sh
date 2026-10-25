#!/bin/bash
# Full A/G/AG run on one GPU: preflight, then run/resume until every unit has a disposition.
# A session ends 0 (all units complete or out of attempts), 1 (incomplete, aborted or crashed),
# 124 (per-call timeout) or 2 (configuration/spec error, which stops the loop). At most 20 sessions.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-$HOME/vera_research/setup/environment/env/bin/python}
export PYTHONUTF8=1
CFG=studies/firearm/experiment_A_G_AG.json
$PY -B run_firearm_study.py validate --config "$CFG" || exit 2
for session in $(seq 1 20); do
  $PY -B run_firearm_study.py run --config "$CFG"
  code=$?
  echo "$(date -u +%FT%TZ) session $session exit $code"
  [ "$code" -eq 0 ] && break
  [ "$code" -eq 2 ] && exit 2
  sleep 30
done
$PY -B run_firearm_study.py reconcile --config "$CFG"
