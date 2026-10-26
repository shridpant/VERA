#!/bin/bash
# Depth runs (docs/superpowers/specs/2026-10-26-firearm-training1-variants-plan.md, rules 3-5), called at the end of each GPU chain.
# Decides once from the candidates' validation logs, then claims the next unclaimed run of the queue (winner's seeds 2 and 3,
# then the runner-up's) until the queue is empty or the last start time has passed.
set -uo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-$HOME/vera_research/setup/environment/env/bin/python}
export PYTHONUTF8=1
C=studies/firearm_training
R=${R:-/local/datasets/sp2117/firearm_visilab/training1/runs}
CANDIDATES=${CANDIDATES:-"$C/training1b_fixed.json $C/training1c_bank.json"}
TRAIN=${TRAIN:-"bash $C/jobs/train.sh"}  # R, CANDIDATES and TRAIN are overridden only in dry runs
LAST_START=${LAST_START:-2026-10-26T10:30:00-04:00}
mkdir -p $R/claims
$PY -B run_firearm_training.py decide --candidates $CANDIDATES \
  --out $R/depth_decision.json --deadline "$LAST_START" > /dev/null || { echo "no depth run: no finished candidate"; exit 0; }
for c in $($PY -c "import json, sys; print(' '.join(json.load(open(sys.argv[1]))['queue']))" $R/depth_decision.json); do
  if [ "$(date +%s)" -ge "$(date -d "$LAST_START" +%s)" ]; then echo "no depth run starts after $LAST_START"; break; fi
  mkdir "$R/claims/$(basename "$c" .json)" 2>/dev/null || continue
  echo "$(date -u +%FT%TZ) depth run $c"
  $TRAIN "$c"
done
