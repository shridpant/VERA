#!/bin/bash
# Bounded prompt-format probe on pilot windows: bash studies/firearm/jobs/probe.sh PILOT_RUN_ID OUT_NAME PROMPT_DIRS
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-$HOME/vera_research/setup/environment/env/bin/python}
export PYTHONUTF8=1
RUN=/local/datasets/sp2117/firearm_visilab/runs/$1
WINDOWS=Handgun__PAH7_C1_P1_V1_HB_2:16,Handgun__PAH9_C2_P3_V1_HB_2:16,Machine_Gun__PAW8_C1_P2_V1_HB_2:16,Machine_Gun__PAW5_C2_P3_V1_HB_2:16,No_Gun__N7_C2_P3_V1_HB_1:16
$PY -B studies/firearm/prompt_probe.py --config studies/firearm/experiment_A_G_AG.json --run-dir "$RUN" \
    --prompt-dirs "$3" --windows "$WINDOWS" --out "$RUN/$2.json"
