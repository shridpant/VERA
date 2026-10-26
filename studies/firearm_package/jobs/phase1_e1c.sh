#!/bin/bash
# Phase 1, A16 pool: E1c scale check, InternVL3.5-38B sharded over the GPUs in CUDA_VISIBLE_DEVICES (set by the launcher line).
source "$(dirname "$0")/common.sh"
set -e  # A failed 38B format probe must stop before the ablation.
$PY4 -B run_firearm_package.py probe --arm internvl35_38b_off
run $PY4 detect internvl35_38b_off
run $PY4 detect internvl35_38b_on
