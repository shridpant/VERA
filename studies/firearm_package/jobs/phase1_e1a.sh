#!/bin/bash
# Phase 1, Ada 4: E1a, the latest InternVL as detector (thinking off, on, on repeated) and the InternVL2 single-frame baseline.
source "$(dirname "$0")/common.sh"
run $PY4 detect internvl35_8b_off
run $PY_VERA detect internvl2_8b_off
run $PY4 detect internvl35_8b_on
run $PY4 detect internvl35_8b_on_r2
