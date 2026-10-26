#!/bin/bash
# Phase 1, Ada 5: E2 recognition test (seed questions on the 302 non-test videos).
source "$(dirname "$0")/common.sh"
for a in iv2_packed iv2_frames iv2_crops; do run $PY_VERA recognise $a; done
for a in iv35_packed iv35_frames iv35_crops iv35_tiles iv35_think_frames; do run $PY4 recognise $a; done
