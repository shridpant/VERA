#!/bin/bash
# Phase 0 box-format probes on one Ada: every family, thinking off and on (6 frames each, both InternVL prompts).
source "$(dirname "$0")/common.sh"
for a in internvl35_8b_off internvl35_8b_on; do $PY4 -B run_firearm_package.py probe --arm $a; done
$PY_VERA -B run_firearm_package.py probe --arm internvl2_8b_off
for a in qwen3vl_8b_instruct qwen3vl_8b_thinking qwen35_9b_off qwen35_9b_on; do $PY5 -B run_firearm_package.py probe --arm $a; done
