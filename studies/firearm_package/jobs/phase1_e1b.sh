#!/bin/bash
# Phase 1, Ada 6: E1b, the Qwen comparators (Qwen3-VL Instruct/Thinking, Qwen3.5 thinking off/on).
source "$(dirname "$0")/common.sh"
run $PY5 detect qwen3vl_8b_instruct
run $PY5 detect qwen35_9b_off
run $PY5 detect qwen3vl_8b_thinking
run $PY5 detect qwen35_9b_on
run $PY5 detect qwen3vl_8b_thinking_r2
run $PY5 detect qwen35_9b_on_r2
