#!/bin/bash
# Phase 0 downloads (user-approved 2026-10-26): pinned revisions, verified, into /local/datasets/sp2117/models.
set -uo pipefail
PY=${PY:-$HOME/vera_research/setup/environment/env/bin/python}
HERE="$(cd "$(dirname "$0")" && pwd)"
while read -r repo rev; do
  [ -z "$repo" ] && continue
  if [ -f "/local/datasets/sp2117/models/$repo/$rev/MODEL_PROVENANCE.json" ]; then echo "have $repo"; continue; fi
  $PY "$HERE/fetch_model.py" "$repo" "$rev" || echo "FAILED $repo"
done <<'LIST'
Qwen/Qwen3-VL-8B-Instruct 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
Qwen/Qwen3-VL-8B-Thinking 92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b
Qwen/Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a
OpenGVLab/InternVL3_5-38B de99855be3642cd44fe97c9b72d70e5ce2c07f69
LIST
