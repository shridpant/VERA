#!/bin/bash
# Real-model smoke of Phase 2 before the full job: 2 training steps of G and P (validation after each step on
# 4 validation videos), selection, and a 3-video test evaluation, in a separate root (grounding1_smoke) that
# copies Phase 1's gate and lift pool. Nothing here is read by the study.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-$HOME/vera_research/setup/environment/env/bin/python}
export PYTHONUTF8=1
G=/local/datasets/sp2117/firearm_visilab/grounding1
S=/local/datasets/sp2117/firearm_visilab/grounding1_smoke
rm -rf "$S"; mkdir -p "$S"
cp "$G/gate.json" "$G/phase1_pool.json" "$S/"
$PY - "$S" <<'EOF'
import json, sys
cfg = json.load(open('studies/firearm_grounding/grounding.json'))
cfg['root'] = sys.argv[1]
json.dump(cfg, open(sys.argv[1] + '/grounding.json', 'w'), indent=1)
EOF
for arm in G P; do
  $PY -B run_firearm_grounding.py train-arm --config "$S/grounding.json" --arm $arm --seed 1 --max-steps 2 --eval-limit 4 --val-check-interval 1
done
for arm in G P; do
  # --max-steps stops before the last epoch, so selection is run here directly
  $PY -B - "$S" $arm <<'EOF'
import json, sys
from pathlib import Path
from vera_runner.ledger import read_jsonl
from studies.firearm_grounding.arms import GroundedRun
run = Path(sys.argv[1]) / 'arms' / f'{sys.argv[2]}_s1'
steps = read_jsonl(run / 'steps.jsonl')[0]
vals = read_jsonl(run / 'validations.jsonl')[0]
print(sys.argv[2], 'steps', [s['step'] for s in steps], 'statuses', [s['status'] for s in steps])
print('grounding block step 1:\n' + steps[0].get('grounding_block', 'MISSING'))
print('validations', [(v['step'], v['reference_only'], v['metrics'].get('category_accuracy'), v['metrics'].get('pooled_lift'),
                       v['metrics'].get('videos_with_lift')) for v in vals])
self = type('R', (), {'dir': run, 'arm': sys.argv[2]})()
best = GroundedRun.select(self)
(run / 'selected.json').write_text(json.dumps({'step': best['step'], 'questions': best['questions'], 'metrics': best['metrics']}) + '\n')
print('selected step', best['step'])
EOF
  $PY -B run_firearm_grounding.py evaluate --config "$S/grounding.json" --run-dir "$S/arms/${arm}_s1" --limit 3
done
echo "smoke done"
