#!/bin/bash
# Pilot on one GPU: preflight, a bounded A/G/AG run, the slot-interpretability check and an A
# preservation check (run_vera.py on the same windows must give the same raw text as study A).
# Usage (on Oz, from launch.sh): bash studies/firearm/jobs/pilot.sh PILOT_ID
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-$HOME/vera_research/setup/environment/env/bin/python}
export PYTHONUTF8=1
ID=$1
CFG=studies/firearm/experiment_A_G_AG.json
D=/local/datasets/sp2117/firearm_visilab
OUT=$D/runs/$ID
# One video per category x camera, drawn with the study seed; the first two windows of each.
VIDEOS=$($PY -I - "$D/inventory.json" <<'PYEOF'
import json, random, sys
rows = sorted(json.load(open(sys.argv[1])), key=lambda r: r['id'])
rng = random.Random(20261025)
picked = [rng.choice([r for r in rows if r['category'] == c and r['camera'] == k])['id']
          for c in ('Handgun', 'Machine_Gun', 'No_Gun') for k in ('C1', 'C2')]
print(','.join(picked))
PYEOF
)
echo "pilot videos: $VIDEOS"
COMMON=(--config "$CFG" --run-dir "$OUT" --study-id "$ID" --videos "$VIDEOS" --max-windows 2)
$PY -B run_firearm_study.py validate "${COMMON[@]}"
$PY -B run_firearm_study.py run "${COMMON[@]}"
HAND=$(echo "$VIDEOS" | tr ',' '\n' | grep '^Handgun__' | sed 's/$/:16/' | paste -sd, -)
$PY -B studies/firearm/slot_check.py --config "$CFG" --run-dir "$OUT" --windows "$HAND" --keep 0,3,7 --out "$OUT/slot_check.json"
# Condition A through the unmodified runner on the same windows.
$PY -I - "$CFG" "$OUT" "$VIDEOS" <<'PYEOF'
import json, sys
cfg, out, videos = json.load(open(sys.argv[1]))['runner'], sys.argv[2], sys.argv[3].split(',')
cfg['selection'] = {'video_ids': videos, 'max_videos': None, 'max_windows_per_video': 2}
cfg['output_root'] = out + '/runner_check'
open(out + '/runner_A_check.json', 'w').write(json.dumps(cfg, indent=2))
PYEOF
$PY -B run_vera.py --config "$OUT/runner_A_check.json"
$PY -I - "$OUT" <<'PYEOF'
import glob, json, sys
out = sys.argv[1]
study = {(r['video'], r['anchor']): r for r in map(json.loads, open(out + '/records/A.jsonl')) if r['status'] == 'complete'}
runner = {}
for f in glob.glob(out + '/runner_check/run 1/output/*/raw_responses.jsonl'):
    for r in map(json.loads, open(f)):
        runner[(f.split('/')[-2], r['anchor'])] = r
same = {f'{k[0]}:{k[1]}': (study[k]['raw_response'] == runner[k]['raw_response'], study[k]['rgb_sha256'] == runner[k]['rgb_sha256'],
                          study[k]['strict_label'] == runner[k]['strict_label']) for k in sorted(runner)}
result = {'windows': len(runner), 'study_windows': len(study), 'identical_text_inputs_label': same,
          'all_identical': len(runner) == len(study) and all(all(v) for v in same.values())}
json.dump(result, open(out + '/a_preservation.json', 'w'), indent=1)
print(json.dumps(result, indent=1))
PYEOF
