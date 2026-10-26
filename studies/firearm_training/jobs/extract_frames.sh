#!/bin/bash
# Extract every split's sampled frames once (CPU), so parallel runs only read them.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=${PY:-$HOME/vera_research/setup/environment/env/bin/python}
$PY -B - <<'PYEOF'
import json
from pathlib import Path
from vera_runner.verbalized import extract_frames, sample_indices
cfg = json.loads(Path('studies/firearm_training/training1.json').read_text())
data, frames = Path(cfg['data']['dir']), Path(cfg['data']['frames'])
done = 0
for split in ('train', 'val', 'test'):
    for row in json.loads((data / f'{split}.json').read_text()):
        idx = sample_indices(row['frames_decoded'], cfg['training']['frames_per_video'])
        d = frames / row['id']
        if all((d / f'{i:06d}.jpg').exists() for i in idx):
            continue
        n = extract_frames(row['staged_path'], idx, d)
        assert n == row['frames_decoded'], (row['id'], n)
        done += 1
print('extracted', done)
PYEOF
