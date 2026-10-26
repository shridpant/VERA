"""One-shot dependency job: analyze only after every Phase 1 job and unit completes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

repo = Path('/local/datasets/sp2117/firearm_visilab/code/VERA-package-runs')
root = Path('/local/datasets/sp2117/firearm_visilab/package')
state = root / 'analysis/completion.json'
state.parent.mkdir(parents=True, exist_ok=True)
jobs = [Path(p) for p in sys.argv[1:]]

def save(status, **extra):
    state.write_text(json.dumps({'status': status, 'jobs': [str(j) for j in jobs], **extra}, indent=2) + '\n')

try:
    if len(jobs) != 4:
        raise ValueError('Expected four Phase 1 dependency jobs')
    save('waiting')
    while True:
        pending = []
        for job in jobs:
            exit_file = job / 'exit_status'
            if exit_file.exists():
                if int(exit_file.read_text()) != 0:
                    raise RuntimeError(f'Job failed: {job}')
            else:
                pid = int((job / 'pid').read_text())
                os.kill(pid, 0)
                pending.append(str(job))
        if not pending:
            break
        time.sleep(30)
    manifest = json.loads((root / 'e1_manifest.json').read_text())
    cfg = json.loads((repo / 'studies/firearm_package/package.json').read_text())
    video_ids = {r['id'] for split in ('train', 'val') for r in json.loads((Path(cfg['data_dir']) / f'{split}.json').read_text())}
    errors, counts = [], {}
    for experiment, key in [('e1', 'frame_id'), ('e2', 'video')]:
        arms = json.loads((repo / f'studies/firearm_package/{experiment}_arms.json').read_text())
        for arm, spec in arms.items():
            path = root / experiment / arm / 'records.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
            successful = {r[key] for r in rows if r['status'] != 'failed'}
            expected = ({f['frame_id'] for f in manifest if spec['frames'] == 'all' or f[spec['frames']]} if experiment == 'e1' else video_ids)
            counts[f'{experiment}/{arm}'] = {'successful': len(successful), 'expected': len(expected)}
            if successful != expected:
                errors.append({'arm': arm, 'missing': sorted(expected - successful), 'unexpected': sorted(successful - expected)})
    if errors:
        save('incomplete', counts=counts, errors=errors)
        raise RuntimeError('Missing or unexpected records; gates were not run')
    save('analyzing', counts=counts)
    subprocess.run([sys.executable, '-B', 'run_firearm_package.py', 'analyze1'], cwd=repo, check=True)
    save('complete', counts=counts, result=str(root / 'analysis/phase1.json'))
except Exception as exc:
    if not state.exists() or json.loads(state.read_text())['status'] != 'incomplete':
        save('failed', error=repr(exc))
    raise
