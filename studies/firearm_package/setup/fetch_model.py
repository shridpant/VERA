"""Download one Hugging Face model at a pinned revision into /local and verify every file against the Hub's hashes.

Usage: python fetch_model.py REPO REVISION
Writes MODEL_PROVENANCE.json next to the weights (same procedure as the InternVL3.5-8B copy of 2026-10-24).
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

REPO, REVISION = sys.argv[1], sys.argv[2]
ROOT = Path('/local/datasets/sp2117')
TARGET = ROOT / 'models' / REPO / REVISION
SKIP_PREFIXES = ('examples/', 'assets/', 'figures/')  # sample media only
os.environ['HF_HOME'] = str(ROOT / 'hf_home')

from huggingface_hub import snapshot_download  # noqa: E402  (after HF_HOME)


def digest(path, algorithm, prefix=b''):
    h = hashlib.new(algorithm)
    h.update(prefix)
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true', timeout=60) as r:
    meta = json.load(r)
assert meta['sha'] == REVISION, meta['sha']
siblings = [s for s in meta['siblings'] if not s['rfilename'].startswith(SKIP_PREFIXES)]
total = sum(s.get('lfs', {}).get('size', s.get('size', 0)) for s in siblings)
print(f'{REPO}@{REVISION}: {len(siblings)} files, {total / 2**30:.2f} GiB', flush=True)
TARGET.mkdir(parents=True, exist_ok=True)
snapshot_download(REPO, revision=REVISION, local_dir=str(TARGET), ignore_patterns=[p + '*' for p in SKIP_PREFIXES])
files = {}
for s in siblings:
    path = TARGET / s['rfilename']
    size = path.stat().st_size
    if 'lfs' in s:
        expected = {'bytes': s['lfs']['size'], 'sha256': s['lfs']['sha256']}
        assert size == expected['bytes'] and digest(path, 'sha256') == expected['sha256'], f'mismatch {s["rfilename"]}'
    else:
        expected = {'bytes': s['size'], 'git_blob_sha1': s['blobId']}
        assert size == expected['bytes'] and digest(path, 'sha1', b'blob %d\0' % size) == expected['git_blob_sha1'], f'mismatch {s["rfilename"]}'
    files[s['rfilename']] = expected
provenance = {'repo': REPO, 'revision': REVISION, 'source': f'https://huggingface.co/{REPO}/tree/{REVISION}',
              'license': (meta.get('cardData') or {}).get('license'),
              'skipped': sorted(s['rfilename'] for s in meta['siblings'] if s['rfilename'].startswith(SKIP_PREFIXES)),
              'verified_at': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'), 'files': files}
(TARGET / 'MODEL_PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n')
print('DONE', TARGET, f'{total / 2**30:.2f} GiB, {len(files)} files verified', flush=True)
