"""CPU-only release extraction and identity/alignment gate; no performance metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import zipfile
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def extract(archive, suffix, dest):
    dest.mkdir(parents=True, exist_ok=False)
    members = []
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None, 'ZIP CRC failure'
        for info in z.infolist():
            assert not Path(info.filename).is_absolute() and '..' not in Path(info.filename).parts
            assert not stat.S_ISLNK(info.external_attr >> 16)
            if not info.filename.endswith(suffix):
                continue
            target = dest/Path(info.filename).name
            assert not target.exists(), 'Duplicate basename'
            target.write_bytes(z.read(info))
            members.append({'member': info.filename, 'file': target.name, 'bytes': info.file_size, 'crc32': f'{info.CRC:08x}', 'sha256': sha(target)})
    return members


def main():
    p = argparse.ArgumentParser(); p.add_argument('--assets', type=Path, required=True); p.add_argument('--root', type=Path, required=True); p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    featurezip = a.assets/'ucf_crime_vision_features.zip'; scorezip = a.assets/'VERA_prediction_scores_UCF_Crime.zip'; meta = a.root/'Data/UCF_Eval.json'
    assert featurezip.stat().st_size == 309807835
    assert sha(scorezip) == 'b7457cee812510ec2331c24cb5da451327c9500152e2437839d31af29166a3b0'
    assert sha(meta) == 'f3857cddd623f767e7b86bd50a4aeb122445fc6e241ddd7130b7e3ec0c4d8b9f'
    fm = extract(featurezip, '.npy', a.assets/'features'); sm = extract(scorezip, '.json', a.assets/'scores')
    d = json.loads(meta.read_text()); ids = {Path(x['video']).stem for x in d}
    assert {Path(x['file']).stem for x in fm} == ids
    assert {Path(x['file']).stem for x in sm} == ids
    shapes, expanded = {}, 0
    for x in d:
        key = Path(x['video']).stem
        r = json.loads((a.assets/'scores'/f'{key}.json').read_text())
        indices = list(map(int, r))
        assert indices == list(range(0, x['length'], 16))
        v = np.load(a.assets/'features'/f'{key}.npy', allow_pickle=False)
        assert v.ndim == 2 and v.shape == (len(r), 1024), (key, v.shape, len(r))
        assert np.isfinite(v).all() and (np.linalg.norm(v, axis=1) > 0).all()
        end = int(r[str(indices[-1])]['end'])
        assert end == x['length'], (key, end, x['length'])
        shapes[key] = {'shape': list(v.shape), 'dtype': str(v.dtype), 'frames': end}; expanded += end
    road = a.assets/'features/RoadAccidents127_x264.npy'
    assert sha(road) == 'f0c2470ccdae5e6d8091dd3e41fe281f02243f37b80bfbf71fd0e0ed668eb657'
    out = {'complete': True, 'gpu_calls': 0, 'feature_archive_sha256': sha(featurezip), 'feature_archive_bytes': featurezip.stat().st_size,
           'score_archive_sha256': sha(scorezip), 'metadata_sha256': sha(meta), 'feature_members': fm, 'score_members': sm,
           'shapes': shapes, 'videos': len(ids), 'total_frames': expanded, 'source': 'Authors release linked in original VERA README; direct public Google Drive download after connector size-limit rejection',
           'authenticity_limit': 'Archive digest records received bytes; all CRCs and shapes checked; one feature independently matches previously verified artifact.'}
    a.out.write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({k: out[k] for k in ['complete', 'videos', 'total_frames', 'feature_archive_sha256', 'gpu_calls']}))


if __name__ == '__main__':
    main()
