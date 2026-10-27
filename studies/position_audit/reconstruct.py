"""CPU reconstruction and comparison with the unmodified upstream script."""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import runpy
import signal
import sys
import tempfile
import time
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from verify_assets import sha


def position_weight(n):
    return np.exp(-0.5*((np.arange(n)-int(n/2))/int(n/2))**2)


def labels(sample):
    y = np.zeros(sample['length'], dtype=np.uint8)
    for start, end in zip(sample['temporal_label'][::2], sample['temporal_label'][1::2]):
        if start != -1:
            y[start:min(end, sample['length'])] = 1
    return y


def replay(feature_path, score_path, ratio):
    v = np.load(feature_path, allow_pickle=False)
    normalized = v/np.linalg.norm(v, axis=1, keepdims=True)
    similarity = normalized @ normalized.T
    n = len(v); k = int(ratio*n)
    indices = np.argsort(similarity, axis=1)[:, -k:]
    selected = np.take_along_axis(similarity, indices, axis=1)
    score = json.loads(Path(score_path).read_text())
    values = [x['score'] for x in score.values()]
    scene = []
    for ix, sim in zip(indices, selected):
        z = sim*10
        exp = np.exp(z-np.max(z)); weights = exp/np.sum(exp)
        scene.append(np.dot(weights, np.array([values[i] for i in ix])))
    # Preserve upstream's asymmetric linspace endpoints and Python rounding.
    kernel = np.exp(-np.linspace(-15//2, 15//2, 15)**2/(2*10**2)); kernel /= kernel.sum()
    smooth = np.convolve(np.asarray(scene), kernel, mode='same')
    keys = list(score)
    lengths = [16]*(n-1)+[int(score[keys[-1]]['end']-int(keys[-1]))]
    r = np.repeat(np.asarray([round(x, 1) for x in smooth]), lengths)
    return r


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True); p.add_argument('--assets', type=Path, required=True); p.add_argument('--out', type=Path, required=True); p.add_argument('--evidence', type=Path, required=True); a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CPU-only environment required'
    signal.alarm(1800)
    a.root = a.root.resolve(); a.assets = a.assets.resolve(); a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False); a.evidence.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    assert sha(a.root/'compute_auc.py') == '21e9705da3e8b04deb5855665a0c8fbc41e6baa705374f2c362b814d24b5eda8'
    data = json.loads((a.root/'Data/UCF_Eval.json').read_text())
    provenance = {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'upstream_sha256': sha(a.root/'compute_auc.py'),
                  'metadata_sha256': sha(a.root/'Data/UCF_Eval.json'), 'runner_sha256': sha(__file__), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                  'root': str(a.root), 'assets': str(a.assets), 'gpu_calls': 0}
    with threadpool_limits(limits=4):
        provenance['threadpools'] = threadpool_info()
        (a.evidence/'reconstruction_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        # No modifications to the source. Only data filesystem paths are staged.
        with tempfile.TemporaryDirectory(prefix='vera-position-reference-') as tmp:
            stage = Path(tmp); (stage/'Data').mkdir()
            (stage/'Data/UCF_Eval.json').symlink_to(a.root/'Data/UCF_Eval.json')
            (stage/'Data/vision_features').symlink_to(a.assets/'features')
            (stage/'Data/segment_level_score').symlink_to(a.assets/'scores')
            old = Path.cwd()
            try:
                os.chdir(stage)
                with (a.evidence/'upstream_stdout.log').open('w') as log, contextlib.redirect_stdout(log):
                    ref = runpy.run_path(str(a.root/'compute_auc.py'), run_name='__main__')
            finally:
                os.chdir(old)
        ref_score = np.asarray(ref['all_predict_score']); ref_y = np.asarray(ref['all_gt'], dtype=np.uint8)
        reference_auc = float(ref['roc_auc'])
        np.savez_compressed(a.out/'upstream_reference.npz', scores=ref_score, y=ref_y)
        del ref
        print(json.dumps({'stage': 'unmodified_reference_complete', 'frames': len(ref_y), 'auc': reference_auc}), flush=True)
        records = []; primary_scores = []; primary_labels = []
        for ratio in (0.15, 0.10):
            folder = a.out/f'ratio_{int(ratio*100):02d}'; folder.mkdir()
            for i, sample in enumerate(data):
                key = Path(sample['video']).stem
                r = replay(a.assets/'features'/f'{key}.npy', a.assets/'scores'/f'{key}.json', ratio)
                y = labels(sample)
                assert len(r) == len(y) == sample['length'] and np.isfinite(r).all()
                dest = folder/f'{key}.npz'; np.savez_compressed(dest, r=r, y=y)
                records.append({'ratio': ratio, 'video_id': key, 'category': Path(sample['video']).parent.name, 'frames': len(y),
                                'positive_frames': int(y.sum()), 'file': str(dest), 'sha256': sha(dest),
                                'r_array_sha256': hashlib.sha256(r.tobytes()).hexdigest(), 'y_array_sha256': hashlib.sha256(y.tobytes()).hexdigest()})
                if ratio == 0.15:
                    primary_scores.append(r*position_weight(len(r))); primary_labels.append(y)
                if i % 30 == 0:
                    print(json.dumps({'stage': 'replay', 'ratio': ratio, 'completed': i+1, 'total': len(data)}), flush=True)
        scores = np.concatenate(primary_scores); y = np.concatenate(primary_labels)
        maxdiff = float(np.max(np.abs(scores-ref_score)))
        validation = {'unmodified_reference_auc': reference_auc, 'frames': len(y), 'label_match': bool(np.array_equal(y, ref_y)),
                      'max_abs_score_difference': maxdiff, 'score_match_1e10': bool(np.allclose(scores, ref_score, atol=1e-10, rtol=1e-10)),
                      'cuda_or_torch_imported': any(k == 'torch' or k.startswith('torch.') for k in sys.modules),
                      'wall_seconds': time.perf_counter()-start, 'gpu_calls': 0, 'ratios': [0.15, 0.10]}
        (a.evidence/'reconstruction_validation.json').write_text(json.dumps(validation, indent=2)+'\n')
        (a.evidence/'reconstructed_manifest.json').write_text(json.dumps(records, indent=2)+'\n')
        assert validation['label_match'] and validation['score_match_1e10'] and not validation['cuda_or_torch_imported']
        print(json.dumps(validation), flush=True)
    signal.alarm(0)


if __name__ == '__main__':
    main()
