"""Fixed counterfactuals and paired whole-video bootstrap, CPU only."""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from metrics import CachedAUC, gate
from reconstruct import position_weight

EXCLUDE = {'Arson011_x264', 'Arson016_x264', 'Explosion033_x264', 'Fighting003_x264', 'Shooting015_x264'}


def metric(y, s):
    return {'auc': float(roc_auc_score(y, s)), 'ap': float(average_precision_score(y, s))} if len(np.unique(y)) == 2 else {'auc': None, 'ap': None}


def cohort(records, fraction, folder, diagnostic):
    t0 = time.perf_counter(); n = len(records)
    parts = []
    for v, record in enumerate(records):
        z = np.load(record['file'], allow_pickle=False)
        r, y = z['r'], z['y']; assert len(r) == len(y)
        parts.append((r, y))
    video = np.concatenate([np.full(len(y), i, dtype=np.int16) for i, (_, y) in enumerate(parts)])
    keys = [x['video_id'] for x in records]
    point = {'original': [], 'no_position': [], 'prior_only': []}; caches = []; pervideo = []; categories = []
    histogram = np.zeros((20, 2), dtype=np.int64)
    for r, y in parts:
        bins = np.minimum(np.arange(len(y))*20//len(y), 19)
        histogram[:, 0] += np.bincount(bins, minlength=20)
        histogram[:, 1] += np.bincount(bins, weights=y, minlength=20).astype(np.int64)
    for offset in range(5):
        ys = []; ss = {'original': [], 'no_position': [], 'prior_only': []}
        for i, (r, y) in enumerate(parts):
            k = offset*len(y)//5; yr = np.roll(y, k); rr = np.roll(r, k); w = position_weight(len(y))
            ys.append(yr); ss['original'].append(w*rr); ss['no_position'].append(rr); ss['prior_only'].append(w)
            if diagnostic:
                for method in ss:
                    pervideo.append({'video_id': keys[i], 'offset_fraction': offset/5, 'method': method, **metric(yr, ss[method][-1])})
        y = np.concatenate(ys)
        for method in ss:
            s = np.concatenate(ss[method]); point[method].append(metric(y, s))
            if method == 'original':
                caches.append(CachedAUC(y, s, video))
            elif method == 'no_position' and offset == 0:
                control = CachedAUC(y, s, video)
            if diagnostic:
                for category in sorted({x['category'] for x in records if x['positive_frames'] > 0}):
                    ids = [i for i, x in enumerate(records) if x['category'] == category or x['positive_frames'] == 0]
                    use = np.isin(video, ids)
                    categories.append({'category': category, 'normal_videos_included': True, 'offset_fraction': offset/5, 'method': method, **metric(y[use], s[use])})
        print(json.dumps({'stage': 'point_metrics', 'ratio': fraction, 'videos': n, 'offset': offset/5}), flush=True)
    assert np.ptp([x['auc'] for x in point['no_position']]) < 1e-12
    assert np.ptp([x['ap'] for x in point['no_position']]) < 1e-12
    one = np.ones(n, dtype=np.int64)
    for cache, p in zip(caches, point['original']):
        assert abs(cache.auc(one)-p['auc']) < 1e-12
    assert abs(control.auc(one)-point['no_position'][0]['auc']) < 1e-12
    rng = np.random.default_rng(20261027); samples = []
    for b in range(1000):
        count = np.bincount(rng.integers(0, n, size=n), minlength=n)
        original = [cache.auc(count) for cache in caches]; nopos = control.auc(count)
        assert np.isfinite(original).all() and np.isfinite(nopos)
        samples.append({'replicate': b, 'worst_benefit': nopos-min(original), 'unshifted_cost': original[0]-nopos, 'spread': max(original)-min(original)})
        if b % 100 == 0:
            print(json.dumps({'stage': 'bootstrap', 'ratio': fraction, 'videos': n, 'completed': b}), flush=True)
    ci = np.percentile([x['worst_benefit'] for x in samples], [2.5, 97.5]).tolist()
    decision = gate([x['auc'] for x in point['original']], [x['auc'] for x in point['no_position']], ci[0])
    out = {'videos': n, 'frames': len(video), 'offsets': [0, .2, .4, .6, .8], 'point': point,
           'worst_benefit_ci95': ci, 'gate': decision, 'bootstrap_replicates': 1000, 'bootstrap_seed': 20261027,
           'wall_seconds': time.perf_counter()-t0, 'label_position_bins': histogram.tolist()}
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'metrics.json').write_text(json.dumps(out, indent=2)+'\n')
    (folder/'bootstrap.json').write_text(json.dumps(samples, indent=2)+'\n')
    if diagnostic:
        (folder/'per_video.json').write_text(json.dumps(pervideo, indent=2)+'\n')
        (folder/'per_category.json').write_text(json.dumps(categories, indent=2)+'\n')
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--manifest', type=Path, required=True); p.add_argument('--out', type=Path, required=True); a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    signal.alarm(1800); start = time.perf_counter()
    records = json.loads(a.manifest.read_text()); a.out.mkdir(parents=True, exist_ok=False)
    summary = {'gpu_calls': 0, 'cohorts': {}}
    with threadpool_limits(limits=4):
        for ratio in (0.15, 0.10):
            selected = [r for r in records if r['ratio'] == ratio]
            assert len(selected) == 290
            for excluded in (False, True):
                subset = [r for r in selected if not excluded or r['video_id'] not in EXCLUDE]
                key = f'ratio_{int(ratio*100):02d}_'+('exclude5' if excluded else 'all')
                result = cohort(subset, ratio, a.out/key, diagnostic=ratio == .15 and not excluded)
                summary['cohorts'][key] = result
                (a.out/'progress.json').write_text(json.dumps(summary, indent=2)+'\n')
    primary = summary['cohorts']['ratio_15_all']; sensitivity = summary['cohorts']['ratio_15_exclude5']
    summary['primary_success'] = primary['gate']['pass'] and sensitivity['gate']['pass']
    summary['wall_seconds'] = time.perf_counter()-start
    summary['interpretation'] = 'Provisional practical robustness finding only if all agreed gates pass; no novel removal algorithm or end-to-end video-edit claim.'
    (a.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({'primary_success': summary['primary_success'], 'wall_seconds': summary['wall_seconds']}), flush=True)
    signal.alarm(0)


if __name__ == '__main__':
    main()
