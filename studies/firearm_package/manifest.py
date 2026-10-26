"""E1 frame manifest (package plan, Module 1): VERA-sampled frames of the non-test videos, never the test split.

Firearm videos give up to 4 of their 8 sampled frames whose audit status is positive (a box exists); No_Gun
videos give 2 frames, for false positives. Choices and the nested subsets (300, 100) are seeded, stratified by
category, and stable under resume.
"""
import json
from pathlib import Path
from vera_runner.verbalized import sample_indices
from vera_runner.verbalized_variants import rng

SPLITS = ('train', 'val')


def build(cfg):
    data, seed, m = Path(cfg['data_dir']), cfg['seed'], cfg['manifest']
    frames = []
    for split in SPLITS:
        rows = json.loads((data / f'{split}.json').read_text())
        boxes = json.loads((data / f'boxes_{split}.json').read_text())
        for r in rows:
            idx = sample_indices(r['frames_decoded'], 8)
            b = boxes[r['id']]
            if r['category'] == 'No_Gun':
                slots, cap = list(range(8)), m['nogun_frames_per_video']
            else:
                slots = [k for k, i in enumerate(idx) if b['statuses'][i] == 'positive' and str(i) in b['boxes']]
                cap = m['firearm_frames_per_video']
            pick = sorted(rng(seed, 'e1', r['id']).permutation(len(slots))[:cap].tolist())
            for p in pick:
                k = slots[p]
                frames.append({'frame_id': f"{r['id']}#{idx[k]}", 'video': r['id'], 'split': split, 'category': r['category'],
                               'action_group': r['action_group'], 'recording_group': r['recording_group'], 'camera': r['camera'],
                               'slot': k, 'frame_index': idx[k], 'width': r['width'], 'height': r['height'],
                               'gt': b['boxes'].get(str(idx[k]), []) if r['category'] != 'No_Gun' else []})
    frames.sort(key=lambda f: f['frame_id'])
    by_cat = {}
    for f in frames:
        by_cat.setdefault(f['category'], []).append(f)
    order = {c: [fs[i] for i in rng(seed, 'subset', c).permutation(len(fs))] for c, fs in by_cat.items()}
    for name in ('subset300', 'subset100'):
        n = m[name]
        quota = {c: round(n * len(fs) / len(frames)) for c, fs in by_cat.items()}
        quota[max(quota, key=quota.get)] += n - sum(quota.values())
        chosen = {f['frame_id'] for c, fs in order.items() for f in fs[:quota[c]]}  # nested: 100 is a prefix of 300
        for f in frames:
            f[name] = f['frame_id'] in chosen
    return frames


def select(frames, which):
    return frames if which == 'all' else [f for f in frames if f[which]]


def write(cfg, path):
    frames = build(cfg)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(frames, indent=0) + '\n')
    count = lambda fs: {c: sum(f['category'] == c for f in fs) for c in sorted({f['category'] for f in frames})}
    return {'frames': len(frames), 'by_category': count(frames), 'subset300': count(select(frames, 'subset300')),
            'subset100': count(select(frames, 'subset100')), 'gt_boxes': sum(len(f['gt']) for f in frames)}
