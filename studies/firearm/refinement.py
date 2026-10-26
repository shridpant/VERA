"""Offline refinement of A and AG labels with VERA's original procedure and identical features.

For each video and condition (A, AG; G has no anomaly label), the strict labels of every window
go through vera_runner.refine.refine unchanged, with the same ImageBind feature file for both
conditions. A video is excluded, with every reason that applies, when:
  incomplete_inference  a window has no complete record
  invalid_labels        a window's strict label is missing (invalid or truncated); never filled
  missing_features      no feature file or a different row count than anchors
  too_few_windows       fewer windows than the smoothing kernel (15) or no neighbour at 15 %
Clips are never padded and the smoothing is never changed. Initial labels stay in the records.
"""
import csv
import json
from pathlib import Path
import numpy as np
from vera_runner.data import sha256
from vera_runner.refine import refine

CONDITIONS = ('A', 'AG')


def reasons_for(records, features, cfg):
    reasons = []
    if any(r is None for r in records):
        reasons.append('incomplete_inference')
    elif any(r['label'] not in (0, 1) for r in records):
        reasons.append('invalid_labels')
    if features is None or features.shape[0] != len(records):
        reasons.append('missing_features')
    if len(records) < cfg['smoothing_size'] or int(cfg['neighbor_fraction'] * len(records)) < 1:
        reasons.append('too_few_windows')
    return reasons


def run_refinement(study, spec, final, features_dir, out_dir):
    out_dir = Path(out_dir)
    cfg, stride = study['runner']['refinement'], study['runner']['sampling']['stride']
    rows, used = [], {}
    for v in spec['videos']:
        npy = Path(features_dir) / f"{v['id']}.npy"
        features = np.load(npy) if npy.exists() else None
        used[v['id']] = sha256(npy) if features is not None else None
        for c in CONDITIONS:
            records = [final[c].get((v['id'], a)) for a in v['anchors']]
            reasons = reasons_for(records, features, cfg)
            row = {'video': v['id'], 'category': v['category'], 'condition': c, 'windows': len(records),
                   'invalid_labels': sum(1 for r in records if r is not None and r['label'] not in (0, 1)),
                   'feature_sha256': used[v['id']], 'refined': not reasons, 'reasons': ';'.join(reasons)}
            if not reasons:
                refined = refine([{'strict_label': r['label'], 'anchor': r['anchor'], 'end': r['end']} for r in records],
                                 features, v['frames'], stride, cfg)
                target = out_dir / c / v['id']
                target.mkdir(parents=True, exist_ok=True)
                for key, values in refined.items():
                    np.save(target / f'{key}.npy', values)
                row.update(mean_frame_score=float(refined['frame_scores'].mean()), max_frame_score=float(refined['frame_scores'].max()),
                           mean_temporal_score=float(refined['temporal'].mean()))
            rows.append(row)
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ['video', 'category', 'condition', 'windows', 'invalid_labels', 'refined', 'reasons', 'feature_sha256',
              'mean_frame_score', 'max_frame_score', 'mean_temporal_score']
    with (out_dir / 'refinement.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / 'feature_hashes.json').write_text(json.dumps(used, indent=1) + '\n')
    return rows
