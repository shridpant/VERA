"""Metadata-only XD feasibility checks. Does not compute detector performance."""
import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--scores', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    meta = json.loads((a.repo/'Data/xd_test_800.json').read_text())
    annotations = json.loads((a.repo/'Data/xd_annotations.json').read_text())
    gt = json.loads((a.repo/'Data/name_gt_pair.json').read_text())
    ids = {x['video'] for x in meta}
    checks = {'score_anchor_grid_mismatch': [], 'score_end_length_mismatch': [],
              'ap_trimmed_length_mismatch': [], 'nonbinary_ap_labels': [],
              'annotation_test_metadata_mismatch': []}
    by_id = {x['video']: x for x in annotations}
    with zipfile.ZipFile(a.scores) as z:
        bad_crc = z.testzip()
        members = {Path(n).stem: n for n in z.namelist() if n.endswith('.json')}
        for sample in meta:
            key = sample['video']; length = sample['length']
            if by_id.get(key) != sample:
                checks['annotation_test_metadata_mismatch'].append(key)
            if key not in members:
                continue
            scores = json.loads(z.read(members[key]))
            if [int(k) for k in scores] != list(range(0, length, 16)):
                checks['score_anchor_grid_mismatch'].append(key)
            last = next(reversed(scores))
            end = int(scores[last]['end'])
            if end != length:
                checks['score_end_length_mismatch'].append(key)
            expanded = 16*(len(scores)-1) + end-int(last)
            # Exact length-only counterpart of released AP script's trimming.
            ap_length = expanded - 1 - ((expanded-1) % 16)
            if key not in gt or ap_length != len(gt[key]):
                checks['ap_trimmed_length_mismatch'].append(key)
            if key in gt and not set(gt[key]).issubset({0, 1}):
                checks['nonbinary_ap_labels'].append(key)
    result = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'score_archive_sha256': hashlib.sha256(a.scores.read_bytes()).hexdigest(),
        'bytes': a.scores.stat().st_size, 'zip_crc_pass': bad_crc is None,
        'score_files': len(members), 'test_ids': len(ids),
        'missing_score_ids': sorted(ids-members.keys()),
        'extra_score_ids': sorted(members.keys()-ids),
        'annotation_ids': len(annotations), 'ap_ground_truth_keys': len(gt),
        'total_metadata_frames': sum(x['length'] for x in meta),
        'max_video_frames': max(x['length'] for x in meta),
        'feature_archive_bytes_from_prior_metadata': 652318723,
        'feature_contents_verified': False,
        'checks': checks,
        'auc_script_uses_position_weighting': True,
        'ap_script_uses_position_weighting': False,
        'gpu_calls': 0, 'performance_experiment_run': False,
        'source_hashes': {name: hashlib.sha256((a.repo/name).read_bytes()).hexdigest()
                          for name in ['compute_auc_XD_Violence.py',
                                       'compute_AP_XD_Violence.py',
                                       'Data/xd_test_800.json',
                                       'Data/xd_annotations.json',
                                       'Data/name_gt_pair.json']},
    }
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ['score_files', 'zip_crc_pass', 'checks',
                                             'performance_experiment_run']}, indent=2))


if __name__ == '__main__':
    main()
