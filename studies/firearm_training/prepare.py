"""Training 1 preprocessing: a leak-free stratified split, per-split label files, box files for Training 2.

Split. Repeated takes of one scripted action by one subject at one place (a recording group) are
near-duplicates, so a group never spans two splits. For each action code, its groups are shuffled
with a seeded generator, the first three go one each to test, val and train (every code appears
in every split), and each further group goes to the split furthest below its video target
(60/15/25; ties to train, then test, then val). A subject-held-out split is not possible: Handgun
was filmed with subjects V1 and V2 only.

Labels. Each split file lists, per video: the category (the dataset folder), the exact action code
from the folder name, its coarse group from the codebook, and the metadata the report stratifies
by. Training 1 reads only these. Boxes (audited origin, 0-1000 grid of the 448x448 input) go to
separate boxes_<split>.json files for Training 2 and are not read by Training 1.
"""
import hashlib
import json
import random
from pathlib import Path

SPLITS = ('train', 'val', 'test')
TARGETS = {'train': 0.60, 'val': 0.15, 'test': 0.25}
TIE_ORDER = ('train', 'test', 'val')
FIELDS = ('id', 'category', 'action', 'folder', 'camera', 'camera_type', 'location', 'subject', 'brightness', 'take',
          'recording_group', 'width', 'height', 'fps', 'frames_decoded', 'duration_s', 'staged_path', 'sha256', 'label_file')


def assign(inventory, seed):
    """{video_id: split}; deterministic for a seed."""
    by_code = {}
    for row in inventory:
        by_code.setdefault(row['action'], {}).setdefault(row['recording_group'], []).append(row['id'])
    out = {}
    for code in sorted(by_code):
        groups = sorted(by_code[code].items())
        random.Random(f'{seed}:{code}').shuffle(groups)
        total = sum(len(v) for _, v in groups)
        filled = {s: 0 for s in SPLITS}
        for n, (_, ids) in enumerate(groups):
            if n < 3:
                split = ('test', 'val', 'train')[n]
            else:
                split = max(TIE_ORDER, key=lambda s: (TARGETS[s] * total - filled[s], -TIE_ORDER.index(s)))
            filled[split] += len(ids)
            for vid in ids:
                out[vid] = split
    return out


def label_row(row, book):
    code = book['codes'][row['action']]
    if code['category'] != row['category']:
        raise ValueError(f"{row['id']}: action {row['action']} belongs to {code['category']}, folder says {row['category']}")
    return {**{k: row[k] for k in FIELDS}, 'action_group': code['group']}


def summary(rows):
    count = lambda key: {k: sum(1 for r in rows if r[key] == k) for k in sorted({r[key] for r in rows})}
    return {'videos': len(rows), 'recording_groups': len({r['recording_group'] for r in rows}),
            'action_codes': len({r['action'] for r in rows}), 'category': count('category'), 'action_group': count('action_group'),
            'camera': count('camera'), 'subject': count('subject'), 'location': count('location')}


def boxes_for(row, original_root, policy):
    from vera_runner.firearm_annotations import frame_boxes, frame_statuses, load_coco
    label = policy.get('label_files', {}).get(row['id'], row['label_file'])
    coco = load_coco(Path(original_root) / row['category'] / row['folder'] / label) if label else None
    statuses = frame_statuses(row['category'], row['frames_decoded'], coco, policy['origin'])
    boxes = frame_boxes(coco, row['width'], row['height'], policy['origin']) if coco else {}
    return {'statuses': statuses, 'boxes': {str(k): v for k, v in sorted(boxes.items())}}


def prepare(inventory_path, codebook_path, policy_path, original_root, out_dir, seed):
    inventory = json.loads(Path(inventory_path).read_text())
    book = json.loads(Path(codebook_path).read_text())
    policy = json.loads(Path(policy_path).read_text())
    missing = sorted({r['action'] for r in inventory} - set(book['codes']))
    unused = sorted(set(book['codes']) - {r['action'] for r in inventory})
    if missing or unused:
        raise ValueError(f'codebook and data disagree: missing {missing}, unused {unused}')
    split_of = assign(inventory, seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {'seed': seed, 'targets': TARGETS, 'inventory_sha256': hashlib.sha256(Path(inventory_path).read_bytes()).hexdigest(),
              'codebook_sha256': hashlib.sha256(Path(codebook_path).read_bytes()).hexdigest(), 'splits': {}}
    for split in SPLITS:
        rows = sorted((label_row(r, book) for r in inventory if split_of[r['id']] == split), key=lambda r: r['id'])
        (out / f'{split}.json').write_text(json.dumps(rows, indent=1) + '\n')
        (out / f'boxes_{split}.json').write_text(json.dumps({r['id']: boxes_for(r, original_root, policy) for r in rows}) + '\n')
        result['splits'][split] = {**summary(rows), 'sha256': hashlib.sha256((out / f'{split}.json').read_bytes()).hexdigest()}
    groups = {}
    for r in inventory:
        groups.setdefault(r['recording_group'], set()).add(split_of[r['id']])
    result['groups_spanning_splits'] = sorted(g for g, s in groups.items() if len(s) > 1)
    result['codes_missing_from_a_split'] = sorted(c for c in book['codes']
                                                  if any(not any(r['action'] == c and split_of[r['id']] == s for r in inventory) for s in SPLITS))
    (out / 'splits.json').write_text(json.dumps(result, indent=1) + '\n')
    return result
