"""Group-disjoint FIT/STOP construction, sealed holdouts, and reversible views."""
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES = ('Handgun', 'Machine_Gun', 'No_Gun')
KNOWN = {'positive', 'negative_category', 'confirmed_absent'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write_json(path, obj, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if immutable and path.exists():
        if path.read_text() != text:
            raise ValueError(f'immutable artifact differs: {path}')
        return
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def indices(n):
    if n < 8:
        raise ValueError('eight distinct frames required')
    # Equivalent to the existing sample_indices for the integral dataset lengths.
    return [int(i * n / 8) for i in range(8)]


def geometry(width, height, size=448):
    scale = min(size / width, size / height)
    rw, rh = round(width * scale), round(height * scale)
    return {'width': width, 'height': height, 'size': size, 'rw': rw, 'rh': rh,
            'left': (size - rw) // 2, 'top': (size - rh) // 2}


def box_to_view(box, g):
    """Existing 0..1000 original-view xyxy -> unit letterboxed xyxy."""
    return [(box[i] / 1000 * (g['rw'] if i % 2 == 0 else g['rh']) +
             (g['left'] if i % 2 == 0 else g['top'])) / g['size'] for i in range(4)]


def box_from_view(box, g):
    return [(box[i] * g['size'] - (g['left'] if i % 2 == 0 else g['top'])) /
            (g['rw'] if i % 2 == 0 else g['rh']) * 1000 for i in range(4)]


def letterbox(image, g):
    from PIL import Image
    if image.size != (g['width'], g['height']):
        raise ValueError(f'frame dimensions changed: {image.size} vs {g}')
    out = Image.new('RGB', (g['size'], g['size']), (0, 0, 0))
    out.paste(image.convert('RGB').resize((g['rw'], g['rh']), Image.Resampling.BICUBIC), (g['left'], g['top']))
    return out


def split_train(rows, seed):
    by_cat = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_cat[row['category']][row['recording_group']].append(row)
    stop_ids = set()
    for cat, grouped in sorted(by_cat.items()):
        groups = sorted(grouped)
        random.Random(f'{seed}:{cat}').shuffle(groups)
        target, chosen = 0.2 * sum(map(len, grouped.values())), 0
        for group in groups:
            n = len(grouped[group])
            if abs(chosen + n - target) < abs(chosen - target):
                stop_ids.update(r['id'] for r in grouped[group])
                chosen += n
    return {name: sorted([r for r in rows if (r['id'] in stop_ids) == (name == 'stop')], key=lambda r: r['id'])
            for name in ('fit', 'stop')}


def balanced_order(rows, seed, pass_index):
    pools = {}
    for cat in CATEGORIES:
        pool = sorted([r['id'] for r in rows if r['category'] == cat])
        random.Random(f'{seed}:{pass_index}:{cat}').shuffle(pool)
        pools[cat] = pool
    order = []
    while any(pools.values()):
        for cat in CATEGORIES:
            if pools[cat]:
                order.append(pools[cat].pop())
    return order


def enrich(row, ann, frames_root):
    g = geometry(row['width'], row['height'])
    sampled = indices(row['frames_decoded'])
    views = []
    for slot, frame in enumerate(sampled):
        path = Path(frames_root) / row['id'] / f'{frame:06d}.jpg'
        if not path.is_file():
            raise ValueError(f'missing frame: {path}')
        status = ann['statuses'][frame]
        raw = ann['boxes'].get(str(frame), [])
        if status == 'positive' and not raw:
            raise ValueError(f'positive without box: {row["id"]}:{frame}')
        if status in ('negative_category', 'confirmed_absent') and raw:
            raise ValueError('absence with box')
        boxes = [box_to_view(b, g) for b in raw] if status in KNOWN else None
        if boxes is not None:
            for box in boxes:
                if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
                    raise ValueError('invalid transformed box')
        views.append({'frame_id': slot, 'source_frame': frame, 'path': str(path),
                      'sha256': digest(path), 'status': status, 'boxes': boxes})
    return {**row, 'action_code': row['action'], 'views': views, 'geometry': g}


def assert_disjoint(splits):
    for key in ('id', 'recording_group', 'sha256'):
        seen = {}
        for split, rows in splits.items():
            for row in rows:
                value = row.get(key)
                if not value:
                    raise ValueError(f'missing identity {key}')
                if value in seen and seen[value] != split:
                    raise ValueError(f'{key} overlap: {seen[value]} / {split}')
                seen[value] = split


def audit(data_dir, frames_root, output, seed=20261026):
    data_dir, output = Path(data_dir), Path(output)
    train = json.loads((data_dir / 'train.json').read_text())
    val = json.loads((data_dir / 'val.json').read_text())
    # Test content is projected immediately to identity metadata. No test images,
    # annotations, action/category targets, or predictions enter the study artifacts.
    test_meta = [{k: r[k] for k in ('id', 'recording_group', 'sha256')}
                 for r in json.loads((data_dir / 'test.json').read_text())]
    assert_disjoint({'train': train, 'val': val, 'test': test_meta})
    parts = split_train(train, seed)
    assert_disjoint(parts)
    annotations = json.loads((data_dir / 'boxes_train.json').read_text())
    result = {'seed': seed, 'test_policy': 'identity overlap audit only; no images/boxes/predictions',
              'validation_policy': 'historically used Phase 1; post-freeze reporting only',
              'historical_test_access': 'not established; do not claim newly pristine',
              'source_hashes': {name: digest(data_dir / name) for name in
                                ('train.json', 'val.json', 'test.json', 'boxes_train.json', 'splits.json')},
              'source_counts': {'train': len(train), 'val': len(val), 'test_metadata': len(test_meta)},
              'splits': {}, 'overlaps': 0}
    for split, rows in parts.items():
        if set(r['category'] for r in rows) != set(CATEGORIES):
            raise ValueError('class missing from FIT/STOP')
        rich = [enrich(r, annotations[r['id']], frames_root) for r in rows]
        # Detect exact frame reuse across source videos without opening holdouts.
        write_json(output / f'{split}.json', rich, immutable=True)
        result['splits'][split] = {'count': len(rows), 'groups': len({r['recording_group'] for r in rows}),
                                  'categories': dict(Counter(r['category'] for r in rows)),
                                  'action_codes': dict(Counter(r['action'] for r in rows)),
                                  'annotations': dict(Counter(v['status'] for r in rich for v in r['views'])),
                                  'manifest_sha256': digest(output / f'{split}.json')}
    frame_sets = {s: {v['sha256'] for r in json.loads((output / f'{s}.json').read_text()) for v in r['views']}
                  for s in parts}
    overlap = frame_sets['fit'] & frame_sets['stop']
    if overlap:
        raise ValueError(f'exact frame overlap FIT/STOP: {len(overlap)}')
    write_json(output / 'audit.json', result, immutable=True)
    return result


def load_split(root, split, frozen_manifest=None):
    if split not in ('fit', 'stop'):
        raise ValueError('holdout requires the explicit frozen evaluation entrypoint')
    return json.loads((Path(root) / f'{split}.json').read_text())
