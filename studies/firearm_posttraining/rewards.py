"""Strict response parsing and annotation-aware deterministic video rewards."""
import json
import math

VERSION = 'hierarchical-iou-v1'
COORDINATE_VERSION = 'native-1000-to-unit-v1'


def _unique(pairs):
    d = {}
    for key, value in pairs:
        if key in d:
            raise ValueError('duplicate JSON key')
        d[key] = value
    return d


def parse_labels(text, book, truncated=False):
    """Same label schema for every arm; box validity cannot change classification."""
    if truncated:
        return None
    try:
        value = json.loads(text, object_pairs_hook=_unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))
        if not isinstance(value, dict) or value.keys() - {'category', 'action_group', 'action_code', 'grounding', 'explanation'}:
            return None
        labels = {k: value[k] for k in ('category', 'action_group', 'action_code')}
        return parse(json.dumps(labels), book)
    except (ValueError, TypeError, KeyError):
        return None


def parse(text, book, grounded=False, truncated=False):
    if truncated:
        return None
    try:
        p = json.loads(text, object_pairs_hook=_unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite')))
        if not isinstance(p, dict):
            return None
        required = {'category', 'action_group', 'action_code'} | ({'grounding'} if grounded else set())
        if not required <= p.keys() or p.keys() - required - {'explanation'}:
            return None
        if p['category'] not in book['categories'] or p['action_group'] not in book['groups'] or p['action_code'] not in book['codes']:
            return None
        if 'explanation' in p and not isinstance(p['explanation'], str):
            return None
        if grounded:
            if not isinstance(p['grounding'], list) or len(p['grounding']) != 8:
                return None
            seen = set()
            for f in p['grounding']:
                if not isinstance(f, dict) or set(f) != {'frame_id', 'boxes'}:
                    return None
                k, boxes = f['frame_id'], f['boxes']
                if type(k) is not int or k not in range(8) or k in seen or not isinstance(boxes, list):
                    return None
                seen.add(k)
                for b in boxes:
                    if (not isinstance(b, list) or len(b) != 4 or
                        any(type(x) not in (int, float) or not math.isfinite(x) for x in b) or
                        not (0 <= b[0] < b[2] <= 1000 and 0 <= b[1] < b[3] <= 1000)):
                        return None
                # Only the model boundary changes; every downstream scorer uses
                # unit coordinates. No unit guessing, clamping or box repair.
                f['boxes'] = [[x / 1000 for x in b] for b in boxes]
        return p
    except (ValueError, TypeError, KeyError):
        return None


def iou(a, b):
    inter = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.


def matching(pred, gt):
    """Maximum-total-IoU one-to-one assignment; unmatched predictions still penalize."""
    if not pred or not gt:
        return []
    from scipy.optimize import linear_sum_assignment
    matrix = [[iou(p, g) if (p[2] - p[0]) * (p[3] - p[1]) < .95 else 0. for g in gt] for p in pred]
    a, b = linear_sum_assignment(matrix, maximize=True)
    return [matrix[i][j] for i, j in zip(a, b)]


def box_quality(parsed, row):
    predictions = {f['frame_id']: f['boxes'] for f in parsed['grounding']}
    scores = []
    for v in row['views']:
        gt = v['boxes']
        if gt is None:
            continue
        pred = predictions[v['frame_id']]
        scores.append((float(not pred) if not gt else 2 * sum(matching(pred, gt)) / (len(gt) + len(pred))))
    return sum(scores) / len(scores) if scores else None


def score(text, row, book, grounded=False, truncated=False):
    p = parse(text, book, grounded, truncated)
    if p is None:
        return {'reward': 0., 'outcome': 0., 'quality': None, 'valid': False,
                'category_correct': False, 'group_correct': False, 'code_correct': False}
    c = p['category'] == row['category']
    g = p['action_group'] == row['action_group']
    a = p['action_code'] == row['action_code']
    outcome = .7 * c + .2 * (c and g) + .1 * (c and g and a)
    q = box_quality(p, row) if grounded else None
    reward = .75 * outcome + .25 * c * q if q is not None else outcome
    return {'reward': float(reward), 'outcome': float(outcome), 'quality': q, 'valid': True,
            'category_correct': c, 'group_correct': g, 'code_correct': a}
