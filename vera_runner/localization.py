"""Fixed-output localization metrics: one-to-one matching of slot boxes to existing boxes.

Boxes are [x1, y1, x2, y2] on the 0-1000 grid of the model input. For one window slot, predicted
and ground-truth boxes are matched one-to-one with scipy's linear_sum_assignment on the cost
-(1 + IoU) for pairs with IoU >= threshold and 0 otherwise: the most matches first, then the
largest total IoU. Inputs are sorted by coordinates first, so ties resolve the same way for any
input order. Unmatched predictions are false positives only where the frame status supports
precision; unmatched ground truth is always a miss. A slot without a usable answer (malformed,
unknown, missing, duplicate, truncated) misses every ground-truth box and adds no prediction.
There are no confidence scores, so no ranked mAP is computed.
"""
import numpy as np

ANSWERED = ('boxes', 'empty')


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds, gts, threshold):
    """[(pred_index, gt_index, iou)] in the caller's indexing."""
    from scipy.optimize import linear_sum_assignment
    if not preds or not gts:
        return []
    p_order = sorted(range(len(preds)), key=lambda i: tuple(preds[i]))
    g_order = sorted(range(len(gts)), key=lambda i: tuple(gts[i]))
    ious = np.array([[iou(preds[i], gts[j]) for j in g_order] for i in p_order])
    cost = np.where(ious >= threshold, -(1.0 + ious), 0.0)
    rows, cols = linear_sum_assignment(cost)
    return sorted((p_order[r], g_order[c], float(ious[r, c])) for r, c in zip(rows, cols) if ious[r, c] >= threshold)


def score_instance(status, preds, gts, threshold, precision_ok):
    """Counts for one window slot. fp is None where precision is not evaluable."""
    if status not in ANSWERED:
        return {'answered': False, 'tp': 0, 'fp': None, 'fn': len(gts), 'matched_ious': [], 'missed_gt': list(range(len(gts)))}
    pairs = match(preds, gts, threshold)
    hit = {g for _, g, _ in pairs}
    return {'answered': True, 'tp': len(pairs), 'fp': (len(preds) - len(pairs)) if precision_ok else None,
            'fn': len(gts) - len(pairs), 'matched_ious': [v for _, _, v in pairs],
            'missed_gt': [j for j in range(len(gts)) if j not in hit]}


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if fp is not None and tp + fp > 0 else None
    recall = tp / (tp + fn) if tp + fn > 0 else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall > 0 else (
        0.0 if precision is not None and recall is not None else None)
    return precision, recall, f1


def size_bin(box, edges):
    area = (box[2] - box[0]) * (box[3] - box[1]) / 1e6
    return 'small' if area < edges[0] else ('medium' if area < edges[1] else 'large')
