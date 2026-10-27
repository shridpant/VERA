"""Shared deterministic answer metrics. Development entrypoints accept STOP only."""
import json
import time
from pathlib import Path
from .data import CATEGORIES, write_json
from .rewards import parse, parse_labels, score
from .runtime import append


def metrics(rows, predictions, book, grounded=False):
    by_id = {p['video']: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r['id'] for r in rows}:
        raise ValueError('evaluation IDs incomplete or duplicated')
    correct = dict(category=0, group=0, code=0)
    recalls = {}
    invalid = truncated = false_positive = 0
    for cat in CATEGORIES:
        subset = [r for r in rows if r['category'] == cat]
        hits = 0
        for r in subset:
            p = by_id[r['id']]
            parsed = parse_labels(p['text'], book, p['truncated'])
            invalid += parse(p['text'], book, grounded, p['truncated']) is None
            truncated += p['truncated']
            if parsed is None:
                continue
            c = parsed['category'] == r['category']
            correct['category'] += c
            correct['group'] += parsed['action_group'] == r['action_group']
            correct['code'] += parsed['action_code'] == r['action_code']
            hits += c
            false_positive += cat == 'No_Gun' and parsed['category'] != 'No_Gun'
        recalls[cat] = hits / len(subset) if subset else None
    n = len(rows)
    if not n or any(v is None for v in recalls.values()):
        raise ValueError('all three classes required for balanced category accuracy')
    negatives = sum(r['category'] == 'No_Gun' for r in rows)
    return {'balanced_category_accuracy': sum(recalls.values()) / 3,
            'category_accuracy': correct['category'] / n, 'category_recalls': recalls,
            'action_group_accuracy': correct['group'] / n, 'action_code_accuracy': correct['code'] / n,
            'no_gun_category_false_positive_rate': false_positive / negatives,
            'invalid_fraction': invalid / n, 'truncated_fraction': truncated / n,
            'videos': n, 'negative_videos': negatives}


def evaluate(native, rows, path, grounded=False, questions=None, deadline=float('inf'), stop_requested=lambda: False):
    path = Path(path)
    if path.exists():
        raise ValueError('evaluation output already exists')
    path.mkdir(parents=True)
    predictions = []
    started = time.monotonic()
    for row in rows:
        if time.monotonic() >= deadline or stop_requested():
            write_json(path / 'status.json', {'status': 'incomplete', 'videos': len(predictions)})
            return None
        t = time.monotonic()
        prompt, _ = native.prompt(row, grounded, questions)
        result = native.generate(prompt)
        result.update({'video': row['id'], 'recording_group': row['recording_group'],
                       'latency_seconds': time.monotonic() - t})
        predictions.append(result)
        append(path / 'predictions.jsonl', result)
    result = metrics(rows, predictions, native.book, grounded)
    result.update(elapsed_seconds=time.monotonic() - started,
                  latency_seconds_per_video=sum(p['latency_seconds'] for p in predictions) / len(rows))
    write_json(path / 'metrics.json', result)
    write_json(path / 'status.json', {'status': 'complete', 'videos': len(predictions)})
    return result
