"""Tables and paired statistics only after frozen selection and complete evaluation."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from .data import CATEGORIES, enrich, box_from_view, write_json, digest
from .evaluation import metrics
from .protocol import verify_selection, METHODS, SEEDS
from .rewards import parse, parse_labels, iou
from .statistics import paired_interval, transitions


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def localization(rows, predictions, book):
    from scipy.optimize import linear_sum_assignment
    by_id = {p['video']: p for p in predictions}
    counts = dict(tp=0, gt=0, predictions=0, negative_frames=0, false_box_frames=0,
                  whole_image_boxes=0, scored_frames=0, all_frames=8 * len(rows))
    for row in rows:
        p = by_id[row['id']]
        answer = parse(p['text'], book, True, p['truncated'])
        predicted = {f['frame_id']: f['boxes'] for f in answer['grounding']} if answer else {}
        for frame in row['views']:
            if frame['boxes'] is None:
                continue
            gt = [box_from_view(b, row['geometry']) for b in frame['boxes']]
            pred = predicted.get(frame['frame_id'], [])
            whole = [((b[2]-b[0]) * (b[3]-b[1]) >= .95) for b in pred]
            mapped = [box_from_view(b, row['geometry']) for b in pred]
            counts['scored_frames'] += 1
            counts['gt'] += len(gt)
            counts['predictions'] += len(pred)
            counts['whole_image_boxes'] += sum(whole)
            if row['category'] == 'No_Gun':
                counts['negative_frames'] += 1
                counts['false_box_frames'] += bool(pred)
            if pred and gt:
                # Threshold matches maximize TP count first, then IoU. Whole-image
                # predictions cannot match but still count in the denominator.
                matrix = np.array([[0. if whole[j] else iou(b, g) for g in gt] for j, b in enumerate(mapped)])
                cost = (matrix >= .5).astype(float) + matrix * .0001
                a, b = linear_sum_assignment(cost, maximize=True)
                counts['tp'] += int((matrix[a, b] >= .5).sum())
    ratio = lambda a, b: counts[a] / counts[b] if counts[b] else None
    return {**counts, 'precision_IoU50': ratio('tp', 'predictions'), 'recall_IoU50': ratio('tp', 'gt'),
            'no_gun_frame_false_box_rate': ratio('false_box_frames', 'negative_frames'),
            'whole_image_box_rate': ratio('whole_image_boxes', 'predictions'),
            'annotation_coverage': ratio('scored_frames', 'all_frames'),
            'coordinate_policy': 'inverse letterbox; padding extents retained, no clipping'}


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(root, frozen_manifest, original_data, frames_root, output):
    root, output = Path(root), Path(output)
    selection, protocol = verify_selection(frozen_manifest, root / 'data')
    output.mkdir(parents=True, exist_ok=False)
    source = Path(original_data)
    book = json.loads(Path('studies/firearm_training/codebook.json').read_text())
    main, ground, transition_rows, contrasts, report_rows = [], [], [], {}, []
    all_metrics = {}
    for split in ('val', 'test'):
        audit = json.loads((root / 'data/audit.json').read_text())
        if digest(source / f'{split}.json') != audit['source_hashes'][f'{split}.json']:
            raise ValueError('original reporting target file changed')
        rows = json.loads((source / f'{split}.json').read_text())
        ann = json.loads((source / f'boxes_{split}.json').read_text())
        rich = [enrich(r, ann[r['id']], frames_root) for r in rows]
        answers = {}
        for method in ('B0', *METHODS):
            seeds = (SEEDS[0],) if method == 'B0' else SEEDS
            for seed in seeds:
                directory = root / 'evaluation' / split / f'{method}-{seed}'
                evaluation_result = json.loads((directory / 'result.json').read_text())
                if evaluation_result.get('status') != 'complete':
                    raise ValueError('incomplete evaluation')
                if digest(directory / 'evaluation/predictions.jsonl') != evaluation_result['predictions_sha256']:
                    raise ValueError('prediction artifacts changed')
                spec = json.loads((directory / 'spec.json').read_text())
                if spec.get('frozen_manifest') != str(frozen_manifest):
                    raise ValueError('evaluation uses another selection manifest')
                predictions = read_jsonl(directory / 'evaluation/predictions.jsonl')
                measured = metrics(rich, predictions, book, method in ('SG', 'RG'))
                parsed = {p['video']: parse_labels(p['text'], book, p['truncated']) for p in predictions}
                correct = [bool(parsed[r['id']] and parsed[r['id']]['category'] == r['category']) for r in rows]
                fp = [bool(r['category'] == 'No_Gun' and parsed[r['id']] and parsed[r['id']]['category'] != 'No_Gun') for r in rows]
                answers[method, seed] = {'correct': correct, 'fp': fp}
                latency = sum(p['latency_seconds'] for p in predictions) / len(predictions)
                run_result = {} if method == 'B0' else json.loads(Path(next(e['result'] for e in selection['entries'] if e['method'] == method and e['seed'] == seed)).read_text())
                warm = 0.
                if method in ('SL', 'RL', 'SG', 'RG'):
                    pairing = json.loads((root / 'sweep' / f'paired-initializations-{seed}.json').read_text())
                    warm = pairing['G' if method in ('SG', 'RG') else 'L']['elapsed_seconds']
                seconds = run_result.get('elapsed_seconds', 0) + warm
                record = {'split': split, 'method': method, 'seed': seed,
                          'supervision': 'labels+boxes' if method in ('SG', 'RG') else ('none' if method == 'B0' else 'labels'),
                          'balanced_category_accuracy': measured['balanced_category_accuracy'],
                          'category_accuracy': measured['category_accuracy'],
                          **{f'recall_{c}': measured['category_recalls'][c] for c in CATEGORIES},
                          'action_group_accuracy': measured['action_group_accuracy'],
                          'action_code_accuracy': measured['action_code_accuracy'],
                          'no_gun_category_false_positive_rate': measured['no_gun_category_false_positive_rate'],
                          'invalid_fraction': measured['invalid_fraction'], 'truncated_fraction': measured['truncated_fraction'],
                          'logical_training_gpu_hours': seconds / 3600, 'inference_seconds_per_video': latency,
                          'fit_passes': run_result.get('fit_passes'), 'censored': run_result.get('censored', False),
                          'distinct_video_exposures': run_result.get('cursor'),
                          'FIT_prediction_queries': run_result.get('FIT_prediction_queries'),
                          'FIT_optimizer_video_queries': run_result.get('FIT_optimizer_video_queries')}
                main.append(record)
                all_metrics[split, method, seed] = measured
                if method in ('SG', 'RG'):
                    ground.append({'split': split, 'method': method, 'seed': seed, **localization(rich, predictions, book)})
                if method != 'B0':
                    base = answers['B0', SEEDS[0]]['correct']
                    changed = transitions(rows, base, correct)
                    for category, counts in [('all', changed['all']), *changed['by_category'].items()]:
                        transition_rows.append({'split': split, 'method': method, 'seed': seed, 'category': category, **counts})
        for a, b in (('RG', 'SG'), ('RL', 'SL'), ('VQ', 'B0'), ('SL', 'B0'), ('SG', 'SL'), ('RG', 'RL'), ('RG', 'VQ')):
            paired = lambda method, field: [answers[method, SEEDS[0] if method == 'B0' else s][field] for s in SEEDS]
            interval = paired_interval(rows, paired(a, 'correct'), paired(b, 'correct'), paired(a, 'fp'), paired(b, 'fp'))
            interval['positive_and_noninferior'] = (interval['balanced_category_accuracy_interval'][0] > 0 and
                                                   interval['no_gun_false_positive_interval'][1] <= .05)
            interval['role'] = 'primary' if (a, b) == ('RG', 'SG') else 'secondary_exploratory_no_multiplicity_adjustment'
            contrasts[f'{split}:{a}-{b}'] = interval
    write_csv(output / 'main-seed.csv', main)
    write_csv(output / 'grounding.csv', ground)
    write_csv(output / 'error-transitions.csv', transition_rows)
    write_json(output / 'paired-contrasts.json', contrasts)
    means = []
    for split in ('val', 'test'):
        for method in ('B0', *METHODS):
            subset = [r for r in main if r['split'] == split and r['method'] == method]
            record = {'split': split, 'method': method, 'seeds': len(subset)}
            for key in ('balanced_category_accuracy', 'category_accuracy', 'action_group_accuracy', 'action_code_accuracy',
                        'no_gun_category_false_positive_rate', 'invalid_fraction', 'logical_training_gpu_hours', 'inference_seconds_per_video'):
                values = [r[key] for r in subset]
                record[key + '_mean'] = float(np.mean(values))
                record[key + '_std_across_seeds'] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.
            means.append(record)
    write_csv(output / 'main-mean.csv', means)
    curves = []
    for entry in selection['entries']:
        if entry['method'] == 'VQ':
            continue
        directory = Path(entry['result']).parent
        pairing = json.loads((root / 'sweep' / f'paired-initializations-{entry["seed"]}.json').read_text())
        warm_seconds = pairing['G' if entry['method'] in ('SG', 'RG') else 'L']['elapsed_seconds']
        for check in read_jsonl(directory / 'selection.jsonl'):
            state = json.loads((Path(check['checkpoint']) / 'state.json').read_text())
            curves.append({'method': entry['method'], 'seed': entry['seed'], 'set': 'STOP_selection_only',
                           'fit_passes': check['exposures'] / audit['splits']['fit']['count'],
                           'logical_gpu_hours': (warm_seconds + state['elapsed_seconds'] + check['elapsed_seconds']) / 3600,
                           'balanced_category_accuracy': check['balanced_category_accuracy'],
                           'no_gun_category_false_positive_rate': check['no_gun_category_false_positive_rate'],
                           'invalid_fraction': check['invalid_fraction']})
    write_csv(output / 'training-curves.csv', curves)
    budget = json.loads((root / 'budget.json').read_text())
    physical = sum(r.get('elapsed_seconds', r['reserved_seconds']) for r in budget.values()) / 3600
    censored = [r for r in main if r['censored']]
    lines = ['# Firearm post-training results', '',
             'These held-out results follow a frozen five-method, three-seed selection protocol. '
             'Original validation informed earlier Phase 1 work, and prior nonempty Training 1 test-output files establish historical test access. '
             'The test is historically used, not a newly pristine external holdout. No test outputs informed training in this execution.', '',
             f'Physical aggregate device occupancy: {physical:.3f} GPU-hours. Per-arm training time includes the shared warm-up charged to both paired arms; '
             'its logical sum can exceed physical execution time.', '',
             '| Split | Method | Balanced category accuracy (mean ± seed SD) | No_Gun false positives | Invalid |',
             '|---|---|---:|---:|---:|']
    for r in means:
        lines.append(f'| {r["split"]} | {r["method"]} | {r["balanced_category_accuracy_mean"]:.3f} ± {r["balanced_category_accuracy_std_across_seeds"]:.3f} | '
                     f'{r["no_gun_category_false_positive_rate_mean"]:.3f} | {r["invalid_fraction_mean"]:.3f} |')
    lines.extend(['', 'Paired intervals resample recording groups, keeping methods and seeds paired. '
                  'Seed SD describes training variability; frames and seeds are not independent video observations. '
                  'RG–SG is primary. RL–SL is secondary and exploratory; no multiplicity-corrected secondary claim is made.', ''])
    for name, r in contrasts.items():
        lines.append(f'- {name}: balanced category accuracy difference {r["balanced_category_accuracy_difference"]:.3f}, '
                     f'95% paired interval {r["balanced_category_accuracy_interval"]}; No_Gun false-positive difference interval '
                     f'{r["no_gun_false_positive_interval"]}. Positive-gain plus +0.05 noninferiority rule: '
                     f'{"met on this dataset" if r["positive_and_noninferior"] else "not met / inconclusive"}.')
    if censored:
        lines.extend(['', 'Time-censored runs exist. Do not claim equal achieved exposure. '
                      'A frozen last-common-exposure sensitivity evaluation is still required before scientific completion.'])
    lines.extend(['', 'See the seed-level, grounding, error-transition and paired-contrast artifacts for full results. '
                  'Localization metrics apply only to SG/RG and are distinct from category false positives. '
                  'Unknown annotations are excluded, and whole-frame boxes cannot match. '
                  'Inspect validity alongside accuracy because supervised gains can include format learning.'])
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    write_json(output / 'status.json', {'status': 'sensitivity_required' if censored else 'complete',
               'selection_hash': selection['selection_hash'], 'physical_gpu_hours': physical,
               'censored_run_records': len(censored)})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--frozen-manifest', required=True)
    p.add_argument('--original-data', required=True)
    p.add_argument('--frames-root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.root, a.frozen_manifest, a.original_data, a.frames_root, a.output)
