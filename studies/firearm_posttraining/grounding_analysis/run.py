"""Freeze, verify and execute a prospective grounding analysis supplement.

Never modifies the base protocol, running release, scorer, outputs or training.
"""
import argparse
import csv
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from ..data import canonical_hash, digest, write_json, enrich
from ..protocol import validate, verify_selection, SEEDS
from ..report import localization
from .statistics import grounding_test, claim

PLAN = {
    'version': 'grounding-inference-v1', 'contrast': 'RG-SG',
    'endpoint': 'arithmetic mean over three fixed seeds of pooled box F1 at IoU>=0.50',
    'formula': 'F1_seed = 2*TP_seed/(predicted_boxes_seed+GT_boxes_seed)',
    'scorer': 'unchanged report.localization; one-to-one matching; unknown frames excluded; whole-image boxes cannot match; invalid grounding is scored by existing parser',
    'primary_reporting_split': 'test', 'validation_role': 'exploratory_only',
    'resampling_unit': 'recording_group; all associated videos, frames, annotations and three seed pairs stay together',
    'test': 'two-sided paired recording-group method-label permutation test',
    'null': 'joint groupwise exchangeability of RG and SG count vectors, conditional on fixed seeds',
    'alpha': .05, 'exact_max_groups': 16, 'monte_carlo_permutations': 100000,
    'monte_carlo_p': '(number as-or-more extreme + 1)/(draws + 1)',
    'bootstrap_replicates': 5000, 'interval': '95% paired recording-group percentile interval for F1 difference',
    'rng_seed': 20261027, 'minimum_positive_gt_groups': 2,
    'minimum_valid_bootstrap_fraction': .95,
    'positive_evidence': 'positive RG-SG difference AND two-sided p<=0.05 AND lower95% paired bootstrap interval>0',
    'claim_family': 'fixed sequence on test: unchanged RG-SG recognition gain+No_Gun noninferiority rule first; grounding F1 second',
    'claim_gate': 'confirmatory grounding claim only if existing test:RG-SG positive_and_noninferior AND grounding positive_evidence',
    'multiplicity': 'nominal alpha0.05 fixed-sequence gatekeeping for these two claims; other splits/endpoints/contrasts exploratory, no additional confirmatory claims',
    'supporting_metrics': ['existing IoU50 precision', 'existing IoU50 recall', 'No_Gun frame false-box rate', 'whole-image box rate', 'invalid/truncated fraction', 'annotation coverage', 'seed-specific F1 and seed spread'],
    'claim_limit': 'F1 improvement is not a guarantee that every precision/recall/false-box component improved; report all components and harms',
    'assumptions': 'independent representative recording groups and paired method exchangeability under the permutation null; training seeds fixed, not independent sample units',
    'insufficient_support': 'inconclusive if fewer than two positive-GT groups or fewer than95% bootstrap samples retain positive GT; low permutation resolution/wide intervals are not evidence of no effect',
    'history': 'historically used reporting dataset, not newly pristine external validation',
    'scoring_rules_changed': False, 'training_or_selection_changed': False,
    'amendment_timing': 'after training started, before final selection/evaluation/results; not original pre-training preregistration',
    'reference': 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html',
}


def code_hashes():
    return {p.name: digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def metadata(root):
    ledger = json.loads((root / 'budget.json').read_text())
    return {'selection_exists': (root / 'selection.json').exists(),
            'evaluation_directory_exists': (root / 'evaluation').exists(),
            'report_directory_exists': (root / 'report').exists(),
            'evaluation_jobs': [k for k, v in ledger.items() if v['stage'] == 'evaluation'],
            'active_training_jobs': [k for k, v in ledger.items() if v['stage'] == 'sweep' and v['status'] != 'finished'],
            'pipeline_status': json.loads((root / 'pipeline/status.json').read_text())}


def freeze(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    protocol = validate(root / 'protocol.json', root / 'data')
    before = metadata(root)
    if any(before[k] for k in ('selection_exists', 'evaluation_directory_exists', 'report_directory_exists', 'evaluation_jobs')):
        raise ValueError('amendment must freeze before selection, evaluation and reporting; inspect metadata only')
    source = Path(__file__).resolve().parents[3]
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=source, text=True):
        raise ValueError('clean committed analysis release required')
    record = {'status': 'frozen_analysis_amendment', 'plan': PLAN,
              'base_protocol_path': str(root / 'protocol.json'), 'base_protocol_hash': protocol['protocol_hash'],
              'base_protocol_file_sha256': digest(root / 'protocol.json'),
              'analysis_code_hashes': code_hashes(),
              'analysis_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
              'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'frozen_unix': time.time(),
              'authorization': "Human in chat01a17dce: add a proper statistical test for grounding before looking at final results; keep scoring rules unchanged; document a protocol amendment.",
              'metadata_before': before, 'no_final_results_inspected': True,
              'frozen_training_release_modified': False}
    after = metadata(root)
    if any(after[k] for k in ('selection_exists', 'evaluation_directory_exists', 'report_directory_exists', 'evaluation_jobs')):
        raise ValueError('evaluation/selection began during amendment preparation')
    record['metadata_after'] = after
    record['amendment_hash'] = canonical_hash(record)
    write_json(output, record, immutable=True)
    return record


def verify(path, root):
    record = json.loads(Path(path).read_text())
    other = dict(record); expected = other.pop('amendment_hash', None)
    if record.get('status') != 'frozen_analysis_amendment' or canonical_hash(other) != expected:
        raise ValueError('amendment modified or not frozen')
    if record['plan'] != PLAN or record['analysis_code_hashes'] != code_hashes():
        raise ValueError('analysis source/plan differs from frozen amendment')
    protocol = validate(record['base_protocol_path'], Path(root) / 'data')
    if (protocol['protocol_hash'] != record['base_protocol_hash'] or
        digest(record['base_protocol_path']) != record['base_protocol_file_sha256']):
        raise ValueError('base protocol changed after analysis amendment')
    return record


def counts_for_videos(rows, predictions, book):
    by_id = {p['video']: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r['id'] for r in rows}:
        raise ValueError('prediction identities incomplete or duplicated')
    counts = [localization([r], [by_id[r['id']]], book) for r in rows]
    # Prove decomposition matches the unchanged pooled scorer, not a new metric implementation.
    pooled = localization(rows, predictions, book)
    for key in ('tp', 'predictions', 'gt', 'negative_frames', 'false_box_frames', 'whole_image_boxes', 'scored_frames', 'all_frames'):
        if sum(c[key] for c in counts) != pooled[key]:
            raise ValueError('per-video decomposition differs from frozen scoring')
    return np.array([[c[k] for k in ('tp', 'predictions', 'gt')] for c in counts]), counts, pooled


def analyze(root, amendment, original_data, frames_root):
    root = Path(root).resolve()
    record = verify(amendment, root)
    selection, protocol = verify_selection(root / 'selection.json', root / 'data')
    if (root / 'selection.json').stat().st_mtime < record['frozen_unix']:
        raise ValueError('selection predates analysis amendment')
    ledger = json.loads((root / 'budget.json').read_text())
    if any(v['started_unix'] < record['frozen_unix'] for v in ledger.values() if v['stage'] == 'evaluation'):
        raise ValueError('evaluation predates analysis amendment')
    if json.loads((root / 'report/status.json').read_text()).get('status') != 'complete':
        raise ValueError('complete base report required before supplemental inference')
    contrasts = json.loads((root / 'report/paired-contrasts.json').read_text())
    with (root / 'report/grounding.csv').open() as f:
        base_grounding = {(x['split'], x['method'], int(x['seed'])): x for x in csv.DictReader(f)}
    source = Path(original_data)
    audit = json.loads((root / 'data/audit.json').read_text())
    book = json.loads(Path('studies/firearm_training/codebook.json').read_text())
    results, input_hashes = {}, {}
    for split in ('val', 'test'):
        if digest(source / f'{split}.json') != audit['source_hashes'][f'{split}.json']:
            raise ValueError('reporting source changed')
        annotations_path = source / f'boxes_{split}.json'
        annotations_hash = digest(annotations_path)
        raw = json.loads((source / f'{split}.json').read_text())
        annotations = json.loads(annotations_path.read_text())
        rows = [enrich(r, annotations[r['id']], frames_root) for r in raw]
        method_counts = {}
        for method in ('RG', 'SG'):
            values = []
            for seed in SEEDS:
                d = root / 'evaluation' / split / f'{method}-{seed}'
                result = json.loads((d / 'result.json').read_text())
                manifest = json.loads((d / 'holdout_manifest.json').read_text())
                p = d / 'evaluation/predictions.jsonl'
                if (result.get('status') != 'complete' or digest(p) != result['predictions_sha256'] or
                    manifest['selection_hash'] != selection['selection_hash'] or manifest['annotation_sha256'] != annotations_hash):
                    raise ValueError('evaluation/annotation binding failed')
                predictions = [json.loads(x) for x in p.read_text().splitlines()]
                counts, per_video, pooled = counts_for_videos(rows, predictions, book)
                for key in ('tp', 'predictions', 'gt', 'negative_frames', 'false_box_frames', 'whole_image_boxes', 'scored_frames', 'all_frames'):
                    if pooled[key] != int(base_grounding[split, method, seed][key]):
                        raise ValueError('supplement differs from committed base report counts')
                values.append(counts)
                input_hashes[str(p)] = digest(p)
            method_counts[method] = np.array(values)
        result = grounding_test([r['recording_group'] for r in rows], method_counts['RG'], method_counts['SG'],
                                bootstraps=PLAN['bootstrap_replicates'], permutations=PLAN['monte_carlo_permutations'],
                                exact_max_groups=PLAN['exact_max_groups'], seed=PLAN['rng_seed'])
        results[split] = claim(result, split, contrasts[f'{split}:RG-SG']['positive_and_noninferior'])
    out = root / 'report/grounding-statistics-v1'
    out.mkdir(parents=True, exist_ok=False)
    final = {'status': 'complete', 'amendment_hash': record['amendment_hash'],
             'protocol_hash': protocol['protocol_hash'], 'selection_hash': selection['selection_hash'],
             'analysis_code_hashes': code_hashes(), 'input_prediction_hashes': input_hashes,
             'base_grounding_csv_sha256': digest(root / 'report/grounding.csv'),
             'base_recognition_contrasts_sha256': digest(root / 'report/paired-contrasts.json'),
             'results': results, 'scoring_rules_changed': False}
    write_json(out / 'results.json', final, immutable=True)
    lines = ['# Grounding statistical amendment results', '',
             'This supplement implements the prospectively frozen analysis amendment. Training had already started when it was frozen; final selections/evaluation/results had not. The base scorer and protocol are unchanged.', '',
             'Endpoint: mean across fixed seeds of pooled box F1 at IoU>=0.50, RG minus SG. Method swaps and bootstrap samples use recording groups; frames/videos/seeds are not independent replicates. Positive evidence requires positive effect, two-sided permutation p<=.05 and positive lower95% interval. A confirmatory test-set grounding claim additionally requires the existing test-set recognition gain/noninferiority gate. Validation is exploratory.', '']
    for split, result in results.items():
        lines.append(f'- {split}: {json.dumps(result, sort_keys=True)}')
    lines += ['', 'Inspect the base grounding.csv precision, recall, false-box, whole-image and annotation-coverage metrics together. F1 superiority does not imply that every component improved. Permutation inference assumes independent groups and within-group method exchangeability under the null. The test dataset was historically used and is not newly pristine.']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    write_json(out / 'status.json', {'status': 'complete', 'amendment_hash': record['amendment_hash'], 'results_sha256': digest(out / 'results.json')}, immutable=True)
    return final


def watch(root, amendment, original_data, frames_root):
    root = Path(root).resolve()
    status_path = root / 'analysis_amendments/grounding-v1-watcher-status.json'
    with (root / 'analysis_amendments/grounding-v1-watcher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        record = verify(amendment, root)
        try:
            write_json(status_path, {'stage': 'waiting_for_complete_base_report', 'pid': os.getpid(), 'amendment_hash': record['amendment_hash']})
            while True:
                report = root / 'report/status.json'
                if report.exists():
                    if json.loads(report.read_text()).get('status') != 'complete':
                        raise RuntimeError('base report incomplete; sensitivity work required')
                    break
                pipeline = json.loads((root / 'pipeline/status.json').read_text())
                if pipeline['stage'] == 'failed':
                    raise RuntimeError('base pipeline failed; preserve it and diagnose before analysis')
                time.sleep(30)
            result = analyze(root, amendment, original_data, frames_root)
            write_json(status_path, {'stage': 'complete', 'pid': os.getpid(), 'amendment_hash': record['amendment_hash'], 'results_path': str(root / 'report/grounding-statistics-v1/results.json')})
            return result
        except Exception as exc:
            write_json(status_path, {'stage': 'failed', 'pid': os.getpid(), 'error': str(exc), 'error_type': type(exc).__name__, 'amendment_hash': record['amendment_hash']})
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('freeze', 'watch', 'run'))
    p.add_argument('--root', required=True)
    p.add_argument('--amendment', required=True)
    p.add_argument('--original-data')
    p.add_argument('--frames-root')
    a = p.parse_args()
    if a.command == 'freeze':
        print(json.dumps(freeze(a.root, a.amendment), indent=2))
    else:
        if not a.original_data or not a.frames_root:
            p.error('analysis requires original-data and frames-root')
        result = (watch if a.command == 'watch' else analyze)(a.root, a.amendment, a.original_data, a.frames_root)
        print(json.dumps(result, indent=2))
