"""Fail-closed protocol and selection freezes, before any reporting holdout access."""
import argparse
import json
import math
from pathlib import Path
from .data import digest, canonical_hash, write_json

METHODS = ('VQ', 'SL', 'RL', 'SG', 'RG')
SEEDS = (20261026, 20261027, 20261028)


def freeze(root, output, label_smoke, grounded_smoke, supervised_pilot, label_panel, grounded_panel, max_passes):
    root = Path(root)
    if not .5 < max_passes <= 5 or max_passes * 2 != int(max_passes * 2):
        raise ValueError('common FIT-pass cap must be a half-pass multiple in (0.5,5]')
    evidence = {}
    for name, path, expected in (
        ('label_smoke', label_smoke, 'engineering_passed'), ('grounded_smoke', grounded_smoke, 'engineering_passed'),
        ('supervised_pilot', supervised_pilot, 'stopped'), ('label_panel', label_panel, 'passed'),
        ('grounded_panel', grounded_panel, 'passed')):
        path = Path(path)
        result = json.loads(path.read_text())
        if result.get('status') != expected:
            raise ValueError(f'gate failed: {name}')
        if 'smoke' in name and not (result.get('resume_exact') and result.get('prediction_reload_exact')):
            raise ValueError('resume gate missing')
        evidence[name] = {'path': str(path), 'sha256': digest(path)}
    setup = root / 'setup'
    if not (setup / 'requirements.lock').is_file() or not (setup / 'wheelhouse.json').is_file():
        raise ValueError('tested hash-locked environment required')
    preflight = json.loads((root / 'data/preflight.json').read_text())
    module = Path(__file__).parent
    record = {'status': 'frozen', 'methods': METHODS, 'seeds': SEEDS, 'baseline': 'B0',
              'primary_contrast': 'RG-SG', 'secondary_contrast': 'RL-SL',
              'no_gun_false_positive_noninferiority_margin': .05,
              'primary_interval': 'paired recording-group percentile bootstrap, 95%, seed-averaged',
              'shared_max_fit_passes': max_passes, 'paired_warmup_fit_passes': .5,
              'warmup_selection': 'fixed_no_STOP_queries', 'evidence': evidence,
              'data_hashes': {name: digest(root / 'data' / name) for name in ('fit.json', 'stop.json', 'audit.json', 'preflight.json')},
              'requirements_lock_sha256': digest(setup / 'requirements.lock'),
              'model': preflight['model'], 'model_manifest_hash': preflight['model_manifest_hash'],
              'prompt_hashes': preflight['prompt_hashes'], 'frames': 8, 'view': '448_letterbox',
              'completion_cap': 512, 'context_cap': 8192, 'precision': 'bf16',
              'lora': {'rank': 16, 'alpha': 32, 'dropout': 0., 'targets': 'language_attention_q_k_v_o_only'},
              'reward_version': 'hierarchical-iou-v1', 'reward_coefficients': {'C': .7, 'CG': .2, 'CGA': .1, 'grounded_outcome': .75, 'gated_Q': .25},
              'code_hashes': {p.name: digest(p) for p in module.glob('*.py')},
              'test_policy': 'sealed until complete selection manifest and explicit frozen-manifest argument',
              'validation_policy': 'historically used Phase 1; reporting only after selection freeze'}
    record['protocol_hash'] = canonical_hash(record)
    write_json(output, record, immutable=True)
    return record


def validate(path, data_root, model_path=None):
    protocol = json.loads(Path(path).read_text())
    copy = dict(protocol)
    saved_hash = copy.pop('protocol_hash', None)
    if protocol.get('status') != 'frozen' or canonical_hash(copy) != saved_hash:
        raise ValueError('protocol missing or modified')
    for name, expected in protocol['data_hashes'].items():
        if digest(Path(data_root) / name) != expected:
            raise ValueError(f'data changed after freeze: {name}')
    for name, expected in protocol['code_hashes'].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError(f'code changed after freeze: {name}; explicit protocol amendment needed')
    if model_path is not None and model_path != protocol['model']:
        raise ValueError('backbone differs from frozen protocol')
    return protocol


def freeze_selection(protocol_path, data_root, runs, output):
    protocol = validate(protocol_path, data_root)
    entries = []
    seen = set()
    for directory in runs:
        directory = Path(directory)
        spec = json.loads((directory / 'spec.json').read_text())
        result = json.loads((directory / 'result.json').read_text())
        method, seed = spec['method'], spec['seed']
        if (method, seed) in seen or method not in METHODS or seed not in SEEDS:
            raise ValueError('duplicate/unexpected method seed')
        if spec.get('frozen_protocol') != str(protocol_path) or result.get('status') != 'stopped':
            raise ValueError('not a completed frozen-protocol run')
        seen.add((method, seed))
        if method == 'VQ':
            selected = Path(result['best_questions_file'])
            files = {str(selected): digest(selected)}
        else:
            selected = Path(result['best_checkpoint'])
            hashes = json.loads((selected / 'hashes.json').read_text())
            files = {str(selected / name): digest(selected / name) for name in hashes}
            if any(files[str(selected / name)] != expected for name, expected in hashes.items()):
                raise ValueError('checkpoint corrupt')
        entries.append({'method': method, 'seed': seed, 'selected': str(selected), 'files': files,
                        'result': str(directory / 'result.json'), 'result_sha256': digest(directory / 'result.json'),
                        'spec_sha256': digest(directory / 'spec.json'), 'censored': result.get('censored', False)})
    if seen != {(m, s) for m in METHODS for s in SEEDS}:
        raise ValueError('all 15 method/seed outcomes required before selection freeze')
    result = {'status': 'selection_frozen', 'protocol_path': str(protocol_path),
              'protocol_hash': protocol['protocol_hash'], 'entries': entries}
    result['selection_hash'] = canonical_hash(result)
    write_json(output, result, immutable=True)
    return result


def verify_selection(path, data_root):
    selection = json.loads(Path(path).read_text())
    other = dict(selection)
    expected = other.pop('selection_hash', None)
    if selection.get('status') != 'selection_frozen' or canonical_hash(other) != expected:
        raise ValueError('selection manifest not frozen or modified')
    protocol = validate(selection['protocol_path'], data_root)
    if protocol['protocol_hash'] != selection['protocol_hash']:
        raise ValueError('protocol mismatch')
    for entry in selection['entries']:
        for file, expected in entry['files'].items():
            if digest(file) != expected:
                raise ValueError('selected artifact changed')
    return selection, protocol
