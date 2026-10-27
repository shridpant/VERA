"""Fail-closed protocol and selection freezes, before any reporting holdout access."""
import argparse
import json
import math
from pathlib import Path
from .data import digest, canonical_hash, write_json
from .sampling import TEMPERATURE, VERSION as SAMPLING_VERSION

METHODS = ('VQ', 'SL', 'RL', 'SG', 'RG')
SEEDS = (20261026, 20261027, 20261028)


def freeze(root, output, label_smoke, grounded_smoke, supervised_pilot, label_panel, grounded_panel, max_passes, warmup_passes=.5):
    root = Path(root)
    if not .5 < max_passes <= 5 or max_passes * 2 != int(max_passes * 2):
        raise ValueError('common FIT-pass cap must be a half-pass multiple in (0.5,5]')
    if not .5 <= warmup_passes < max_passes or warmup_passes * 2 != int(warmup_passes * 2):
        raise ValueError('shared warm-up must be a half-pass multiple below the total pass cap')
    evidence = {}
    fit_count = len(json.loads((root / 'data' / 'fit.json').read_text()))
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
        if name.endswith('_panel'):
            spec = json.loads((path.parent / 'spec.json').read_text())
            warm_state = json.loads((Path(spec['adapter']) / 'state.json').read_text())
            if warm_state['cursor'] != math.floor(fit_count * warmup_passes) or warm_state['selection']['checks'] != 0:
                raise ValueError('real-reward panel reference differs from proposed fixed warm-up exposure')
            if spec['grounded'] != (name == 'grounded_panel'):
                raise ValueError('panel supervision mismatch')
        evidence[name] = {'path': str(path), 'sha256': digest(path)}
    setup = root / 'setup'
    if not (setup / 'requirements.lock').is_file() or not (setup / 'wheelhouse.json').is_file():
        raise ValueError('tested hash-locked environment required')
    preflight = json.loads((root / 'data/preflight.json').read_text())
    mask_audit = json.loads((root / 'data/target-mask-audit-v3.json').read_text())
    if (mask_audit.get('status') != 'passed' or mask_audit.get('target_mask') != 'known-schema-unknown-payload-v2' or
        mask_audit.get('current_model_sha256') != digest(Path(__file__).with_name('model.py'))):
        raise ValueError('passing target-mask audit for current source required')
    forecast = json.loads((root / 'throughput-forecast.json').read_text())
    if (forecast.get('status') != 'measured_forecast' or forecast['warmup_passes'] != warmup_passes or
        forecast['shared_max_fit_passes'] != max_passes):
        raise ValueError('matching measured throughput forecast required before freeze')
    module = Path(__file__).parent
    record = {'status': 'frozen', 'methods': METHODS, 'seeds': SEEDS, 'baseline': 'B0',
              'primary_contrast': 'RG-SG', 'secondary_contrast': 'RL-SL',
              'no_gun_false_positive_noninferiority_margin': .05,
              'no_gun_support_screen': 'paired bootstrap upper AND conservative independent-group harm/benefit Clopper-Pearson support upper must be <=.05; observed min/max size weighting; sparse evidence is inconclusive',
              'primary_interval': 'paired recording-group percentile bootstrap, 95%, seed-averaged',
              'shared_max_fit_passes': max_passes, 'paired_warmup_fit_passes': warmup_passes,
              'resource_policy': {'full_run_gpu_hour_ceiling': None, 'evaluation_gpu_hour_ceiling': None,
                                  'aggregate_gpu_hour_ceiling': None, 'pilot_aggregate_gpu_hours': 6,
                                  'sft_pilot_hours': 4, 'rl_pilot_hours': 2,
                                  'scientific_stopping_rules_retained': True},
              'warmup_selection': 'fixed_no_STOP_queries', 'evidence': evidence,
              'data_hashes': {name: digest(root / 'data' / name) for name in ('fit.json', 'stop.json', 'audit.json', 'preflight.json', 'target-mask-audit-v3.json')},
              'requirements_lock_sha256': digest(setup / 'requirements.lock'),
              'codebook_sha256': preflight['codebook_sha256'],
              'throughput_forecast_sha256': digest(root / 'throughput-forecast.json'),
              'model': preflight['model'], 'model_manifest_hash': preflight['model_manifest_hash'],
              'prompt_hashes': preflight['prompt_hashes'], 'frames': 8, 'view': '448_letterbox',
              'completion_cap': 512, 'context_cap': 8192, 'precision': 'bf16',
              'lora': {'rank': 16, 'alpha': 32, 'dropout': 0., 'targets': 'language_attention_q_k_v_o_only'},
              'target_mask': 'known-schema-unknown-payload-v2',
              'rollout_policy': {'version': SAMPLING_VERSION, 'temperature': TEMPERATURE, 'top_p': 1., 'top_k': 0,
                                 'scoring': 'same temperature-scaled full-support policy for old/current/reference probabilities and entropy'},
              'sft_loss': 'mean_label_and_known_schema + 0.5 * mean_known_box_payload; tokens touching unknown payload masked',
              'reward_version': 'hierarchical-iou-v1', 'reward_coefficients': {'C': .7, 'CG': .2, 'CGA': .1, 'grounded_outcome': .75, 'gated_Q': .25},
              'code_hashes': {p.name: digest(p) for p in module.glob('*.py')},
              'test_policy': 'sealed until complete selection manifest and explicit frozen-manifest argument',
              'validation_policy': 'historically used Phase 1; reporting only after selection freeze'}
    record['protocol_hash'] = canonical_hash(record)
    write_json(output, record, immutable=True)
    return record


def validate(path, data_root, model_path=None, codebook_path='studies/firearm_training/codebook.json'):
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
    if digest(codebook_path) != protocol['codebook_sha256']:
        raise ValueError('codebook differs from frozen protocol')
    lock_path = Path(data_root).parent / 'setup' / 'requirements.lock'
    if digest(lock_path) != protocol['requirements_lock_sha256']:
        raise ValueError('environment lock changed after freeze')
    from importlib.metadata import version
    for line in lock_path.read_text().splitlines():
        name, expected = line.split()[0].split('==')
        if version(name) != expected:
            raise ValueError(f'installed environment differs from tested lock: {name}')
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
