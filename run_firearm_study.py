#!/usr/bin/env python3
"""Firearm A/G/AG study on top of the configurable runner. See studies/firearm/README.md.

  prepare   inventory, staged IDs and runner registry for the VISILAB dataset
  audit     validate the existing boxes and draw visual-check montages
  convert   annotated YAML -> validated study JSON and the runner-only A config
  validate  preflight: config, prompts, registry, inventory, model files, GPU, storage
  run       run or resume A/G/AG inference; with --videos/--max-windows/--run-dir a smoke run
  reconcile every scheduled unit with a disposition (complete, exhausted, pending)
  features  ImageBind segment features (LAVAD procedure) from the run's decoded frames
  features-check  validation gate: our extractor vs VERA's released UCF features
"""
import argparse
import collections
import json
from pathlib import Path
import sys


def prepare(args):
    from vera_runner.firearm_data import build_inventory, stage, write_inventory, write_registry
    data = Path(args.data_root).resolve()
    rows = stage(build_inventory(args.original, args.workers), args.original, data / 'staged')
    write_inventory(rows, data)
    write_registry(data / 'registry.json', data / 'staged', data / 'features')
    mismatched = [(r['id'], r['frames_meta'], r['frames_decoded']) for r in rows if r['frames_meta'] != r['frames_decoded']]
    summary = {'videos': len(rows), 'by_category': collections.Counter(r['category'] for r in rows),
               'by_camera': collections.Counter(f"{r['category']}/{r['camera']}/{r['width']}x{r['height']}@{r['fps']:g}" for r in rows),
               'expected_windows': sum(r['expected_windows'] for r in rows),
               'frames_meta_vs_decoded_mismatches': mismatched,
               'label_files': collections.Counter(r['label_file'] for r in rows),
               'unexpected_files': {r['id']: r['unexpected_files'] for r in rows if r['unexpected_files']}}
    (data / 'inventory_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0


def audit(args):
    from studies.firearm.audit import run_audit
    validation, cover, manifest = run_audit(args.inventory, args.original, args.out)
    problems = {k: v for k, v in validation.items() if v['problems']}
    print(json.dumps({'label_files': len(validation), 'with_problems': problems,
                      'montages': {k: len(v) for k, v in manifest.items() if isinstance(v, list)}}, indent=2))
    return 0


def convert(args):
    """Annotated YAML -> study JSON (validated) plus the runner block alone as a run_vera.py config."""
    import yaml
    from vera_runner.study import load_study
    raw = yaml.safe_load(Path(args.yaml).read_text())
    load_study(raw)
    out = Path(args.out)
    out.write_text(json.dumps(raw, indent=2) + '\n')
    runner = out.with_name(out.stem + '.runner_A.json')
    runner.write_text(json.dumps(raw['runner'], indent=2) + '\n')
    print(f'{out}\n{runner}')
    return 0


def validate(args):
    """Preflight without a model call: config, prompts, registry, inventory, model files, GPU, storage."""
    import datetime
    import shutil
    from vera_runner.config import load, path
    from vera_runner.data import resolve
    from vera_runner.study import build_spec, digest, gpu_description, load_study
    study = overridden(args)
    checks, problems = {}, []
    spec, chosen = build_spec(study, path(study['data']['inventory']))
    checks['units'] = spec['units']
    checks['prompt_sha256'] = spec['prompt_sha256']
    runner_json = Path(args.config).with_name(Path(args.config).stem + '.runner_A.json')
    if runner_json.exists() and load(runner_json) != load_study(args.config)['runner']:
        problems.append(f'{runner_json} differs from the runner block')
    resolved = {item['id']: item['path'] for item in resolve(study['runner'])}
    missing = [row['id'] for row, _ in chosen if row['id'] not in resolved]
    if missing:
        problems.append(f'{len(missing)} selected videos are not resolvable through the registry: {missing[:5]}')
    model = study['runner']['model']
    if model['backend'] == 'internvl2':
        directory = path(model['path'])
        provenance = directory / 'MODEL_PROVENANCE.json'
        if not provenance.exists():
            problems.append(f'no MODEL_PROVENANCE.json in {directory}')
        else:
            prov = json.loads(provenance.read_text())
            checks['model_revision'] = prov.get('revision')
            if prov.get('revision') != model['revision']:
                problems.append('model provenance revision differs from the config')
            for name, expected in prov['files'].items():
                if not (directory / name).is_file() or (directory / name).stat().st_size != expected['bytes']:
                    problems.append(f'model file missing or truncated: {name}')
    run_dir = path(study['execution']['run_dir'])
    existing = run_dir
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    estimate = sum(len(w) for _, w in chosen) * 8 * 400_000 + sum(r['bytes'] for r, _ in chosen) * 12
    checks.update(free_bytes=free, storage_estimate_bytes=estimate, gpu=gpu_description())
    if free < 1.5 * estimate:
        problems.append(f'free space {free / 1e9:.1f} GB < 1.5 x estimate {estimate / 1e9:.1f} GB')
    result = {'checked_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'spec_sha256': digest(spec),
              'checks': checks, 'problems': problems, 'ok': not problems}
    out = run_dir.parent / 'checks'
    out.mkdir(parents=True, exist_ok=True)
    (out / f"validate_{study['study_id']}_{result['checked_at'][:19].replace(':', '')}.json").write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return 0 if not problems else 1


def weights_identity(weights):
    from vera_runner.data import sha256
    return {'path': str(weights), 'bytes': Path(weights).stat().st_size, 'sha256': sha256(weights)}


def package_versions():
    from importlib import metadata
    out = {}
    for name in ('imagebind', 'pytorchvideo', 'torch', 'torchvision', 'av'):
        try:
            dist = metadata.distribution(name)
            direct = dist.read_text('direct_url.json')
            out[name] = json.loads(direct) if direct else dist.version
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def features(args):
    """ImageBind features for every video of a finished run, from the run's own decoded frames."""
    import numpy as np
    from vera_runner.config import path
    from vera_runner.data import sha256
    from vera_runner.imagebind_features import Extractor, anchors, extract_video, write_manifest
    from vera_runner.study import digest
    study = overridden(args)
    settings, run_dir = study['features'], path(study['execution']['run_dir'])
    out = path(settings['dir'])
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((run_dir / 'spec.json').read_text())
    identity = {'weights': weights_identity(settings['weights']), 'packages': package_versions(),
                'settings': {k: settings[k] for k in ('clip_seconds', 'fps', 'num_samples')}, 'precision': 'float32'}
    key = digest(identity)
    extractor = Extractor(settings['weights'])
    videos = {}
    for v in spec['videos']:
        npy, side = out / f"{v['id']}.npy", out / f"{v['id']}.json"
        if side.exists() and json.loads(side.read_text()).get('identity_sha256') == key and npy.exists() \
                and sha256(npy) == json.loads(side.read_text())['npy_sha256']:
            videos[v['id']] = 'reused'
            continue
        frames = run_dir / 'videos' / v['id'] / 'frames'
        info = run_dir / 'videos' / v['id'] / 'video.json'
        if not info.exists() or json.loads(info.read_text())['decoded_frames'] != v['frames']:
            videos[v['id']] = 'missing_frames'
            continue
        began = __import__('time').monotonic()
        rows, used = extract_video(extractor, frames, v['frames'], settings, study['runner']['sampling']['stride'])
        if len(rows) != len(anchors(v['frames'], study['runner']['sampling']['stride'])):
            raise ValueError(f"{v['id']}: feature rows do not match the anchors")
        np.save(npy, rows)
        side.write_text(json.dumps({'video': v['id'], 'identity_sha256': key, 'npy_sha256': sha256(npy), 'rows': len(rows),
                                    'clip_frame_indices': used, 'seconds': __import__('time').monotonic() - began}) + '\n')
        videos[v['id']] = 'extracted'
        print(v['id'], len(rows), flush=True)
    write_manifest(out / 'features_manifest.json', {'identity': identity, 'identity_sha256': key, 'run_dir': str(run_dir), 'videos': videos})
    missing = [k for k, s in videos.items() if s == 'missing_frames']
    print(json.dumps({'videos': len(videos), 'missing_frames': missing}))
    return 0 if not missing else 1


def features_check(args):
    """Validation gate: our extractor on a UCF video vs the features VERA released for it."""
    import numpy as np
    from vera_runner.config import DEFAULTS
    from vera_runner.data import preprocess
    from vera_runner.imagebind_features import Extractor, compare, extract_video
    from vera_runner.refine import refine
    from vera_runner.study import load_study
    study = load_study(args.config)
    out = Path(args.out)
    frames = out / 'frames'
    if not (out / 'video.json').exists():
        out.mkdir(parents=True, exist_ok=True)
        info = preprocess({'id': Path(args.video).stem, 'path': args.video, 'annotation': {}}, frames)
        (out / 'video.json').write_text(json.dumps(info) + '\n')
    info = json.loads((out / 'video.json').read_text())
    settings = {**study['features'], 'fps': args.fps}
    rows, _ = extract_video(Extractor(settings['weights']), frames, info['decoded_frames'], settings)
    np.save(out / 'ours.npy', rows)
    released = np.load(args.released)
    report = {'video': args.video, 'released': args.released, 'decoded_frames': info['decoded_frames'],
              'weights': weights_identity(settings['weights']), 'packages': package_versions(), **compare(rows, released)}
    if args.labels:
        scores = json.loads(Path(args.labels).read_text())['anchors']
        records = [{'strict_label': s['score'], 'anchor': int(a), 'end': s['end']} for a, s in sorted(scores.items(), key=lambda x: int(x[0]))]
        a = refine(records, released, info['decoded_frames'], 16, DEFAULTS['refinement'])['frame_scores']
        b = refine(records, rows, info['decoded_frames'], 16, DEFAULTS['refinement'])['frame_scores']
        report['refined_frame_scores_max_abs_diff'] = float(np.abs(a - b).max())
        report['refined_frame_scores_identical'] = bool(np.array_equal(a, b))
    report['gate_passed'] = bool(report.get('same_shape') and report['cosine_median'] >= 0.99)
    (out / 'features_check.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))
    return 0 if report['gate_passed'] else 1


def overridden(args):
    """The study with command-line selection overrides (smoke runs); they are part of its spec."""
    from vera_runner.study import load_study
    raw = json.loads(Path(args.config).read_text())
    if args.run_dir:
        raw.setdefault('execution', {})['run_dir'] = args.run_dir
    sel = raw['runner'].setdefault('selection', {})
    if args.videos:
        sel['video_ids'] = args.videos.split(',')
    if args.max_windows:
        sel['max_windows_per_video'] = args.max_windows
    if args.study_id:
        raw['study_id'] = args.study_id
    return load_study(raw)


def run(args):
    import signal
    from vera_runner.study import run_study
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    code = run_study(overridden(args))
    print(json.dumps({'exit': code}))
    return code


def reconcile_command(args):
    from vera_runner.study import reconcile
    study = overridden(args)
    result = reconcile(study, study['execution']['run_dir'])
    print(json.dumps(result, indent=2))
    return 0 if result['counts'].get('pending', 0) == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='inventory, staged symlinks and registry')
    p.add_argument('--original', required=True, help='extracted Gun_Action_Recognition_Dataset folder (read-only)')
    p.add_argument('--data-root', required=True, help='study data folder; staged/, inventory and registry go here')
    p.add_argument('--workers', type=int, default=16)
    p.set_defaults(func=prepare)
    p = sub.add_parser('audit', help='validate existing boxes and draw visual-check montages')
    p.add_argument('--inventory', required=True)
    p.add_argument('--original', required=True)
    p.add_argument('--out', required=True)
    p.set_defaults(func=audit)
    p = sub.add_parser('convert', help='annotated YAML -> validated study JSON (+ runner_A.json)')
    p.add_argument('--yaml', required=True)
    p.add_argument('--out', required=True)
    p.set_defaults(func=convert)
    for name, func, text in (('validate', validate, 'preflight checks; no model call'),
                             ('run', run, 'run or resume A/G/AG inference (one GPU process)'),
                             ('reconcile', reconcile_command, 'give every scheduled unit a disposition')):
        p = sub.add_parser(name, help=text)
        p.add_argument('--config', required=True, help='study JSON (made by convert)')
        p.add_argument('--run-dir', help='override execution.run_dir (smoke runs)')
        p.add_argument('--videos', help='comma-separated staged IDs (smoke runs)')
        p.add_argument('--max-windows', type=int, help='first N windows per video (smoke runs)')
        p.add_argument('--study-id', help='override study_id (smoke runs)')
        p.set_defaults(func=func)
    p = sub.add_parser('features', help='ImageBind features (LAVAD procedure) from a run\'s decoded frames')
    p.add_argument('--config', required=True)
    p.set_defaults(func=features, run_dir=None, videos=None, max_windows=None, study_id=None)
    p = sub.add_parser('features-check', help='validation gate against a released VERA feature file')
    p.add_argument('--config', required=True)
    p.add_argument('--video', required=True)
    p.add_argument('--released', required=True)
    p.add_argument('--labels', help='experiment01 segment_scores.json for the refinement comparison')
    p.add_argument('--fps', type=float, default=30.0)
    p.add_argument('--out', required=True)
    p.set_defaults(func=features_check)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
