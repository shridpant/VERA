#!/usr/bin/env python3
"""Training 1: VERA's verbalized learning on the VISILAB firearm dataset. See studies/firearm_training/README.md.

  prepare   leak-free split, per-split label files and Training 2 box files
  train     VERA's verbalized learning: learner + optimizer, validation every 100 batches (resumable)
  test      the seed and the selected questions on the test split
  analyze   tables S1-S10, figures F1-F4, qualitative answers and the report numbers
"""
import argparse
import json
from pathlib import Path
import sys


def prepare_command(args):
    from studies.firearm_training.prepare import prepare
    here = Path(__file__).resolve().parent / 'studies/firearm_training'
    result = prepare(args.inventory, here / 'codebook.json', Path(__file__).resolve().parent / 'studies/firearm/annotation_policy.json',
                     args.original, args.out, args.seed)
    print(json.dumps({k: v for k, v in result.items() if k != 'splits'}, indent=1))
    print(json.dumps({s: {k: v for k, v in r.items() if k != 'sha256'} for s, r in result['splits'].items()}, indent=1))
    return 0 if not result['groups_spanning_splits'] and not result['codes_missing_from_a_split'] else 1


ROOT = Path(__file__).resolve().parent


def load_config(path, overrides):
    cfg = json.loads(Path(path).read_text())
    for key, value in overrides.items():
        if value is not None:
            node = cfg
            *head, last = key.split('.')
            for h in head:
                node = node[h]
            node[last] = value
    return cfg


def build_spec(cfg):
    import hashlib
    from vera_runner.config import load as load_runner
    tr, data = cfg['training'], Path(cfg['data']['dir'])
    read = lambda rel: (ROOT / rel).read_text()
    runner_file = Path(cfg['run_dir']) / 'runner_config.json'
    runner_file.parent.mkdir(parents=True, exist_ok=True)
    runner_file.write_text(json.dumps(cfg['runner'], indent=2) + '\n')
    runner = load_runner(runner_file)  # the runner's own validation of the model and generation settings
    spec = {'experiment_id': cfg['experiment_id'], 'runner': runner,
            'training': {k: tr[k] for k in ('batch_size', 'frames_per_video', 'max_epochs', 'val_check_interval', 'seed')},
            'prompts': {'learner_instruct': read(tr['learner_instruct_file']), 'optimizer_instruct': read(tr['optimizer_instruct_file'])},
            'codebook_sha256': hashlib.sha256((ROOT / tr['codebook']).read_bytes()).hexdigest(),
            'splits_sha256': {s: hashlib.sha256((data / f'{s}.json').read_bytes()).hexdigest() for s in ('train', 'val', 'test')}}
    spec['training']['seed_questions'] = read(tr['seed_questions_file'])
    spec['training']['eval_limit'] = tr.get('eval_limit')  # smoke runs only; None evaluates every video
    method = tr.get('method', 'vera')
    if method != 'vera':  # variants only: the faithful runs' specs stay exactly as before
        spec['training']['method'] = method
        for key in VARIANT_KEYS[method]:
            spec['training'][key] = tr[key]
        for name in ('bank_proposer', 'yes_no'):
            if f'{name}_file' in tr:
                spec['prompts'][name] = read(tr[f'{name}_file'])
    return spec, runner


VARIANT_KEYS = {'vera_fixed': ('videos_per_class', 'check_per_class', 'history', 'val_every_steps')}
VARIANT_KEYS.update({m: ('rounds', 'refresh_every', 'score_per_class', 'keep_threshold', 'top_k', 'per_pair_cap')
                     for m in ('bank', 'bank_random', 'bank_noscore')})


def session_for(run_dir, spec):
    import datetime, os, subprocess
    from vera_runner.verbalized import digest
    spec_path = run_dir / 'spec.json'
    if spec_path.exists():
        old = json.loads(spec_path.read_text())
        if old['spec_sha256'] != digest({k: v for k, v in spec.items()}):
            raise ValueError(f'{spec_path} differs from this configuration; use a new run_dir')
    else:
        spec_path.write_text(json.dumps({**spec, 'spec_sha256': digest(spec)}, indent=1) + '\n')
        os.chmod(spec_path, 0o444)
    sessions = run_dir / 'sessions'
    sessions.mkdir(exist_ok=True)
    n = 1 + max([int(p.name) for p in sessions.iterdir() if p.name.isdigit()], default=0)
    sdir = sessions / f'{n:03d}'
    sdir.mkdir()
    git = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout
    info = {'session': sdir.name, 'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'pid': os.getpid(),
            'code_commit': git('rev-parse', 'HEAD').strip(), 'git_status': git('status', '--short'),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
    try:
        info['gpu'] = subprocess.run(['nvidia-smi', '--id=' + info['cuda_visible_devices'], '--query-gpu=uuid,name,driver_version,memory.total',
                                      '--format=csv,noheader'], capture_output=True, text=True).stdout.strip() if info['cuda_visible_devices'] else None
    except OSError:
        info['gpu'] = None
    return sdir, info


def open_model(spec, runner, info, sdir):
    import subprocess, time
    from importlib import metadata
    from vera_runner.backends import BACKENDS
    from vera_runner.config import prepare_environment
    from vera_runner.verbalized import InternVL2Calls
    prepare_environment(runner)
    begin = time.monotonic()
    backend = BACKENDS[runner['model']['backend']](runner)
    info['model_load_seconds'] = time.monotonic() - begin
    info['model_identity'] = getattr(backend, 'model_identity', None)
    versions = {}
    for name in ('torch', 'transformers', 'flash-attn', 'accelerate', 'timm', 'pillow', 'opencv-python-headless', 'numpy'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    info['library_versions'] = versions
    (sdir / 'environment.txt').write_text(subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True).stdout)
    return InternVL2Calls(backend, runner['generation'])


def train_command(args, test=False):
    import fcntl
    from vera_runner.verbalized import Run
    from vera_runner.verbalized_variants import VARIANTS
    cfg = load_config(args.config, {'run_dir': args.run_dir, 'experiment_id': args.experiment_id,
                                    'training.val_check_interval': args.val_check_interval, 'training.eval_limit': args.eval_limit})
    run_dir = Path(cfg['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / 'run.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec, runner = build_spec(cfg)
    sdir, info = session_for(run_dir, spec)
    status = 'failed'
    try:
        model = open_model(spec, runner, info, sdir)
        book = json.loads((ROOT / cfg['training']['codebook']).read_text())
        method = spec['training'].get('method', 'vera')
        run = (Run if method == 'vera' else VARIANTS[method])(spec, run_dir, cfg['data']['dir'], cfg['data']['frames'], model, book)
        if test:
            best = run.test(sdir.name)
            status = f"tested (selected step {best['step']})"
        else:
            status = run.train(sdir.name, stop_after=args.max_steps)
        return 0
    finally:
        import datetime
        info.update(status=status, finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (sdir / 'session.json').write_text(json.dumps(info, indent=1) + '\n')
        print(json.dumps({'session': sdir.name, 'status': status}))


def check_spec_command(args):
    """Does this code still build exactly the locked spec of an existing run? (Needed before resuming it.)"""
    from vera_runner.verbalized import digest
    cfg = load_config(args.config, {})
    spec, _ = build_spec(cfg)
    stored = json.loads((Path(cfg['run_dir']) / 'spec.json').read_text())['spec_sha256']
    same = digest(spec) == stored
    print(json.dumps({'run_dir': cfg['run_dir'], 'spec_matches': same}))
    return 0 if same else 1


def analyze_command(args):
    from studies.firearm_training.analysis.run import analyze
    cfg = load_config(args.config, {'run_dir': args.run_dir})
    manifest = analyze(cfg['run_dir'], cfg['data']['dir'], ROOT / cfg['training']['codebook'], args.out)
    print(json.dumps(manifest, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='split, label files and box files')
    p.add_argument('--inventory', required=True, help='inventory.json from run_firearm_study-style prepare (firearm_data)')
    p.add_argument('--original', required=True, help='extracted Gun_Action_Recognition_Dataset folder')
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=20261026)
    p.set_defaults(func=prepare_command)
    for name, func in (('train', lambda a: train_command(a)), ('test', lambda a: train_command(a, test=True))):
        p = sub.add_parser(name)
        p.add_argument('--config', required=True)
        p.add_argument('--run-dir', help='override run_dir (smoke runs)')
        p.add_argument('--experiment-id', help='override experiment_id (smoke runs)')
        p.add_argument('--max-steps', type=int, help='stop after this many new training steps (smoke runs)')
        p.add_argument('--val-check-interval', type=int, help='override training.val_check_interval (smoke runs)')
        p.add_argument('--eval-limit', type=int, help='evaluate only the first N videos of val/test (smoke runs)')
        p.set_defaults(func=func)
    p = sub.add_parser('check-spec')
    p.add_argument('--config', required=True)
    p.set_defaults(func=check_spec_command)
    p = sub.add_parser('analyze')
    p.add_argument('--config', required=True)
    p.add_argument('--run-dir', help='override run_dir')
    p.add_argument('--out', required=True)
    p.set_defaults(func=analyze_command)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
