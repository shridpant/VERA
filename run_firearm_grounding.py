#!/usr/bin/env python3
"""Right for the right reasons: grounding measures and grounding feedback in VERA's verbalized learning.
Design: docs/superpowers/specs/2026-10-26-firearm-grounding-design.md

  phase0    real-model checks: our generate == batch_chat, eager replay == greedy tokens, eager == flash scores,
            token-grid orientation by vision-feature occlusion
  measure   M1-M4 for one question set on chosen splits (resumable JSONL)
  gate      Phase 1 validity gate (V1 true vs shuffled boxes, V2 attention-occlusion agreement) and the
            training-split lift pool used by the grounded and placebo arms
  train-arm Phase 2: one arm (T1, G or P) with one training-order seed, then nothing else (resumable)
  evaluate  the selected questions of a trained arm on the test split (M1-M4)
  adopt-t1  Training 1's own run as the T1 arm's first seed (VERA's selection rule on its validation log)
  analyze   tables R1-R7, figures and number macros from the saved artifacts (no model; needs matplotlib)
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def config(path):
    return json.loads(Path(path).read_text())


def open_scorer(cfg, out_dir):
    import subprocess, time
    from importlib import metadata
    from vera_runner.backends import BACKENDS
    from vera_runner.config import load as load_runner, prepare_environment
    from vera_runner.reliance import Scorer
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'runner_config.json').write_text(json.dumps(cfg['runner'], indent=2) + '\n')
    runner = load_runner(out_dir / 'runner_config.json')
    prepare_environment(runner)
    begin = time.monotonic()
    backend = BACKENDS['internvl2'](runner)
    book = json.loads((ROOT / cfg['codebook']).read_text())
    scorer = Scorer(backend, runner['generation'], list(book['categories']))
    scorer.book = book
    versions = {}
    for name in ('torch', 'transformers', 'flash-attn', 'numpy', 'pillow'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    info = {'model_identity': backend.model_identity, 'library_versions': versions, 'model_load_seconds': time.monotonic() - begin,
            'code_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            'git_status': subprocess.run(['git', 'status', '--short'], cwd=ROOT, capture_output=True, text=True).stdout,
            'cuda_visible_devices': __import__('os').environ.get('CUDA_VISIBLE_DEVICES')}
    return scorer, backend, info


def phase0_command(args):
    from studies.firearm_grounding.phase0 import run_phase0
    cfg = config(args.config)
    out = Path(cfg['root']) / 'phase0'
    scorer, backend, info = open_scorer(cfg, out)
    result = run_phase0(scorer, backend, cfg, ROOT, out, args.videos)
    (out / 'phase0.json').write_text(json.dumps({**result, 'session': info}, indent=1, default=str) + '\n')
    print(json.dumps(result['checks'], indent=1))
    return 0 if all(c['passed'] for c in result['checks'].values()) else 1


def measure_command(args):
    from studies.firearm_grounding.measure import Measurer
    cfg = config(args.config)
    out = Path(args.out)
    scorer, backend, info = open_scorer(cfg, out.parent)
    questions = Path(args.questions).read_text() if args.questions else (ROOT / cfg['seed_questions_file']).read_text()
    (out.parent / f'{out.stem}.session.json').write_text(json.dumps({**info, 'questions': questions, 'splits': args.splits}, indent=1, default=str) + '\n')
    measurer = Measurer(scorer, cfg['data']['dir'], cfg['data']['frames'], cfg['data']['dir'], (ROOT / cfg['learner_instruct_file']).read_text())
    measurer.run(args.splits.split(','), questions, out, occlusion=not args.no_occlusion, limit=args.limit)
    print(json.dumps({'out': str(out)}))
    return 0


def gate_command(args):
    from vera_runner.ledger import read_jsonl
    from studies.firearm_grounding.arms import phase1_summary
    from studies.firearm_grounding.gate import gate
    cfg = config(args.config)
    root = Path(cfg['root'])
    result = gate(args.measurements, root / 'gate.json')
    summary = phase1_summary(read_jsonl(args.measurements)[0])
    (root / 'phase1_pool.json').write_text(json.dumps(summary) + '\n')
    print(json.dumps({k: v for k, v in result.items()}, indent=1))
    return 0 if result['gate_passed'] else 3


def arm_paths(cfg, arm, seed):
    return Path(cfg['root']) / 'arms' / f'{arm}_s{seed}'


def train_arm_command(args):
    import fcntl
    import run_firearm_training as t1
    from studies.firearm_grounding.arms import GroundedRun, grounded_template
    gcfg = config(args.config)
    root = Path(gcfg['root'])
    gate = json.loads((root / 'gate.json').read_text())
    if not gate['gate_passed'] and not args.ignore_gate:
        raise SystemExit('Phase 1 gate did not pass; Phase 2 does not run (pre-registration)')
    phase1 = json.loads((root / 'phase1_pool.json').read_text())
    run_dir = arm_paths(gcfg, args.arm, args.seed)
    cfg = t1.load_config(ROOT / 'studies/firearm_training/training1.json',
                         {'run_dir': str(run_dir), 'experiment_id': f'grounding_{args.arm}_s{args.seed}', 'training.seed': args.seed,
                          'training.eval_limit': args.eval_limit, 'training.val_check_interval': args.val_check_interval})
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / 'run.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec, runner = t1.build_spec(cfg)
    spec['grounding'] = {'arm': args.arm, 'phase1_pool_sha256': __import__('hashlib').sha256((root / 'phase1_pool.json').read_bytes()).hexdigest(),
                         'optimizer_template': grounded_template(spec['prompts']['optimizer_instruct']) if args.arm != 'T1' else None,
                         'design': 'docs/superpowers/specs/2026-10-26-firearm-grounding-design.md'}
    sdir, info = t1.session_for(run_dir, spec)
    status = 'failed'
    try:
        model = t1.open_model(spec, runner, info, sdir)
        from vera_runner.reliance import Scorer
        book = json.loads((ROOT / cfg['training']['codebook']).read_text())
        scorer = Scorer(model.b, runner['generation'], list(book['categories']))
        boxes = {}
        for s in ('train', 'val'):  # never the test boxes during training
            boxes.update(json.loads((Path(cfg['data']['dir']) / f'boxes_{s}.json').read_text()))
        run = GroundedRun(spec, run_dir, cfg['data']['dir'], cfg['data']['frames'], model, book, scorer, boxes, args.arm, phase1)
        status = run.train(sdir.name, stop_after=args.max_steps)
        if status == 'complete':
            best = run.select()
            (run_dir / 'selected.json').write_text(json.dumps({'step': best['step'], 'questions': best['questions'],
                                                               'metrics': best['metrics']}, indent=1) + '\n')
        return 0
    finally:
        import datetime
        info.update(status=status, finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (sdir / 'session.json').write_text(json.dumps(info, indent=1, default=str) + '\n')
        print(json.dumps({'session': sdir.name, 'status': status}))


def evaluate_command(args):
    from studies.firearm_grounding.measure import Measurer
    gcfg = config(args.config)
    selected = json.loads((Path(args.run_dir) / 'selected.json').read_text())
    out = Path(args.run_dir) / 'test_measurements.jsonl'
    scorer, backend, info = open_scorer(gcfg, Path(args.run_dir))
    (Path(args.run_dir) / 'test_measurements.session.json').write_text(json.dumps({**info, 'selected': selected}, indent=1, default=str) + '\n')
    measurer = Measurer(scorer, gcfg['data']['dir'], gcfg['data']['frames'], gcfg['data']['dir'], (ROOT / gcfg['learner_instruct_file']).read_text())
    measurer.run(['test'], selected['questions'], out, limit=args.limit)
    print(json.dumps({'out': str(out)}))
    return 0


def adopt_t1_command(args):
    from vera_runner.ledger import read_jsonl
    gcfg = config(args.config)
    src = Path(args.training_run)
    steps = [s for s in read_jsonl(src / 'steps.jsonl')[0] if s['status'] == 'complete']
    spec = json.loads((src / 'spec.json').read_text())
    n_train = len(json.loads((Path(gcfg['data']['dir']) / 'train.json').read_text()))
    expected = spec['training']['max_epochs'] * -(-n_train // spec['training']['batch_size'])
    if len(steps) < expected:
        print(json.dumps({'adopted': False, 'reason': f'Training 1 has {len(steps)} of {expected} steps'}))
        return 4
    vals = [v for v in read_jsonl(src / 'validations.jsonl')[0] if not v['reference_only']]
    best = max(vals, key=lambda v: (v['metrics']['category_accuracy'], v['metrics']['group_accuracy'], -v['step']))
    dest = arm_paths(gcfg, 'T1', spec['training']['seed'])
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'selected.json').write_text(json.dumps({'step': best['step'], 'questions': best['questions'], 'metrics': best['metrics'],
                                                   'adopted_from': str(src), 'spec_sha256': spec['spec_sha256']}, indent=1) + '\n')
    print(json.dumps({'adopted': True, 'step': best['step'], 'dest': str(dest)}))
    return 0


def analyze_command(args):
    from studies.firearm_grounding.analysis import analyze
    cfg = config(args.config)
    out = Path(args.out or Path(cfg['root']) / 'analysis')
    manifest = analyze(cfg['root'], cfg['data']['dir'], ROOT / 'studies/firearm/annotation_policy.json', out)
    print(json.dumps({**manifest, 'out': str(out)}, indent=1))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('phase0')
    p.add_argument('--config', default='studies/firearm_grounding/grounding.json')
    p.add_argument('--videos', type=int, default=4)
    p.set_defaults(func=phase0_command)
    p = sub.add_parser('measure')
    p.add_argument('--config', default='studies/firearm_grounding/grounding.json')
    p.add_argument('--splits', default='train,val,test')
    p.add_argument('--questions', help='file with the guiding questions; default: the seed questions')
    p.add_argument('--out', required=True, help='JSONL of per-video records (resumed if it exists)')
    p.add_argument('--limit', type=int)
    p.add_argument('--no-occlusion', action='store_true')
    p.set_defaults(func=measure_command)
    p = sub.add_parser('gate')
    p.add_argument('--config', default='studies/firearm_grounding/grounding.json')
    p.add_argument('--measurements', required=True)
    p.set_defaults(func=gate_command)
    p = sub.add_parser('train-arm')
    p.add_argument('--config', default='studies/firearm_grounding/grounding.json')
    p.add_argument('--arm', choices=['T1', 'G', 'P'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--eval-limit', type=int)
    p.add_argument('--val-check-interval', type=int)
    p.add_argument('--ignore-gate', action='store_true', help='smoke runs only')
    p.set_defaults(func=train_arm_command)
    p = sub.add_parser('evaluate')
    p.add_argument('--config', default='studies/firearm_grounding/grounding.json')
    p.add_argument('--run-dir', required=True)
    p.add_argument('--limit', type=int)
    p.set_defaults(func=evaluate_command)
    p = sub.add_parser('adopt-t1')
    p.add_argument('--config', default='studies/firearm_grounding/grounding.json')
    p.add_argument('--training-run', required=True)
    p.set_defaults(func=adopt_t1_command)
    p = sub.add_parser('analyze')
    p.add_argument('--config', default='studies/firearm_grounding/grounding.json')
    p.add_argument('--out', help='default: <root>/analysis')
    p.set_defaults(func=analyze_command)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
