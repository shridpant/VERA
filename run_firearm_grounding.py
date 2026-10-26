#!/usr/bin/env python3
"""Right for the right reasons: grounding measures and grounding feedback in VERA's verbalized learning.
Design: docs/superpowers/specs/2026-10-26-firearm-grounding-design.md

  phase0    real-model checks: our generate == batch_chat, eager replay == greedy tokens, eager == flash scores,
            token-grid orientation by vision-feature occlusion
  measure   M1-M4 for one question set on chosen splits (resumable JSONL)
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
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
