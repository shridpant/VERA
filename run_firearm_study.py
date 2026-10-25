#!/usr/bin/env python3
"""Firearm A/G/AG study on top of the configurable runner. See studies/firearm/README.md.

  prepare   inventory, staged IDs and runner registry for the VISILAB dataset
  audit     validate the existing boxes and draw visual-check montages
  run       run or resume A/G/AG inference; with --videos/--max-windows/--run-dir a smoke run
  reconcile every scheduled unit with a disposition (complete, exhausted, pending)
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
    for name, func, text in (('run', run, 'run or resume A/G/AG inference (one GPU process)'),
                             ('reconcile', reconcile_command, 'give every scheduled unit a disposition')):
        p = sub.add_parser(name, help=text)
        p.add_argument('--config', required=True, help='study JSON (made by convert)')
        p.add_argument('--run-dir', help='override execution.run_dir (smoke runs)')
        p.add_argument('--videos', help='comma-separated staged IDs (smoke runs)')
        p.add_argument('--max-windows', type=int, help='first N windows per video (smoke runs)')
        p.add_argument('--study-id', help='override study_id (smoke runs)')
        p.set_defaults(func=func)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
