#!/usr/bin/env python3
"""Firearm A/G/AG study on top of the configurable runner. See studies/firearm/README.md.

  prepare   inventory, staged IDs and runner registry for the VISILAB dataset
  audit     validate the existing boxes and draw visual-check montages
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
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
