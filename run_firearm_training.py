#!/usr/bin/env python3
"""Training 1: VERA's verbalized learning on the VISILAB firearm dataset. See studies/firearm_training/README.md.

  prepare   leak-free split, per-split label files and Training 2 box files
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='split, label files and box files')
    p.add_argument('--inventory', required=True, help='inventory.json from run_firearm_study-style prepare (firearm_data)')
    p.add_argument('--original', required=True, help='extracted Gun_Action_Recognition_Dataset folder')
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=20261026)
    p.set_defaults(func=prepare_command)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
