#!/usr/bin/env python3
"""The firearm VERA package (docs/superpowers/specs/2026-10-26-firearm-package-design.md).

  manifest    E1 frame manifest from the non-test videos (with the 300/100 subsets)
  probe       box-format probe: a few frames, every candidate prompt, raw text kept
  detect      E1 detector ablation, one arm (resumable)
  recognise   E2 recognition test, one arm (resumable)
  analyze1    Phase 1 results and the gates G1/G2 (docs/superpowers/specs/2026-10-26-firearm-package-phase1-rules.md)
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / 'studies/firearm_package/package.json'


def config(path):
    return json.loads(Path(path or CONFIG).read_text())


def arms(name):
    return json.loads((ROOT / f'studies/firearm_package/{name}_arms.json').read_text())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', help='package.json (default: the committed one)')
    sub = ap.add_subparsers(dest='command', required=True)
    sub.add_parser('manifest')
    for name in ('probe', 'detect'):
        p = sub.add_parser(name)
        p.add_argument('--arm', required=True)
        p.add_argument('--limit', type=int, help='at most N new frames (smoke runs)')
        p.add_argument('--out', help='override the output folder (smoke runs)')
    p = sub.add_parser('recognise')
    p.add_argument('--arm', required=True)
    p.add_argument('--limit', type=int)
    p.add_argument('--out')
    sub.add_parser('analyze1')
    args = ap.parse_args(argv)
    cfg = config(args.config)
    out_root = Path(cfg['out_root'])
    manifest_path = out_root / 'e1_manifest.json'
    if args.command == 'manifest':
        from studies.firearm_package.manifest import write
        print(json.dumps(write(cfg, manifest_path), indent=1))
    elif args.command == 'probe':
        from studies.firearm_package.e1 import probe
        rows = probe(cfg, args.arm, arms('e1')[args.arm], manifest_path, Path(args.out or out_root / 'probes') / f'{args.arm}.json')
        print('\n'.join(json.dumps(r) for r in rows))
    elif args.command == 'detect':
        from studies.firearm_package.e1 import run_arm
        print(json.dumps(run_arm(cfg, args.arm, arms('e1')[args.arm], manifest_path, args.out or out_root / 'e1' / args.arm,
                                 limit=args.limit)))
    elif args.command == 'recognise':
        from studies.firearm_package.e2 import run_arm
        print(json.dumps(run_arm(cfg, args.arm, arms('e2')[args.arm], args.out or out_root / 'e2' / args.arm, ROOT, limit=args.limit)))
    elif args.command == 'analyze1':
        from studies.firearm_package.analysis_phase1 import analyze_e1, analyze_e2
        rules = {'min_recall50': 0.5, 'max_whole_image': 0.1, 'max_nogun_fp': 0.1}
        e1r, g1 = analyze_e1(out_root, manifest_path, cfg['seed_measurements'], list(arms('e1')), rules)
        e2r, g2 = analyze_e2(out_root, list(arms('e2')))
        (out_root / 'analysis').mkdir(parents=True, exist_ok=True)
        (out_root / 'analysis/phase1.json').write_text(json.dumps({'e1': e1r, 'g1': g1, 'e2': e2r, 'g2': g2}, indent=1) + '\n')
        print(json.dumps({'g1': g1, 'g2': g2}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
