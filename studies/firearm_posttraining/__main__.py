import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description='Isolated firearm post-training study')
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('audit')
    a.add_argument('--data-dir', required=True)
    a.add_argument('--frames-root', required=True)
    a.add_argument('--output', required=True)
    a.add_argument('--seed', type=int, default=20261026)
    for command in ('smoke', 'train', 'rl-panel'):
        a = sub.add_parser(command)
        a.add_argument('--model', required=True)
        a.add_argument('--data', required=True)
        a.add_argument('--codebook', default='studies/firearm_training/codebook.json')
        a.add_argument('--output', required=True)
        a.add_argument('--seed', type=int, default=20261026)
        a.add_argument('--grounded', action='store_true')
        if command == 'train':
            a.add_argument('--method', choices=('SL', 'SG', 'RL', 'RG'), default='SL')
            a.add_argument('--adapter')
            a.add_argument('--resume')
            a.add_argument('--max-seconds', type=float, default=None, help='default: 4h exploratory pilot; 0 for frozen comparison')
            a.add_argument('--max-passes', type=float, default=5.)
            a.add_argument('--accumulation', type=int, default=8)
            a.add_argument('--warmup-only', action='store_true')
            a.add_argument('--warmup-passes', type=float, default=.5)
            a.add_argument('--frozen-protocol')
        if command == 'rl-panel':
            a.add_argument('--adapter', required=True)
            a.add_argument('--max-seconds', type=float, default=7200)
    a = sub.add_parser('vq')
    a.set_defaults(method='VQ')
    a.add_argument('--model', required=True)
    a.add_argument('--data', required=True)
    a.add_argument('--output', required=True)
    a.add_argument('--frozen-protocol', required=True)
    a.add_argument('--codebook', default='studies/firearm_training/codebook.json')
    a.add_argument('--seed', type=int, required=True)
    a.add_argument('--max-seconds', type=float, default=0)
    a = sub.add_parser('holdout')
    a.add_argument('--frozen-manifest', required=True)
    a.add_argument('--data', required=True)
    a.add_argument('--original-data', required=True)
    a.add_argument('--frames-root', required=True)
    a.add_argument('--split', choices=('val', 'test'), required=True)
    a.add_argument('--method', choices=('B0', 'VQ', 'SL', 'RL', 'SG', 'RG'), required=True)
    a.add_argument('--seed', type=int, default=20261026)
    a.add_argument('--output', required=True)
    a.add_argument('--codebook', default='studies/firearm_training/codebook.json')
    args = p.parse_args()
    if args.command == 'audit':
        from .data import audit
        print(json.dumps(audit(args.data_dir, args.frames_root, args.output, args.seed), indent=2))
    elif args.command == 'smoke':
        from .smoke import run
        run(args)
    elif args.command == 'train':
        from .trainer import run
        run(args)
    elif args.command == 'rl-panel':
        from .rl_panel import run
        run(args)
    elif args.command == 'vq':
        from .verbalized import run
        run(args)
    elif args.command == 'holdout':
        from .holdout import run
        run(args)


if __name__ == '__main__':
    main()
