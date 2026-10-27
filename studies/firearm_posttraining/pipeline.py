"""Detached CPU controller: scientific sweep -> frozen selection -> evaluation/report."""
import argparse
import json
import os
import time
from pathlib import Path
from .data import write_json
from .protocol import validate
from .sweep import run as sweep
from .finish import run as finish


def run(root, protocol, original_data, frames_root):
    root = Path(root)
    protocol = str(Path(protocol).resolve())
    validate(protocol, root / 'data')
    control = root / 'pipeline'
    control.mkdir(parents=True, exist_ok=False)
    started = time.time()
    def status(stage, **extra):
        write_json(control / 'status.json', {'stage': stage, 'started_unix': started,
                   'updated_unix': time.time(), 'controller_pid': os.getpid(), **extra})
    try:
        status('sweep_running')
        selected = sweep(root, protocol)
        status('evaluation_running', selection_hash=selected['selection_hash'])
        finish(root, root / 'selection.json', original_data, frames_root)
        report = json.loads((root / 'report/status.json').read_text())
        status('complete' if report['status'] == 'complete' else 'additional_verification_required', report=report)
    except Exception as exc:
        status('failed', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--protocol', required=True)
    p.add_argument('--original-data', required=True)
    p.add_argument('--frames-root', required=True)
    args = p.parse_args()
    run(args.root, args.protocol, args.original_data, args.frames_root)
