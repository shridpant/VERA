"""Frozen evaluation jobs on two independent workers, then deterministic reporting."""
import argparse
import concurrent.futures
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from .data import write_json
from .execute import ALLOWED
from .protocol import verify_selection, METHODS, SEEDS
from .report import run as report


def run(root, frozen_manifest, original_data, frames_root):
    root = Path(root)
    frozen_manifest = str(Path(frozen_manifest).resolve())
    verify_selection(frozen_manifest, root / 'data')
    output = root / 'evaluation'
    output.mkdir(parents=True, exist_ok=False)
    pool, failed = queue.Queue(), threading.Event()
    for gpu in ALLOWED:
        pool.put(gpu)
    def launch(split, method, seed):
        gpu = pool.get()
        try:
            if failed.is_set():
                raise RuntimeError('evaluation halted after another job failure')
            name = f'eval-{split}-{method}-{seed}'
            directory = output / split / f'{method}-{seed}'
            command = [sys.executable, '-m', 'studies.firearm_posttraining.execute', '--root', str(root),
                       '--gpu', gpu, '--name', name, '--stage', 'evaluation', '--seconds', '2400', '--',
                       sys.executable, '-m', 'studies.firearm_posttraining', 'holdout',
                       '--frozen-manifest', frozen_manifest, '--data', str(root / 'data'),
                       '--original-data', str(original_data), '--frames-root', str(frames_root),
                       '--split', split, '--method', method, '--seed', str(seed), '--output', str(directory)]
            with (output / f'{name}.supervisor.log').open('x') as f:
                subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
            result = json.loads((directory / 'result.json').read_text())
            if result.get('status') != 'complete':
                raise RuntimeError('incomplete holdout evaluation')
            return str(directory)
        except Exception:
            failed.set()
            raise
        finally:
            pool.put(gpu)
    completed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(launch, split, method, seed)
                   for split in ('val', 'test') for method in ('B0', *METHODS)
                   for seed in ((SEEDS[0],) if method == 'B0' else SEEDS)]
        for future in concurrent.futures.as_completed(futures):
            completed.append(future.result())
            write_json(output / 'progress.json', {'completed': completed})
    report(root, frozen_manifest, original_data, frames_root, root / 'report')
    write_json(output / 'result.json', {'status': 'complete', 'directories': completed})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--frozen-manifest', required=True)
    p.add_argument('--original-data', required=True)
    p.add_argument('--frames-root', required=True)
    a = p.parse_args()
    run(a.root, a.frozen_manifest, a.original_data, a.frames_root)
