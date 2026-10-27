"""Two independent single-GPU workers; shared fixed warm starts and fail-closed jobs."""
import argparse
import concurrent.futures
import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from .data import digest, write_json
from .execute import ALLOWED
from .protocol import validate, freeze_selection, SEEDS


def run(root, protocol_path):
    root = Path(root)
    protocol_path = str(Path(protocol_path).resolve())
    protocol = validate(protocol_path, root / 'data')
    output = root / 'sweep'
    output.mkdir(parents=True, exist_ok=False)
    available = queue.Queue()
    for gpu in ALLOWED:
        available.put(gpu)
    records = []
    failure = threading.Event()
    def launch(name, command, seconds):
        gpu = available.get()
        try:
            if failure.is_set():
                raise RuntimeError('sweep halted after another job failure')
            full = [sys.executable, '-m', 'studies.firearm_posttraining.execute', '--root', str(root),
                    '--gpu', gpu, '--name', name, '--stage', 'sweep', '--seconds', str(seconds), '--',
                    sys.executable, '-m', 'studies.firearm_posttraining', *command]
            with (output / f'{name}.supervisor.log').open('x') as log:
                subprocess.run(full, stdout=log, stderr=subprocess.STDOUT, check=True)
            directory = output / name
            result = json.loads((directory / 'result.json').read_text())
            if result.get('status') != 'stopped':
                raise RuntimeError(f'incomplete scientific run: {name}')
            return directory, result
        except Exception:
            failure.set()
            raise
        finally:
            available.put(gpu)
    common = ['--model', protocol['model'], '--data', str(root / 'data'),
              '--frozen-protocol', protocol_path, '--max-passes', str(protocol['shared_max_fit_passes'])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for seed in SEEDS:
            warm = {}
            futures = {}
            for kind, method in (('L', 'SL'), ('G', 'SG')):
                name = f'C0{kind}-{seed}'
                command = ['train', *common, '--method', method, '--seed', str(seed),
                           '--output', str(output / name), '--warmup-only', '--max-seconds', '0']
                if kind == 'G':
                    command.append('--grounded')
                futures[executor.submit(launch, name, command, 0)] = kind
            for future in concurrent.futures.as_completed(futures):
                kind = futures[future]
                directory, result = future.result()
                if result['stop_reason'] != 'fixed_warmup_complete' or result['fit_passes'] != .5:
                    raise RuntimeError('warm start did not complete exactly half a FIT pass')
                adapter = result['best_checkpoint']
                warm[kind] = {'adapter': adapter, 'elapsed_seconds': result['elapsed_seconds'],
                              'adapter_sha256': digest(Path(adapter) / 'adapter_model.safetensors')}
            write_json(output / f'paired-initializations-{seed}.json', warm, immutable=True)
            futures = {}
            for method in ('SL', 'RL', 'SG', 'RG', 'VQ'):
                name = f'{method}-{seed}'
                if method == 'VQ':
                    command = ['vq', '--model', protocol['model'], '--data', str(root / 'data'),
                               '--frozen-protocol', protocol_path, '--seed', str(seed),
                               '--output', str(output / name), '--max-seconds', '0']
                    seconds = 0
                else:
                    init = warm['G' if method in ('SG', 'RG') else 'L']
                    # Continue recording warm-up time for both arms; user removed
                    # scientific-run GPU-hour ceilings. Scientific stops still apply.
                    seconds = 0
                    command = ['train', *common, '--method', method, '--seed', str(seed),
                               '--adapter', init['adapter'], '--output', str(output / name),
                               '--max-seconds', '0']
                    if method in ('SG', 'RG'):
                        command.append('--grounded')
                futures[executor.submit(launch, name, command, seconds)] = method
            for future in concurrent.futures.as_completed(futures):
                method = futures[future]
                directory, result = future.result()
                records.append(str(directory))
                write_json(output / 'progress.json', {'completed': records, 'last_method': method,
                           'last_seed': seed, 'last_stop_reason': result['stop_reason']})
    frozen = freeze_selection(protocol_path, root / 'data', records, root / 'selection.json')
    write_json(output / 'result.json', {'status': 'complete', 'runs': records, 'selection_hash': frozen['selection_hash']})
    return frozen


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--protocol', required=True)
    args = p.parse_args()
    run(args.root, args.protocol)
