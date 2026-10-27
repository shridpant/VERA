"""Detached single-GPU command supervisor with shared aggregate budget reservations.

Usage: python -m studies.firearm_posttraining.execute --root ROOT --gpu UUID
       --name UNIQUE --stage pilot --seconds 900 -- python -m ... smoke ...
"""
import argparse
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from .data import write_json
from .runtime import append

ALLOWED = ('GPU-2931eba2-9a4d-61da-6bd1-e75692edad03', 'GPU-721a5bd6-d1e2-d5c5-af9a-6db4bdf691f7')
CAPS = {'total': 48 * 3600, 'pilot': 3 * 3600, 'sweep': 37.5 * 3600, 'evaluation': 7.5 * 3600}


@contextlib.contextmanager
def lock(path, nonblock=False):
    with Path(path).open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblock else 0))
        yield


def budget_used(records, stage=None):
    return sum(r.get('elapsed_seconds', r['reserved_seconds']) for r in records.values()
               if stage is None or r['stage'] == stage)


def reserve(root, name, stage, seconds, gpu):
    with lock(root / 'budget.lock'):
        p = root / 'budget.json'
        records = json.loads(p.read_text()) if p.exists() else {}
        if name in records:
            raise RuntimeError('duplicate run name; explicit new attempt required')
        if budget_used(records) + seconds > CAPS['total'] or budget_used(records, stage) + seconds > CAPS[stage]:
            raise RuntimeError('aggregate GPU-hour reservation exceeds remaining hard budget')
        records[name] = {'stage': stage, 'gpu': gpu, 'reserved_seconds': seconds,
                         'started_unix': time.time(), 'status': 'reserved'}
        write_json(p, records)


def finish(root, name, elapsed, code, reason):
    with lock(root / 'budget.lock'):
        p = root / 'budget.json'
        records = json.loads(p.read_text())
        records[name].update(elapsed_seconds=elapsed, exit_code=code, reason=reason,
                             finished_unix=time.time(), status='finished')
        write_json(p, records)


def gpu_state(uuid):
    text = subprocess.check_output(['nvidia-smi', f'--id={uuid}',
        '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu,driver_version',
        '--format=csv,noheader,nounits'], text=True).strip()
    parts = [s.strip() for s in text.split(',')]
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                         '--format=csv,noheader'], text=True)
    return {'raw': text, 'memory_used_mib': int(parts[2]), 'memory_total_mib': int(parts[3]),
            'utilization_pct': int(parts[4]), 'processes': [s for s in processes.splitlines() if uuid in s]}


def run(args):
    if args.gpu not in ALLOWED or args.seconds <= 30 or args.seconds > CAPS[args.stage]:
        raise ValueError('invalid authorized device or duration')
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    with lock(root / f'{args.gpu}.lock', nonblock=True):
        state = gpu_state(args.gpu)
        if state['processes'] or state['memory_used_mib'] >= 500 or state['utilization_pct'] >= 5:
            raise RuntimeError(f'GPU is not idle: {state}')
        job = root / 'jobs' / args.name
        job.mkdir(parents=True, exist_ok=False)
        reserve(root, args.name, args.stage, args.seconds, args.gpu)
        started = time.monotonic()
        child = None
        received = []
        def stop(signum, _):
            received.append(signum)
        old_handlers = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
        code, reason = 1, 'supervisor_exception'
        try:
            command = args.child[1:] if args.child[0] == '--' else args.child
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu, 'PYTHONUNBUFFERED': '1',
                   'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
                   'TRANSFORMERS_OFFLINE': '1', 'TOKENIZERS_PARALLELISM': 'false',
                   'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
            write_json(job / 'job.json', {'command': command, 'cwd': os.getcwd(), 'gpu': args.gpu,
                       'before': state, 'seconds': args.seconds, 'supervisor_pid': os.getpid(),
                       'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}, immutable=True)
            with (job / 'stdout.log').open('w') as stdout, (job / 'stderr.log').open('w') as stderr:
                child = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
                write_json(job / 'process.json', {'pid': child.pid, 'started_unix': time.time()})
                while child.poll() is None:
                    elapsed = time.monotonic() - started
                    append(job / 'gpu.jsonl', {'elapsed_seconds': elapsed, **gpu_state(args.gpu)})
                    if elapsed >= args.seconds - 30 or received:
                        reason = 'hard_time_cap' if not received else 'external_signal'
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=25)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        break
                    try:
                        child.wait(timeout=min(5, max(1, args.seconds - 30 - elapsed)))
                    except subprocess.TimeoutExpired:
                        pass
                code = child.returncode
                if reason == 'supervisor_exception':
                    reason = 'completed' if code == 0 else 'child_failure'
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            elapsed = time.monotonic() - started
            finish(root, args.name, elapsed, code, reason)
            write_json(job / 'status.json', {'exit_code': code, 'reason': reason, 'elapsed_seconds': elapsed})
            for s, handler in old_handlers.items():
                signal.signal(s, handler)
        return code


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--gpu', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--stage', choices=('pilot', 'sweep', 'evaluation'), required=True)
    p.add_argument('--seconds', type=int, required=True)
    p.add_argument('child', nargs=argparse.REMAINDER)
    args = p.parse_args()
    if not args.child:
        p.error('child command required')
    sys.exit(run(args))


if __name__ == '__main__':
    main()
