"""Bounded FIT-only warm-up calibration, then true RL gates and the full pipeline.

The first passing common format exposure among 1, 1.5 and 2 FIT passes is proposed.
Both actual reward panels must then pass; no failed panel triggers a blind retry.
"""
import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from .data import write_json
from .execute import ALLOWED
from .protocol import freeze
from .pipeline import run as pipeline


def run(root, original_data, frames_root, resume=False, control_name='calibration', attempt_offset=0, grounded_smoke=None, label_smoke=None):
    root = Path(root).resolve()
    original_data, frames_root = str(original_data), str(frames_root)
    if Path(control_name).name != control_name or control_name in ('.', '..') or attempt_offset < 0:
        raise ValueError('simple control directory and nonnegative attempt offset required')
    control = root / control_name
    control.mkdir(parents=True, exist_ok=resume)
    grounded_smoke = Path(grounded_smoke) if grounded_smoke else root / 'smoke-grounded-003/result.json'
    label_smoke = Path(label_smoke) if label_smoke else root / 'smoke-labels-003/result.json'
    write_json(control / 'config.json', {'control_name': control_name, 'attempt_offset': attempt_offset,
               'grounded_smoke': str(grounded_smoke), 'label_smoke': str(label_smoke),
               'target_mask': 'known-schema-unknown-payload-v2'}, immutable=True)
    if resume and (control / 'status.json').exists() and json.loads((control / 'status.json').read_text()).get('stage') == 'complete':
        return
    model = json.loads((root / 'data/preflight.json').read_text())['model']
    fit_count = len(json.loads((root / 'data/fit.json').read_text()))
    stop_count = len(json.loads((root / 'data/stop.json').read_text()))
    pilot = json.loads((root / 'pilot-sl-001/result.json').read_text())
    if pilot.get('status') != 'stopped':
        raise RuntimeError('supervised pilot must stop before RL calibration')
    started = time.time()
    def status(stage, **extra):
        write_json(control / 'status.json', {'stage': stage, 'started_unix': started,
                   'updated_unix': time.time(), 'controller_pid': os.getpid(), **extra})
    def completed(name, wait=False):
        job = root / 'jobs' / name
        while wait and not (job / 'status.json').exists():
            process = job / 'process.json'
            if process.exists():
                pid = json.loads(process.read_text())['pid']
                if not Path(f'/proc/{pid}').exists():
                    # Allow the supervisor to settle its final ledger record.
                    time.sleep(2)
                    if not (job / 'status.json').exists():
                        raise RuntimeError(f'process exited without settled job status: {name}')
            time.sleep(5)
        result = json.loads((root / name / 'result.json').read_text())
        if json.loads((job / 'status.json').read_text())['exit_code'] != 0:
            raise RuntimeError(f'job failed: {name}')
        return result
    def launch(name, gpu, seconds, child):
        if resume and ((root / name).exists() or (root / 'jobs' / name).exists()):
            result = completed(name, wait=True)
            spec = json.loads((root / name / 'spec.json').read_text())
            for flag in ('--model', '--data', '--adapter'):
                if flag in child and spec[flag[2:].replace('-', '_')] != child[child.index(flag) + 1]:
                    raise RuntimeError(f'resumed job input mismatch: {name}:{flag}')
            if spec['grounded'] != ('--grounded' in child):
                raise RuntimeError('resumed job supervision mismatch')
            return result
        command = [sys.executable, '-m', 'studies.firearm_posttraining.execute', '--root', str(root),
                   '--gpu', gpu, '--name', name, '--stage', 'pilot', '--seconds', str(seconds), '--',
                   sys.executable, *child]
        log = control / f'{name}.supervisor.log'
        if resume and log.exists():
            # No job/output exists: this was a prelaunch failure, never GPU work.
            # Retain its evidence and fail closed if a ledger reservation exists.
            ledger = root / 'budget.json'
            if ledger.exists() and name in json.loads(ledger.read_text()):
                raise RuntimeError(f'unsettled prior reservation requires operator review: {name}')
            log.rename(control / f'{name}.prelaunch-{time.time_ns()}.log')
        with log.open('x') as f:
            subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, check=True)
        return completed(name)
    def pair(tasks):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = {key: pool.submit(function, *arguments) for key, function, arguments in tasks}
            return {key: future.result() for key, future in futures.items()}
    try:
        chosen = None
        for initial_attempt, passes in ((2, 1.), (3, 1.5), (4, 2.)):
            attempt = initial_attempt + attempt_offset
            status('fixed_warmups', attempt=attempt, warmup_passes=passes)
            warm = {}
            tasks = []
            for index, kind, method in ((0, 'l', 'SL'), (1, 'g', 'SG')):
                name = f'pilot-c0{kind}-{attempt:03d}'
                warm[kind] = name
                if (root / name).exists() or (root / 'jobs' / name).exists():
                    # Reuse only explicitly named, already launched fixed warm-ups.
                    tasks.append((kind, completed, (name, True)))
                else:
                    child = ['-m', 'studies.firearm_posttraining', 'train', '--model', model,
                             '--data', str(root / 'data'), '--output', str(root / name), '--method', method,
                             '--warmup-only', '--warmup-passes', str(passes), '--max-passes', '5', '--max-seconds', '1765']
                    if kind == 'g':
                        child.append('--grounded')
                    tasks.append((kind, launch, (name, ALLOWED[index], 1800, child)))
            warmed = pair(tasks)
            adapters = {}
            for kind, result in warmed.items():
                if result['status'] != 'stopped' or result['stop_reason'] != 'fixed_warmup_complete' or result['fit_passes'] != passes:
                    raise RuntimeError('incomplete fixed warm-up; no extension/retry is automatic')
                adapters[kind] = result['best_checkpoint']
                state = json.loads((Path(adapters[kind]) / 'state.json').read_text())
                spec = json.loads((root / warm[kind] / 'spec.json').read_text())
                if (state['cursor'] != math.floor(fit_count * passes) or state['selection']['checks'] != 0 or
                    spec['model'] != model or spec['warmup_passes'] != passes or spec['grounded'] != (kind == 'g')):
                    raise RuntimeError('fixed warm-up identity/supervision mismatch')
            status('FIT_format_sampling', attempt=attempt, warmup_passes=passes)
            tasks = []
            for index, kind in enumerate(('l', 'g')):
                name = f'format-{kind}-{attempt:03d}'
                child = ['-m', 'studies.firearm_posttraining.fit_format', '--model', model,
                         '--data', str(root / 'data'), '--output', str(root / name), '--adapter', adapters[kind]]
                if kind == 'g':
                    child.append('--grounded')
                tasks.append((kind, launch, (name, ALLOWED[index], 600 if kind == 'l' else 900, child)))
            formats = pair(tasks)
            write_json(control / f'format-decision-{attempt:03d}.json', {'warmup_passes': passes, 'results': formats}, immutable=True)
            if all(r['status'] == 'passed' for r in formats.values()):
                chosen = (attempt, passes, adapters)
                break
        if chosen is None:
            status('failed_format_calibration', reason='none of the declared FIT-only warm-up candidates met the format gate')
            return
        attempt, passes, adapters = chosen
        status('real_reward_panels', attempt=attempt, warmup_passes=passes)
        tasks, panels = [], {}
        for index, kind in enumerate(('l', 'g')):
            name = f'pilot-{"rl" if kind == "l" else "rg"}-{attempt:03d}'
            panels[kind] = root / name / 'result.json'
            seconds = 1800 if kind == 'l' else 3600
            child = ['-m', 'studies.firearm_posttraining', 'rl-panel', '--model', model,
                     '--data', str(root / 'data'), '--output', str(root / name), '--adapter', adapters[kind],
                     '--max-seconds', str(seconds - 35)]
            if kind == 'g':
                child.append('--grounded')
            tasks.append((kind, launch, (name, ALLOWED[index], seconds, child)))
        rewards = pair(tasks)
        if not all(r['status'] == 'passed' for r in rewards.values()):
            status('failed_real_reward_gate', warmup_passes=passes, results=rewards)
            return
        label_latency = json.loads((root / 'pilot-sl-001/stop-initial/metrics.json').read_text())['latency_seconds_per_video']
        ground_latency = formats['g']['elapsed_seconds'] / formats['g']['responses']
        checks = 1 + min(10, int((5 - passes) * 2))
        remaining = (5 - passes) * fit_count
        estimates = {}
        for method, kind, objective in (('SL', 'l', 'sft'), ('RL', 'l', 'rl'), ('SG', 'g', 'sft'), ('RG', 'g', 'rl')):
            warm_seconds = warmed[kind]['elapsed_seconds']
            seconds_per_video = (warm_seconds / (passes * fit_count) if objective == 'sft' else
                                 rewards[kind]['elapsed_seconds'] / rewards[kind]['groups'])
            latency = label_latency if kind == 'l' else ground_latency
            estimates[method] = 1.2 * (warm_seconds + remaining * seconds_per_video + checks * stop_count * latency)
        estimates['VQ'] = 1.2 * (120 * label_latency + 11 * stop_count * label_latency + 10 * 10)
        physical_per_seed = sum(estimates.values()) - sum(warmed[k]['elapsed_seconds'] for k in ('l', 'g'))
        source_counts = json.loads((root / 'data/audit.json').read_text())['source_counts']
        eval_videos = source_counts['val'] + source_counts['test_metadata']
        forecast = {'status': 'measured_forecast', 'warmup_passes': passes, 'shared_max_fit_passes': 5.,
                    'assumption': 'maximum exposure/query schedule with 20% overhead; scientific stops can shorten runs; not a runtime promise',
                    'estimated_seconds_per_method_seed': estimates,
                    'estimated_training_physical_gpu_hours': 3 * physical_per_seed / 3600,
                    'estimated_training_wall_hours_two_workers': 3 * max(physical_per_seed / 2, max(estimates.values())) / 3600,
                    'estimated_final_evaluation_gpu_hours': (6 * eval_videos * ground_latency + 10 * eval_videos * label_latency) * 1.2 / 3600,
                    'evaluation_videos_per_model': eval_videos, 'VQ_optimizer_seconds_assumption': 10.,
                    'measurement_sources': {'warmups': warm, 'panels': {k: str(v) for k, v in panels.items()}},
                    'time_ceilings': 'scientific-run ceilings removed by user; no exposure reduction to fit old limits'}
        write_json(root / 'throughput-forecast.json', forecast, immutable=True)
        status('freezing_protocol', warmup_passes=passes)
        protocol = freeze(root, root / 'protocol.json', label_smoke,
                          grounded_smoke, root / 'pilot-sl-001/result.json',
                          panels['l'], panels['g'], max_passes=5., warmup_passes=passes)
        status('scientific_pipeline_running', warmup_passes=passes, protocol_hash=protocol['protocol_hash'])
        pipeline(root, root / 'protocol.json', original_data, frames_root)
        final = json.loads((root / 'pipeline/status.json').read_text())
        status(final['stage'], pipeline=final)
    except Exception as exc:
        status('failed', error_type=type(exc).__name__, error=str(exc))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--original-data', required=True)
    p.add_argument('--frames-root', required=True)
    p.add_argument('--resume', action='store_true', help='reuse matching calibration jobs without repeating them')
    p.add_argument('--control-name', default='calibration')
    p.add_argument('--attempt-offset', type=int, default=0)
    p.add_argument('--grounded-smoke', help='passing engineering evidence for the current grounded target mask')
    p.add_argument('--label-smoke', help='passing engineering evidence for the current label target mask')
    args = p.parse_args()
    run(args.root, args.original_data, args.frames_root, args.resume, args.control_name, args.attempt_offset, args.grounded_smoke, args.label_smoke)
