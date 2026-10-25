"""The firearm study's A/G/AG run: one study JSON, an immutable spec, a resumable unit ledger.

Each window is decoded and resized exactly as run_vera.py does. Its eight images are then
sent, in the frozen order A, G, AG, through the same backend.predict call: A with VERA's
prompt, G and AG with theirs. Each call is a fresh batch_chat without history. A unit
(video, anchor, condition) is skipped only when a complete record matches the current config,
model, prompt and input (8 RGB sha256) hashes. Exceptions and timeouts are retried up to
execution.max_attempts. An answer that cannot be parsed is complete, not retried: greedy
decoding would repeat it, and its invalidity is a result.
"""
import copy
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback
from .config import ROOT, from_dict, merge, path, prepare_environment
from .conditions import CONDITIONS, prompt_texts
from .data import preprocess, sha256, windows
from .ledger import JsonlAppender, Ledger, now, read_jsonl, unit_key

STUDY_DEFAULTS = {
    'study_schema': 1, 'study_id': None, 'runner': None,
    'conditions': {'order': ['A', 'G', 'AG'], 'prompt_dir': 'studies/firearm/prompts', 'prompt_sha256': None},
    'data': {'inventory': None, 'annotation_policy': 'studies/firearm/annotation_policy.json'},
    'execution': {'run_dir': None, 'per_call_timeout_seconds': 600, 'max_attempts': 3, 'max_consecutive_failures': 3},
    'features': {'dir': None, 'clip_seconds': 10.0, 'fps': 30.0, 'num_samples': 10, 'weights': None, 'batch_size': 16},
    'matching': {'iou_thresholds': [0.5, 0.75]},
    'analysis': {'bootstrap_resamples': 2000, 'bootstrap_seed': 20261025, 'ci_level': 0.95},
}
# Parts that change what the model is asked or shown; a resumed run must match them exactly.
SPEC_KEYS = ('study_schema', 'study_id', 'runner', 'conditions')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256((value if isinstance(value, str) else canonical(value)).encode('utf-8')).hexdigest()


def load_study(source):
    raw = copy.deepcopy(source) if isinstance(source, dict) else json.loads(Path(source).read_text())
    runner = raw.pop('runner', None)
    if not isinstance(runner, dict):
        raise ValueError('The study needs a runner block (a configurable-runner experiment)')
    study = merge({k: v for k, v in STUDY_DEFAULTS.items() if k != 'runner'}, raw)
    study['runner'] = from_dict(runner)
    if study['study_schema'] != 1 or not study['study_id']:
        raise ValueError('study_schema must be 1 and study_id set')
    if sorted(study['conditions']['order']) != sorted(CONDITIONS):
        raise ValueError('conditions.order must list A, G and AG once each')
    r = study['runner']
    if r['stages'] != ['preprocess', 'infer'] or r['intervention']['mode'] != 'baseline' or r['boxes']['mode'] != 'off':
        raise ValueError('The study runs plain inference; refinement is a separate step and no runner intervention/boxes apply')
    ex = study['execution']
    for key in ('per_call_timeout_seconds', 'max_attempts', 'max_consecutive_failures'):
        if not isinstance(ex[key], (int, float)) or ex[key] <= 0:
            raise ValueError(f'execution.{key} must be positive')
    return study


def select(study, inventory):
    """Ordered (row, windows) for the run's selection; windows exactly as data.windows schedules them."""
    sel = study['runner']['selection']
    rows = sorted(inventory, key=lambda r: r['id'])
    if sel['video_ids']:
        known = {r['id']: r for r in rows}
        missing = [v for v in sel['video_ids'] if v not in known]
        if missing:
            raise ValueError(f'Selected videos not in the inventory: {missing}')
        rows = [known[v] for v in sel['video_ids']]
    if sel['max_videos']:
        rows = rows[:sel['max_videos']]
    out = []
    for row in rows:
        wins = list(windows(row['frames_decoded'], study['runner']['sampling']))
        out.append((row, wins[:sel['max_windows_per_video']] if sel['max_windows_per_video'] else wins))
    return out


def build_spec(study, inventory_path):
    inventory = json.loads(Path(inventory_path).read_text())
    prompts = prompt_texts(study['runner'], study['conditions']['prompt_dir'])
    locked = study['conditions']['prompt_sha256']
    if locked is not None and locked != {c: digest(t) for c, t in prompts.items()}:
        raise ValueError('Prompt texts differ from the locked conditions.prompt_sha256')
    chosen = select(study, inventory)
    spec = {k: study[k] for k in SPEC_KEYS}
    spec.update(prompts=prompts, prompt_sha256={c: digest(t) for c, t in prompts.items()},
                inventory_sha256=sha256(inventory_path),
                videos=[{'id': r['id'], 'category': r['category'], 'frames': r['frames_decoded'], 'video_sha256': r['sha256'],
                         'anchors': [w['anchor'] for w in wins]} for r, wins in chosen])
    spec['units'] = sum(len(v['anchors']) for v in spec['videos']) * len(CONDITIONS)
    return spec, chosen


def git(*args):
    try:
        return subprocess.check_output(['git', *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_description():
    uuid = os.environ.get('CUDA_VISIBLE_DEVICES')
    try:
        query = ['nvidia-smi', '--query-gpu=uuid,name,driver_version,memory.total', '--format=csv,noheader']
        if uuid:
            query.insert(1, f'--id={uuid}')
        return {'cuda_visible_devices': uuid, 'nvidia_smi': subprocess.check_output(query, text=True).strip()}
    except (OSError, subprocess.CalledProcessError):
        return {'cuda_visible_devices': uuid, 'nvidia_smi': None}


def model_hash(identity):
    keep = {k: identity.get(k) for k in ('repo_or_path', 'revision', 'device', 'dtype', 'flash_attention', 'files', 'synthetic')}
    return digest(keep)


def load_images(frames_dir, indices, size):
    from PIL import Image
    images = []
    for index in indices:
        with Image.open(frames_dir / f'{index:06d}.jpg') as im:
            # Exactly run_vera.py: the upstream grid_size=1 PIL resize, no aspect preservation or crop.
            images.append(im.convert('RGB').resize((size, size)))
    return images


def rgb_hash(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def ensure_frames(row, video_dir, ledger, session):
    """Decode once with data.preprocess; reuse only a complete decode of the same file."""
    info_path = video_dir / 'video.json'
    frames = video_dir / 'frames'
    if info_path.exists():
        info = json.loads(info_path.read_text())
        if (info['video_sha256'] == row['sha256'] and info['decoded_frames'] == row['frames_decoded']
                and len(list(frames.glob('*.jpg'))) == info['decoded_frames']):
            return info
        ledger.note(session, video=row['id'], note='decoded frames did not match; decoding again', old=info)
    if frames.exists():
        aside = video_dir / f'frames.incomplete.{session}'
        frames.rename(aside)
        ledger.note(session, video=row['id'], note=f'moved an incomplete decode to {aside.name}')
    video_dir.mkdir(parents=True, exist_ok=True)
    info = preprocess({'id': row['id'], 'path': row['staged_path'], 'annotation': {}}, frames)
    if info['video_sha256'] != row['sha256'] or info['decoded_frames'] != row['frames_decoded']:
        raise ValueError(f"{row['id']}: decoded video differs from the inventory")
    (info_path).write_text(json.dumps(info, indent=2) + '\n')
    return info


def save_inputs(directory, images, hashes):
    """PNG copies of the exact model inputs; an existing set is kept when its pixels match."""
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'hashes.json'
    if manifest.exists() and json.loads(manifest.read_text()) == hashes and all(
            rgb_hash(Image.open(directory / f'{i:02d}.png').convert('RGB')) == h for i, h in enumerate(hashes)):
        return False
    for i, im in enumerate(images):
        im.save(directory / f'{i:02d}.png')
    manifest.write_text(json.dumps(hashes) + '\n')
    return True


def complete_records(run_dir):
    """{unit: [complete records]} from every condition's records file."""
    done = {}
    for condition in CONDITIONS:
        rows, _ = read_jsonl(Path(run_dir) / 'records' / f'{condition}.jsonl')
        for r in rows:
            if r['status'] == 'complete':
                done.setdefault(unit_key(r['video'], r['anchor'], r['condition']), []).append(r)
    return done


def matches(record, hashes):
    return all(record['hashes'].get(k) == v for k, v in hashes.items())


def parse_record(condition, text, call, max_new_tokens):
    from .backends import strict_label
    from .slots import final_output_label, legacy_label, parse_slots
    truncated = (not call['eos']) and call['generated_tokens'] >= max_new_tokens
    out = {'truncated': truncated}
    if condition in ('A', 'AG'):
        final = final_output_label(text)
        out.update(label=None if truncated else final, final_output_label=final,
                   strict_label=strict_label(text), upstream_label=legacy_label(text))
    if condition in ('G', 'AG'):
        out['grounding'] = parse_slots(call['tagged_text'] if call['tagged_text'] else text, truncated=truncated)
    return out


class Timeout:
    """Per-call watchdog: a stuck generation ends the process with 124 after an explicit ledger entry."""
    def __init__(self, seconds, on_fire):
        self.seconds, self.on_fire, self.timer = seconds, on_fire, None

    def __enter__(self):
        self.timer = threading.Timer(self.seconds, self.on_fire)
        self.timer.daemon = True
        self.timer.start()

    def __exit__(self, *exc):
        self.timer.cancel()


class Abort(Exception):
    pass


def run_study(study, inventory_path=None, backend=None):
    """Run or resume. Returns 0 when every unit is complete or out of attempts, 1 otherwise."""
    run_dir = path(study['execution']['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir / 'run.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec, chosen = build_spec(study, inventory_path or path(study['data']['inventory']))
    spec_path = run_dir / 'spec.json'
    if spec_path.exists():
        old = json.loads(spec_path.read_text())
        if digest({k: v for k, v in old.items() if k != 'spec_sha256'}) != old.get('spec_sha256') or old['spec_sha256'] != digest(spec):
            raise ValueError(f'{spec_path} differs from this study; start a new run_dir instead of resuming')
    else:
        spec_path.write_text(json.dumps({**spec, 'spec_sha256': digest(spec)}, indent=1, ensure_ascii=False) + '\n')
        os.chmod(spec_path, 0o444)
    for c, text in spec['prompts'].items():
        prompt_file = run_dir / 'prompts' / f'{c}.txt'
        prompt_file.parent.mkdir(exist_ok=True)
        if not prompt_file.exists():
            prompt_file.write_text(text)
    sessions = run_dir / 'sessions'
    sessions.mkdir(exist_ok=True)
    number = 1 + max([int(p.name) for p in sessions.iterdir() if p.name.isdigit()], default=0)
    session = f'{number:03d}'
    sdir = sessions / session
    sdir.mkdir()
    state = {'session': session, 'pid': os.getpid(), 'started_at': now(), 'status': 'starting', 'host': platform.node(),
             'code_commit': (git('rev-parse', 'HEAD') or '').strip() or None, 'code_diff_sha256': digest(git('diff', 'HEAD') or ''),
             'gpu': gpu_description(), 'python': sys.executable, 'spec_sha256': digest(spec), 'calls_done': 0}
    (sdir / 'session.json').write_text(json.dumps(state, indent=2) + '\n')
    (sdir / 'git_diff.patch').write_text(git('diff', 'HEAD') or '')
    status_file = sdir / 'status.json'
    stop = threading.Event()

    def beat():
        while not stop.wait(5):
            tmp = status_file.with_suffix('.tmp')
            tmp.write_text(json.dumps({**state, 'heartbeat_at': now()}, indent=2) + '\n')
            tmp.replace(status_file)
    threading.Thread(target=beat, daemon=True).start()
    ledger = Ledger(run_dir)
    if ledger.torn:
        ledger.note(session, note='torn last ledger line skipped', text=ledger.torn)
    writers = {c: JsonlAppender(run_dir / 'records' / f'{c}.jsonl') for c in CONDITIONS}
    done = complete_records(run_dir)
    cfg = study['runner']
    order, max_attempts = study['conditions']['order'], study['execution']['max_attempts']
    try:
        if backend is None:
            prepare_environment(cfg)
            if cfg['model']['backend'] == 'internvl2':
                (sdir / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
            from .backends import BACKENDS
            state['status'] = 'loading_model'
            loading = time.monotonic()
            backend = BACKENDS[cfg['model']['backend']](cfg)
            state['model_load_seconds'] = time.monotonic() - loading
        identity = getattr(backend, 'model_identity', {'synthetic': True})
        state.update(status='running', model_identity=identity, model_sha256=model_hash(identity))
        (sdir / 'session.json').write_text(json.dumps(state, indent=2) + '\n')
        base_hashes = {'config': digest({k: spec[k] for k in SPEC_KEYS}), 'model': state['model_sha256']}
        failures_in_a_row = 0
        for row, wins in chosen:
            vid = row['id']
            vdir = run_dir / 'videos' / vid
            pending = lambda anchor, c: ledger.attempts.get(unit_key(vid, anchor, c), 0) < max_attempts
            if not any(pending(w['anchor'], c) for w in wins for c in order):
                continue
            state.update(video=vid, stage='preprocess')
            try:
                ensure_frames(row, vdir, ledger, session)
            except Exception:
                # Counted as an attempt of every open unit, so a video that cannot be decoded ends as exhausted.
                error = traceback.format_exc()
                for w in wins:
                    for c in order:
                        if pending(w['anchor'], c):
                            unit = unit_key(vid, w['anchor'], c)
                            ledger.end(unit, ledger.start(unit, session), session, 'preprocess_failed', error=error)
                continue
            schedule = vdir / 'windows.json'
            if schedule.exists():
                if json.loads(schedule.read_text()) != wins:
                    raise ValueError(f'{vid}: window schedule differs from the saved one')
            else:
                schedule.write_text(json.dumps(wins) + '\n')
            for w in wins:
                anchor = w['anchor']
                images = load_images(vdir / 'frames', w['frame_indices'], cfg['preprocessing']['image_size'])
                rgb = [rgb_hash(im) for im in images]
                save_inputs(vdir / 'inputs' / str(anchor), images, rgb)
                for c in order:
                    unit = unit_key(vid, anchor, c)
                    hashes = {**base_hashes, 'prompt': spec['prompt_sha256'][c], 'input': digest(''.join(rgb))}
                    if any(matches(r, hashes) for r in done.get(unit, [])):
                        continue
                    if ledger.attempts.get(unit, 0) >= max_attempts:
                        continue
                    attempt = ledger.start(unit, session)
                    state.update(stage='inference', anchor=anchor, condition=c, attempt=attempt)
                    started = now()

                    def fire(unit=unit, attempt=attempt):
                        ledger.end(unit, attempt, session, 'timeout', seconds=study['execution']['per_call_timeout_seconds'])
                        state['status'] = 'timed_out'
                        (sdir / 'session.json').write_text(json.dumps(state, indent=2) + '\n')
                        os._exit(124)
                    base = {'video': vid, 'anchor': anchor, 'start': w['start'], 'end': w['end'], 'frame_indices': w['frame_indices'],
                            'condition': c, 'slots': [], 'attempt': attempt, 'session': session, 'hashes': hashes,
                            'rgb_sha256': rgb, 'started_at': started, 'synthetic': cfg['model']['backend'] == 'mock'}
                    try:
                        with Timeout(study['execution']['per_call_timeout_seconds'], fire):
                            prediction = backend.predict(images, question=spec['prompts'][c])
                    except KeyboardInterrupt:
                        ledger.end(unit, attempt, session, 'interrupted')
                        raise
                    except Exception:
                        failures_in_a_row += 1
                        writers[c].write({**base, 'status': 'failed', 'error': traceback.format_exc(), 'finished_at': now()})
                        ledger.end(unit, attempt, session, 'failed')
                        if failures_in_a_row >= study['execution']['max_consecutive_failures']:
                            raise Abort(f'{failures_in_a_row} consecutive failed calls; ending this session')
                        continue
                    failures_in_a_row = 0
                    call = backend.last_call
                    ids = prediction['input_token_ids']
                    ids_sha = digest(canonical(ids)) if ids is not None else None
                    if ids is not None and not (run_dir / 'input_ids' / f'{ids_sha}.json').exists():
                        (run_dir / 'input_ids').mkdir(exist_ok=True)
                        (run_dir / 'input_ids' / f'{ids_sha}.json').write_text(json.dumps(ids) + '\n')
                    record = {**base, 'status': 'complete', 'finished_at': now(), 'raw_response': prediction['raw_response'],
                              'tagged_text': call['tagged_text'], 'output_ids': call['output_ids'],
                              'generated_tokens': call['generated_tokens'], 'eos': call['eos'], 'seconds': call['seconds'],
                              'peak_gpu_bytes': call['peak_gpu_bytes'], 'input_ids_sha256': ids_sha,
                              'input_ids_length': len(ids[0]) if ids else None,
                              **parse_record(c, prediction['raw_response'], call, cfg['generation']['max_new_tokens'])}
                    writers[c].write(record)
                    ledger.end(unit, attempt, session, 'complete', record_sha256=digest(record))
                    done.setdefault(unit, []).append(record)
                    state['calls_done'] += 1
                    rendered = run_dir / f'model_input_{c}.txt'
                    if not rendered.exists():
                        rendered.write_text(backend.rendered_input())
        dispositions = reconcile(study, run_dir, spec=spec, model_sha256=state['model_sha256'])
        open_units = dispositions['counts'].get('pending', 0)
        state['status'] = 'complete' if open_units == 0 else 'incomplete'
        return 0 if open_units == 0 else 1
    except Abort as exc:
        state.update(status='aborted', error=str(exc))
        return 1
    except KeyboardInterrupt:
        state['status'] = 'interrupted'
        return 130
    except BaseException:
        state.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        stop.set()
        state['finished_at'] = now()
        (sdir / 'session.json').write_text(json.dumps(state, indent=2) + '\n')
        for w in writers.values():
            w.close()
        ledger.close()
        lock.close()


def final_records(study, run_dir):
    """{condition: {(video, anchor): record}}: the complete record of each unit that reconcile counts as
    complete (the latest, if a unit was ever completed twice). Analyses read nothing else."""
    run_dir = Path(run_dir)
    result = json.loads((run_dir / 'reconcile.json').read_text()) if (run_dir / 'reconcile.json').exists() else None
    if result is None:
        reconcile(study, run_dir)
        result = json.loads((run_dir / 'reconcile.json').read_text())
    spec = {k: v for k, v in json.loads((run_dir / 'spec.json').read_text()).items() if k != 'spec_sha256'}
    config_hash = digest({k: spec[k] for k in SPEC_KEYS})
    complete = {u['unit'] for u in result['units'] if u['disposition'] == 'complete'}
    out = {c: {} for c in study['conditions']['order']}
    inputs = {}
    for unit, rows in complete_records(run_dir).items():
        if unit not in complete:
            continue
        video, anchor, _ = unit.split('|')
        if (video, anchor) not in inputs:
            saved = run_dir / 'videos' / video / 'inputs' / anchor / 'hashes.json'
            inputs[(video, anchor)] = digest(''.join(json.loads(saved.read_text())))
        good = [r for r in rows if r['hashes']['config'] == config_hash and r['hashes']['model'] == result['model_sha256']
                and r['hashes']['prompt'] == spec['prompt_sha256'][r['condition']] and r['hashes']['input'] == inputs[(video, anchor)]]
        if good:
            r = good[-1]
            out[r['condition']][(r['video'], r['anchor'])] = r
    return out, spec


def reconcile(study, run_dir, spec=None, model_sha256=None):
    """Every scheduled unit with one disposition: complete, exhausted (out of attempts) or pending.

    Without model_sha256, the model hash of the newest session that loaded a model is used, and all
    complete records must share it."""
    run_dir = Path(run_dir)
    spec = spec or {k: v for k, v in json.loads((run_dir / 'spec.json').read_text()).items() if k != 'spec_sha256'}
    if model_sha256 is None:
        seen = sorted((json.loads(p.read_text()) for p in (run_dir / 'sessions').glob('*/session.json')), key=lambda s: s['session'])
        model_sha256 = next((s['model_sha256'] for s in reversed(seen) if s.get('model_sha256')), None)
    done = complete_records(run_dir)
    ledger_events, _ = read_jsonl(run_dir / 'ledger.jsonl')
    attempts = {}
    for e in ledger_events:
        if e['event'] == 'start':
            attempts[e['unit']] = max(attempts.get(e['unit'], 0), e['attempt'])
    config_hash = digest({k: spec[k] for k in SPEC_KEYS})
    max_attempts = study['execution']['max_attempts']
    rows, counts = [], {}
    for v in spec['videos']:
        for anchor in v['anchors']:
            saved = run_dir / 'videos' / v['id'] / 'inputs' / str(anchor) / 'hashes.json'
            input_hash = digest(''.join(json.loads(saved.read_text()))) if saved.exists() else None
            for c in study['conditions']['order']:
                unit = unit_key(v['id'], anchor, c)
                hashes = {'config': config_hash, 'model': model_sha256, 'prompt': spec['prompt_sha256'][c], 'input': input_hash}
                good = [r for r in done.get(unit, []) if input_hash and matches(r, hashes)]
                n = attempts.get(unit, 0)
                disposition = 'complete' if good else ('exhausted' if n >= max_attempts else 'pending')
                counts[disposition] = counts.get(disposition, 0) + 1
                rows.append({'unit': unit, 'video': v['id'], 'category': v['category'], 'anchor': anchor, 'condition': c,
                             'disposition': disposition, 'attempts': n, 'complete_records': len(good)})
    result = {'expected_units': spec['units'], 'listed_units': len(rows), 'counts': counts, 'model_sha256': model_sha256}
    (run_dir / 'reconcile.json').write_text(json.dumps({**result, 'units': rows}, indent=1) + '\n')
    return result
