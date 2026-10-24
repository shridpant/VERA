#!/usr/bin/env python3
"""Local-only VERA Lab: run one window or a full video and inspect every saved stage."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import atexit
import fcntl
import importlib.util
import json
import math
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import usage
from tools.stage_view import stage_manifest
from vera_runner.config import path as runner_path
WEB = ROOT / 'tools' / 'web'
BASE_CONFIG = ROOT / 'configs' / 'baseline.json'
RUNS = ROOT / 'demo_runs'
CURRENT = RUNS / 'current'
JOBS = {}
LOCK = threading.Lock()
SESSION_TOKEN = secrets.token_urlsafe(32)
LIBRARY = None
MODEL_PATH = 'OpenGVLab/InternVL2-8B'
FEATURE_DIR = None
ANNOTATION_FILE = None
VIDEO_SUFFIXES = {'.mp4', '.avi', '.mov', '.mkv'}
PIPELINE_OUTPUT = Path('pipeline') / 'run 1' / 'output'
WINDOW_TIMEOUT = 3600
VIDEO_TIMEOUT = 4 * 3600
WORKER_IDLE_SECONDS = 900
STAGE_TARGETS = {
    'window': ['preprocess', 'infer'],
    'preprocess': ['preprocess'],
    'segments': ['preprocess', 'infer'],
    'refine': ['preprocess', 'infer', 'refine'],
    'evaluate': ['preprocess', 'infer', 'refine', 'evaluate'],
}
STATIC = {
    '/': ('index.html', 'text/html; charset=utf-8'),
    '/how-it-works': ('how.html', 'text/html; charset=utf-8'),
    '/frame-game': ('game.html', 'text/html; charset=utf-8'),
    '/scores': ('scores.html', 'text/html; charset=utf-8'),
    '/static/styles.css': ('styles.css', 'text/css; charset=utf-8'),
    '/static/common.js': ('common.js', 'text/javascript; charset=utf-8'),
    '/static/model.js': ('model.js', 'text/javascript; charset=utf-8'),
    '/static/game.js': ('game.js', 'text/javascript; charset=utf-8'),
    '/static/scores.js': ('scores.js', 'text/javascript; charset=utf-8'),
}


def choose_random_video(library, require_feature=False):
    if library is None or not library.is_dir():
        raise ValueError('No sample library configured. Start the server with --video-dir.')
    candidates = sorted(p for p in library.rglob('*') if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)
    if require_feature:
        candidates = [p for p in candidates if FEATURE_DIR is not None and
                      (FEATURE_DIR / (p.stem + '.npy')).is_file()]
    if not candidates:
        raise ValueError('The configured sample library has no video with the files required for this stage.')
    selected = candidates[secrets.randbelow(len(candidates))]
    anchor = 5.0
    try:
        import cv2
        cap = cv2.VideoCapture(str(selected))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        finally:
            cap.release()
        if fps > 0 and count > 0:
            duration = count / fps
            anchor = round(secrets.randbelow(max(1, int(duration * 10))) / 10, 1)
    except ImportError:
        pass
    feature = FEATURE_DIR / (selected.stem + '.npy') if FEATURE_DIR else None
    return {'path': str(selected), 'name': selected.name, 'anchor_seconds': anchor,
            'feature_file': str(feature) if feature and feature.is_file() else '',
            'annotation_file': str(ANNOTATION_FILE) if ANNOTATION_FILE else ''}


def missing_dependencies(target):
    modules = ['cv2', 'PIL', 'numpy']
    if target != 'preprocess':
        modules += ['torch', 'torchvision', 'transformers', 'accelerate',
                    'einops', 'timm', 'sentencepiece', 'safetensors']
    return [module for module in modules if importlib.util.find_spec(module) is None]


def output_path(job, relative):
    """Constrain browsing to the active scratch run."""
    root = job['folder'].resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root) or not target.exists():
        raise ValueError('Artifact is outside the active run or does not exist')
    return target


def output_file(job, relative):
    """Constrain downloads and previews to supported files in the active run."""
    target = output_path(job, relative)
    if not target.is_file():
        raise ValueError('Artifact is not a file')
    if target.suffix.lower() not in {'.json', '.jsonl', '.txt', '.log', '.npy', '.png', '.jpg'}:
        raise ValueError('Unsupported artifact type')
    return target


def stop_stalled_job(proc, folder, seconds):
    """Release the model slot if a probe or runner exceeds its time budget."""
    try:
        proc.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        with (folder / 'console.log').open('a') as log_file:
            log_file.write(f'\nVERA Lab stopped the run after {seconds} seconds.\n')
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


class FinishedRun:
    def __init__(self, returncode):
        self.returncode = returncode

    def poll(self):
        return self.returncode


def restore_current_job():
    """Make the last scratch run inspectable after the UI server restarts."""
    record = CURRENT / 'lab_job.json'
    if not record.is_file():
        return
    saved = json.loads(record.read_text())
    if saved['mode'] == 'window':
        ok = (CURRENT / 'probe' / 'result.json').is_file()
    else:
        status = CURRENT / PIPELINE_OUTPUT / 'status.json'
        ok = status.is_file() and json.loads(status.read_text()).get('status') == 'complete'
    JOBS[saved['job']] = {'process': FinishedRun(0 if ok else 1),
                          'folder': CURRENT, 'mode': saved['mode'],
                          'video_id': saved['video_id'],
                          'target_stage': saved['target_stage']}


class WorkerJob:
    """One window request answered by the warm model process; polls like Popen."""
    def __init__(self):
        self.returncode = None
        self.done = threading.Event()

    def poll(self):
        return self.returncode

    def finish(self, code):
        self.returncode = code
        self.done.set()


class WarmModel:
    """Keep one window-probe process, and its loaded weights, between window runs."""
    # Retire the worker this long before its own idle exit, so a request never races that exit.
    IDLE_MARGIN = 60

    def __init__(self):
        self.process = None
        self.settings = None
        self.loaded = None  # Device and precision the worker actually resolved.
        self.last_reply = 0.0

    def usable(self, process):
        return (process is not None and process.poll() is None and
                time.monotonic() - self.last_reply < WORKER_IDLE_SECONDS - self.IDLE_MARGIN)

    def describe(self):
        process, settings, loaded = self.process, self.settings, self.loaded
        alive = loaded is not None and settings is not None and self.usable(process)
        return {'loaded': alive, 'model_path': settings['path'] if alive else None,
                **(loaded if alive else {'device': None, 'dtype': None}),
                'idle_seconds': WORKER_IDLE_SECONDS}

    def stop(self, force=False):
        process, self.process, self.settings, self.loaded = self.process, None, None, None
        if process is None:
            return
        if process.poll() is None:
            if force:
                process.kill()
            else:
                try:
                    process.stdin.close()  # The worker exits at end of input.
                except OSError:
                    pass
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for pipe in (process.stdin, process.stdout):
            try:
                pipe.close()
            except OSError:
                pass

    def start(self, settings):
        cmd = [sys.executable, str(ROOT / 'tools' / 'window_probe.py'), '--serve',
               '--idle-seconds', str(WORKER_IDLE_SECONDS)]
        with (RUNS / 'model_worker.log').open('w') as log_file:
            # Fixed argument list without a shell; request values travel over stdin as JSON.
            self.process = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.PIPE,  # nosemgrep
                                            stdout=subprocess.PIPE, stderr=log_file, text=True)
        self.settings = settings
        self.last_reply = time.monotonic()

    def send(self, request):
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()

    def submit(self, settings, request, console):
        """Send one window request; start or replace the worker when model settings change."""
        reused = self.usable(self.process) and settings == self.settings
        if not reused:
            self.stop()
            self.start(settings)
        try:
            self.send(request)
        except OSError:  # The worker exited after the check; start a fresh one once.
            self.stop()
            self.start(settings)
            self.send(request)
            reused = False
        console.write_text('Reusing the loaded model process.\n' if reused else
                           'Started a model process. The first window loads the checkpoint.\n')
        job = WorkerJob()
        threading.Thread(target=self.wait, args=(self.process, job, console), daemon=True).start()
        threading.Thread(target=self.watch, args=(self.process, job, console), daemon=True).start()
        return job

    def wait(self, process, job, console):
        ok = False
        try:
            try:
                line = process.stdout.readline()
            except (OSError, ValueError):  # stop() closed the pipe of a killed process.
                line = ''
            reply = json.loads(line) if line else {
                'ok': False, 'error': 'The model process exited before replying. See demo_runs/model_worker.log.'}
            ok = reply['ok']
            if line and process is self.process:
                self.last_reply = time.monotonic()
                if ok:
                    self.loaded = {'device': reply['device'], 'dtype': reply['dtype']}
            with console.open('a') as log_file:
                log_file.write(json.dumps({key: value for key, value in reply.items() if key != 'error'}) + '\n')
                if not ok:
                    log_file.write(reply['error'] + '\n')
        finally:
            job.finish(0 if ok else 1)

    def watch(self, process, job, console):
        if job.done.wait(WINDOW_TIMEOUT):
            return
        with console.open('a') as log_file:
            log_file.write(f'\nVERA Lab stopped the model process after {WINDOW_TIMEOUT} seconds.\n')
        if process is self.process:
            self.stop(force=True)
        else:
            process.kill()
        if not job.done.wait(30):
            job.finish(1)


WARM = WarmModel()


class Handler(BaseHTTPRequestHandler):
    def local_host(self):
        return urlsplit('http://' + self.headers.get('Host', '')).hostname in ('127.0.0.1', 'localhost')

    def send(self, code, payload, content_type='application/json'):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.local_host():
            return self.send(403, {'error': 'VERA Lab accepts local hostnames only'})
        route = urlsplit(self.path)
        if route.path in STATIC:
            name, mime = STATIC[route.path]
            return self.send(200, (WEB / name).read_bytes(), mime)
        if route.path == '/api/defaults':
            return self.send(200, {'model_path': MODEL_PATH,
                                   'session_token': SESSION_TOKEN,
                                   'current_job': next(iter(JOBS), None),
                                   'missing_dependencies': missing_dependencies('window'),
                                   'feature_dir': str(FEATURE_DIR) if FEATURE_DIR else '',
                                   'annotation_file': str(ANNOTATION_FILE) if ANNOTATION_FILE else '',
                                   'scratch_directory': str(CURRENT),
                                   'warm_model': WARM.describe()})
        if route.path == '/api/usage':
            # Which process holds the model right now, and what it and the GPU are using.
            with LOCK:
                running = next((job for job in JOBS.values() if job['process'].poll() is None), None)
            warm = WARM.process
            if running is not None and running['mode'] == 'video':
                pid, role = running['process'].pid, 'full-video runner'
            elif warm is not None and warm.poll() is None:
                pid, role = warm.pid, 'window model' if running is not None else 'window model, loaded and idle'
            else:
                pid, role = None, None
            return self.send(200, {**usage.sample(pid), 'role': role, 'running': running is not None})
        if route.path == '/api/lucky':
            try:
                target = parse_qs(route.query).get('target', ['window'])[0]
                if target not in STAGE_TARGETS:
                    raise ValueError('Invalid target stage')
                return self.send(200, choose_random_video(LIBRARY, target in ('refine', 'evaluate')))
            except ValueError as exc:
                return self.send(400, {'error': str(exc)})
        args = parse_qs(route.query)
        job_id = args.get('job', [''])[0]
        with LOCK:
            job = JOBS.get(job_id)
        if route.path == '/api/status':
            if job is None:
                return self.send(404, {'error': 'Unknown job'})
            proc = job['process']
            state = 'running' if proc.poll() is None else ('complete' if proc.returncode == 0 else 'failed')
            result_file = job['folder'] / 'probe' / 'result.json'
            result = json.loads(result_file.read_text()) if state == 'complete' and result_file.is_file() else None
            log_file = job['folder'] / 'console.log'
            log = log_file.read_text(errors='replace')[-4000:] if state != 'running' and log_file.is_file() else ''
            progress = None
            if job['mode'] == 'video':
                output = job['folder'] / PIPELINE_OUTPUT
                status_file = output / 'status.json'
                coverage_file = output / job['video_id'] / 'coverage.json'
                if status_file.is_file():
                    progress = json.loads(status_file.read_text())
                    if coverage_file.is_file():
                        progress['total_windows'] = len(json.loads(coverage_file.read_text())['selected_anchors'])
            stages = [] if state == 'running' else stage_manifest(job['folder'], job['mode'], job['video_id'])
            if state != 'running':
                manifest = job['folder'] / 'stage_manifest.json'
                if not manifest.exists():
                    manifest.write_text(json.dumps({'mode': job['mode'], 'video_id': job['video_id'],
                                                    'stages': stages}, indent=2) + '\n')
            return self.send(200, {'state': state, 'result': result, 'log': log,
                                   'job': job_id, 'returncode': proc.returncode,
                                   'mode': job['mode'], 'stages': stages, 'progress': progress,
                                   'scratch_directory': str(job['folder'])})
        if route.path in ('/api/artifact', '/api/array'):
            if job is None:
                return self.send(404, {'error': 'Unknown job'})
            try:
                selected = output_file(job, args.get('file', [''])[0])
                if route.path == '/api/array':
                    if selected.suffix != '.npy':
                        raise ValueError('Array preview requires an .npy file')
                    import numpy as np
                    values = np.load(selected, mmap_mode='r', allow_pickle=False)
                    if values.ndim != 1:
                        raise ValueError('Only one-dimensional score arrays can be previewed')
                    stride = max(1, math.ceil(len(values) / 500))
                    return self.send(200, {'length': len(values), 'stride': stride,
                                           'values': values[::stride].tolist(),
                                           'minimum': float(values.min()),
                                           'maximum': float(values.max())})
                mime = {'.json': 'application/json', '.jsonl': 'application/x-ndjson',
                        '.txt': 'text/plain; charset=utf-8', '.log': 'text/plain; charset=utf-8',
                        '.png': 'image/png', '.jpg': 'image/jpeg'}.get(selected.suffix, 'application/octet-stream')
                return self.send(200, selected.read_bytes(), mime)
            except (ValueError, OSError) as exc:
                return self.send(400, {'error': str(exc)})
        if route.path == '/api/windows':
            if job is None:
                return self.send(404, {'error': 'Unknown job'})
            # Compact per-window view of the runner's raw responses, without token IDs.
            video = PIPELINE_OUTPUT / job['video_id']
            raw = job['folder'] / video / 'raw_responses.jsonl'
            if job['mode'] != 'video' or job['process'].poll() is None or not raw.is_file():
                return self.send(200, {'windows': []})
            keep = ('anchor', 'start', 'end', 'condition', 'frame_indices', 'strict_label', 'raw_response', 'seconds')
            rows = [{key: row[key] for key in keep} for row in map(json.loads, raw.read_text().splitlines())]
            info = job['folder'] / video / 'video.json'
            fps = json.loads(info.read_text())['fps_actual'] if info.is_file() else 30.0
            return self.send(200, {'windows': rows, 'fps': fps, 'inputs': str(video / 'inputs')})
        if route.path == '/api/list':
            if job is None:
                return self.send(404, {'error': 'Unknown job'})
            try:
                folder = output_path(job, args.get('dir', [''])[0])
                if not folder.is_dir():
                    raise ValueError('Artifact is not a directory')
                offset = int(args.get('offset', ['0'])[0])
                if offset < 0:
                    raise ValueError('Offset must be nonnegative')
                children = sorted(folder.iterdir(), key=lambda p: p.name)
                root = job['folder'].resolve()
                visible = []
                for path in children[offset:offset + 50]:
                    if path.is_symlink() or not path.resolve().is_relative_to(root):
                        continue
                    kind = 'directory' if path.is_dir() else path.suffix.lower().lstrip('.')
                    visible.append({'path': str(path.relative_to(root)), 'kind': kind})
                return self.send(200, {'directory': str(folder.relative_to(root)),
                                       'total': len(children), 'offset': offset,
                                       'items': visible, 'next_offset': offset + 50 if offset + 50 < len(children) else None})
            except (ValueError, OSError) as exc:
                return self.send(400, {'error': str(exc)})
        if route.path == '/api/image':
            if job is None:
                return self.send(404, {'error': 'Unknown job'})
            try:
                slot = int(args.get('slot', [''])[0])
            except ValueError:
                return self.send(400, {'error': 'Invalid slot'})
            if slot not in range(8):
                return self.send(400, {'error': 'Slot outside 0–7'})
            matches = sorted((job['folder'] / 'probe' / 'frames').glob(f'{slot:02d}_source_*.png'))
            frame = matches[0] if matches else None
            return self.send(200, frame.read_bytes(), 'image/png') if frame and frame.is_file() else self.send(404, {'error': 'Frame unavailable'})
        return self.send(404, {'error': 'Unknown path'})

    def do_POST(self):
        if not self.local_host():
            return self.send(403, {'error': 'VERA Lab accepts local hostnames only'})
        # A localhost web page can be targeted by another browser origin. A
        # custom header prevents simple cross-origin forms from starting a
        # model run, and the token is unreadable to another origin.
        if self.headers.get('X-VERA-Token') != SESSION_TOKEN:
            return self.send(403, {'error': 'Reload VERA Lab before submitting a run'})
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers.get('Host', ''):
            return self.send(403, {'error': 'Cross-origin requests are not accepted'})
        if self.path == '/api/upload':
            return self.upload()
        if self.path != '/api/run':
            return self.send(404, {'error': 'Unknown path'})
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length < 1 or length > 16384:
                raise ValueError('Request must be 1–16384 bytes')
            request = json.loads(self.rfile.read(length))
            if self.headers.get('Content-Type', '').split(';', 1)[0].strip() != 'application/json':
                raise ValueError('Run request must be application/json')
            video = Path(request['video']).expanduser().resolve()
            if video.suffix.lower() not in VIDEO_SUFFIXES or not video.is_file():
                raise ValueError('Enter an existing video path (.mp4, .avi, .mov, .mkv)')
            pending = RUNS / 'uploads'
            uploaded = video.is_relative_to(pending.resolve())
            video_name = video.name.split('_', 1)[1] if uploaded else video.name
            video_id = Path(video_name).stem
            anchor = float(request.get('anchor_seconds', 0))
            if not math.isfinite(anchor) or anchor < 0:
                raise ValueError('Anchor time must be a nonnegative finite number')
            device = request.get('device', 'auto')
            if device not in ('auto', 'mps', 'cuda', 'cpu'):
                raise ValueError('Invalid device')
            dtype = request.get('dtype', 'auto')
            if dtype not in ('auto', 'bfloat16', 'float16', 'float32'):
                raise ValueError('Invalid dtype')
            target_stage = request.get('target_stage', 'window')
            if target_stage not in STAGE_TARGETS:
                raise ValueError('Choose a supported stage: window, preprocess, segments, refine, evaluate')
            missing = missing_dependencies(target_stage)
            if missing:
                raise ValueError('This Python environment is missing ' + ', '.join(missing) +
                                 '. Run the lab in the Oz VERA environment or install its pinned dependencies; see DEMO.md.')
            model_path = request.get('model_path', MODEL_PATH)
            if not isinstance(model_path, str) or not model_path or len(model_path) > 512:
                raise ValueError('Invalid model path')
            questions = request.get('questions', '')
            if not isinstance(questions, str) or len(questions) > 3000:
                raise ValueError('Questions too long')
            qlist = [line.strip() for line in questions.splitlines() if line.strip()]
            if len(qlist) > 10:
                raise ValueError('Use at most 10 question lines')
            allow_download = request.get('allow_download', False)
            if type(allow_download) is not bool:
                raise ValueError('Invalid download setting')
            feature_path = None
            annotation_path = None
            if target_stage in ('refine', 'evaluate'):
                feature_path = Path(request.get('feature_file', '')).expanduser().resolve()
                if not feature_path.is_file() or feature_path.name != video_id + '.npy':
                    raise ValueError('Refinement needs a feature .npy named ' + video_id + '.npy')
            if target_stage == 'evaluate':
                annotation_path = Path(request.get('annotation_file', '')).expanduser().resolve()
                if not annotation_path.is_file() or annotation_path.suffix != '.json':
                    raise ValueError('Evaluation needs an existing UCF-style annotation JSON file')
            job_id = uuid.uuid4().hex[:12]
            folder = CURRENT
            cfg = json.loads(BASE_CONFIG.read_text())
            # The pinned commit belongs to the baseline repository and verified local copies.
            # Another Hub repository uses its default branch; the backend records the loaded commit.
            pinned = model_path == cfg['model']['path'] or runner_path(model_path).is_dir()
            # Defaults follow upstream VERA: FlashAttention on CUDA, BF16, 1024 new tokens (baseline.json).
            cfg['model'].update(path=model_path, device=device, dtype=dtype, flash_attention='auto',
                                revision=cfg['model']['revision'] if pinned else None,
                                local_files_only=not allow_download,
                                # 'auto' resolves to MPS on a Mac, so it needs the same CPU fallback.
                                mps_cpu_fallback=device in ('mps', 'auto'))
            cfg['prompt']['guiding_questions'] = qlist if qlist else 'upstream_ucf'
            mode = 'window' if target_stage == 'window' else 'video'
            cfg['runtime']['timeout_seconds'] = WINDOW_TIMEOUT if mode == 'window' else VIDEO_TIMEOUT
            cfg['stages'] = STAGE_TARGETS[target_stage]
            with LOCK:
                if any(j['process'].poll() is None for j in JOBS.values()):
                    raise ValueError('A model call is already running. Wait for it to finish.')
                # Only the tool-owned scratch location is cleared. Source videos and features stay untouched.
                if folder.exists():
                    shutil.rmtree(folder)
                folder.mkdir(parents=True)
                if uploaded:
                    upload_dir = folder / 'uploaded'
                    upload_dir.mkdir()
                    stored = upload_dir / video_name
                    shutil.move(str(video), str(stored))
                    video = stored
                if pending.exists():
                    shutil.rmtree(pending)
                config_file = folder / 'config.json'
                if mode == 'window':
                    config_file.write_text(json.dumps(cfg, indent=2) + '\n')
                    job = WARM.submit(cfg['model'], {
                        'config': str(config_file), 'video': str(video),
                        'anchor_seconds': anchor, 'output': str(folder / 'probe')},
                        folder / 'console.log')
                else:
                    # The runner loads its own copy of the model; free the warm one first.
                    WARM.stop()
                    inputs = folder / 'input'
                    inputs.mkdir()
                    (inputs / video.name).symlink_to(video)
                    split = {'videos': str(inputs)}
                    if feature_path is not None:
                        split['features'] = str(feature_path.parent)
                    if annotation_path is not None:
                        split['annotations'] = str(annotation_path)
                    registry = {'lab_video': {'annotation_format': 'vera_frames',
                                              'splits': {'test': split}}}
                    (folder / 'registry.json').write_text(json.dumps(registry, indent=2) + '\n')
                    cfg.update(dataset='lab_video', split='test', registry=str(folder / 'registry.json'),
                               output_root=str(folder / 'pipeline'))
                    cfg['selection']['video_ids'] = [video.stem]
                    config_file.write_text(json.dumps(cfg, indent=2) + '\n')
                    cmd = [sys.executable, str(ROOT / 'run_vera.py'), '--config', str(config_file)]
                    with (folder / 'console.log').open('w') as log_file:
                        # Fixed argument list without a shell; user input is only in the config file.
                        job = subprocess.Popen(cmd, cwd=ROOT, stdout=log_file,  # nosemgrep
                                               stderr=subprocess.STDOUT)
                    threading.Thread(target=stop_stalled_job, args=(job, folder, VIDEO_TIMEOUT),
                                     daemon=True).start()
                JOBS.clear()
                JOBS[job_id] = {'process': job, 'folder': folder, 'mode': mode,
                                'video_id': video_id, 'target_stage': target_stage}
                (folder / 'lab_job.json').write_text(json.dumps({
                    'job': job_id, 'mode': mode, 'video_id': video_id,
                    'target_stage': target_stage}, indent=2) + '\n')
            return self.send(202, {'job': job_id, 'scratch_directory': str(folder)})
        except (KeyError, ValueError, TypeError, OSError, json.JSONDecodeError) as exc:
            return self.send(400, {'error': str(exc)})

    def upload(self):
        target = None
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length < 1 or length > 200 * 1024 * 1024:
                raise ValueError('Video upload must be between 1 byte and 200 MB')
            filename = Path(self.headers.get('X-Filename', '')).name
            suffix = Path(filename).suffix.lower()
            if suffix not in VIDEO_SUFFIXES:
                raise ValueError('Upload must be MP4, AVI, MOV, or MKV')
            directory = RUNS / 'uploads'
            with LOCK:
                if directory.exists():
                    shutil.rmtree(directory)
                directory.mkdir(parents=True)
            target = directory / (uuid.uuid4().hex + '_' + filename)
            remaining = length
            with target.open('xb') as output:
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError('Upload ended before declared Content-Length')
                    output.write(chunk)
                    remaining -= len(chunk)
            return self.send(201, {'path': str(target), 'name': filename, 'bytes': length})
        except (ValueError, OSError) as exc:
            if target is not None:
                target.unlink(missing_ok=True)
            return self.send(400, {'error': str(exc)})


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--video-dir', help='Local video library for the random sample button')
    ap.add_argument('--model-path', default='OpenGVLab/InternVL2-8B',
                    help='Default InternVL2 checkpoint directory or Hub ID')
    ap.add_argument('--feature-dir', help='Directory of aligned per-video ImageBind .npy files')
    ap.add_argument('--annotation-file', help='UCF-style temporal-label manifest for evaluation')
    args = ap.parse_args()
    global LIBRARY, MODEL_PATH, FEATURE_DIR, ANNOTATION_FILE
    LIBRARY = Path(args.video_dir).expanduser().resolve() if args.video_dir else None
    MODEL_PATH = args.model_path
    FEATURE_DIR = Path(args.feature_dir).expanduser().resolve() if args.feature_dir else None
    ANNOTATION_FILE = Path(args.annotation_file).expanduser().resolve() if args.annotation_file else None
    if LIBRARY is not None and not LIBRARY.is_dir():
        ap.error(f'video-dir does not exist: {LIBRARY}')
    if FEATURE_DIR is not None and not FEATURE_DIR.is_dir():
        ap.error(f'feature-dir does not exist: {FEATURE_DIR}')
    if ANNOTATION_FILE is not None and not ANNOTATION_FILE.is_file():
        ap.error(f'annotation-file does not exist: {ANNOTATION_FILE}')
    RUNS.mkdir(parents=True, exist_ok=True)
    server_lock = (RUNS / '.server.lock').open('w')
    try:
        fcntl.flock(server_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        ap.error('Another VERA Lab server already owns this scratch directory')
    restore_current_job()
    atexit.register(WARM.stop)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'Open http://127.0.0.1:{args.port}/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
