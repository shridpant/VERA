#!/usr/bin/env python3
"""Run one real VERA window through the existing InternVL2 backend.

This is an educational probe. It does not perform whole-video refinement or AUC.
With --serve it stays alive and answers one JSON request per stdin line, so
repeated windows reuse the loaded model instead of reloading its weights. A request
with "kind": "ground" boxes a phrase on chosen saved images instead (manual boxes).
Every step is appended to timeline.jsonl for the lab's live view.
"""
import argparse
import io
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vera_runner.config import load, prepare_environment
from vera_runner.data import windows
from vera_runner.backends import BACKENDS, prompt, strict_label
from vera_runner.grounding import box_images, box_window
from vera_runner.timeline import Timeline


def extract(video, indices, size):
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {video}')
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    wanted = set(indices)
    found = {}
    count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count in wanted:
                ok, jpg = cv2.imencode('.jpg', frame)
                if not ok:
                    raise ValueError(f'JPEG encoding failed at frame {count}')
                with Image.open(io.BytesIO(jpg.tobytes())) as source:
                    found[count] = source.convert('RGB').resize((size, size))
            count += 1
    finally:
        cap.release()
    if expected > 0 and count != expected:
        raise ValueError(f'Incomplete decode: expected {expected}, decoded {count}')
    if set(indices) - found.keys():
        raise ValueError('At least one selected frame was not decoded')
    return [found[i] for i in indices], count, fps


def load_backend(cfg, timeline):
    timeline.emit('model_loading', path=cfg['model']['path'])
    begin = time.monotonic()
    backend = BACKENDS[cfg['model']['backend']](cfg)
    identity = getattr(backend, 'model_identity', {})
    timeline.emit('model_ready', device=backend.device, dtype=identity.get('dtype'),
                  flash_attention=identity.get('flash_attention'), seconds=time.monotonic() - begin)
    return backend


def fresh_backend(cfg, timeline):
    return load_backend(cfg, timeline), False


def probe(cfg, video, anchor_seconds, out, backend_for):
    """Sample one window, save its exact inputs, and ask the backend once."""
    video = Path(video).expanduser().resolve()
    if not video.is_file():
        raise ValueError(f'Video not found: {video}')
    if not math.isfinite(anchor_seconds) or anchor_seconds < 0:
        raise ValueError('anchor-seconds must be a nonnegative finite number')
    if cfg['stages'] != ['preprocess', 'infer']:
        raise ValueError('Window probe requires stages=[preprocess,infer]')
    if cfg['model']['backend'] not in ('internvl2', 'mock'):
        raise ValueError('Unknown backend')
    out = Path(out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    timeline = Timeline(out / 'timeline.jsonl')
    timeline.emit('run', stages=['window'], videos=[video.name], model=cfg['model']['path'], boxes=cfg['boxes']['mode'])
    try:
        result = sample_and_ask(cfg, video, anchor_seconds, out, backend_for, timeline)
    except Exception as exc:
        timeline.emit('error', message=f'{type(exc).__name__}: {exc}')
        timeline.emit('done', status='failed')
        raise
    timeline.emit('done', status='complete')
    return result


def sample_and_ask(cfg, video, anchor_seconds, out, backend_for, timeline):
    import cv2
    cap = cv2.VideoCapture(str(video))
    try:
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if length < 1 or actual_fps <= 0:
        raise ValueError('Video must report positive frame count and FPS')
    anchors = list(windows(length, cfg['sampling']))
    target = anchor_seconds * actual_fps
    win = min(anchors, key=lambda w: abs(w['anchor'] - target))
    timeline.emit('video', video=video.name, frames=length, fps=actual_fps, windows=len(anchors))
    images, decoded, fps = extract(video, win['frame_indices'], cfg['preprocessing']['image_size'])
    frame_dir = out / 'frames'
    frame_dir.mkdir()
    paths = []
    for index, image in enumerate(images):
        name = f'{index:02d}_source_{win["frame_indices"][index]:06d}.png'
        image.save(frame_dir / name)
        paths.append('frames/' + name)
    where = dict(video=video.name, anchor=win['anchor'], condition='baseline')
    timeline.emit('window', **where, index=anchors.index(win), total=len(anchors), start=win['start'], end=win['end'],
                  frame_indices=win['frame_indices'], images=paths)
    (out / 'sample_manifest.json').write_text(json.dumps({
        'video': str(video), 'video_frames': decoded, 'actual_fps': fps,
        'anchor_frame': win['anchor'], 'window_start_frame': win['start'],
        'window_end_frame': win['end'], 'sampled_frame_indices': win['frame_indices'],
        'sampled_times_seconds': [i / fps for i in win['frame_indices']],
        'frame_files': [f'{i:02d}_source_{source:06d}.png'
                        for i, source in enumerate(win['frame_indices'])],
    }, indent=2) + '\n', encoding='utf-8')
    (out / 'prompt.txt').write_text(prompt(cfg), encoding='utf-8')
    timeline.emit('prompt', text=prompt(cfg), generation=cfg['generation'])
    start = time.monotonic()
    backend, reused = backend_for(cfg, timeline)
    loaded = time.monotonic()
    prediction = backend.predict(images, on_text=timeline.streamer(**where))
    finished = time.monotonic()
    label = strict_label(prediction['raw_response'])
    rendered = backend.rendered_input()
    (out / 'model_input.txt').write_text(rendered, encoding='utf-8')
    timeline.emit('model_input', text=rendered)
    timeline.emit('answer', **where, text=prediction['raw_response'], label=label, seconds=finished - loaded)
    boxes = None
    if cfg['boxes']['mode'] == 'alerts' and label == 1:
        boxes = box_window(backend, images, prediction['raw_response'], cfg['boxes'], timeline,
                           win['frame_indices'], paths, **where)
        (out / 'boxes.json').write_text(json.dumps(boxes, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    result = {
        'video': str(video), 'video_frames': decoded, 'actual_fps': fps,
        'requested_anchor_seconds': anchor_seconds,
        'anchor_frame': win['anchor'], 'window_start_frame': win['start'],
        'window_end_frame': win['end'], 'sampled_frame_indices': win['frame_indices'],
        'sampled_times_seconds': [i / fps for i in win['frame_indices']],
        'model_backend': cfg['model']['backend'], 'model_path': cfg['model']['path'],
        'device': backend.device, 'dtype': getattr(backend, 'model_identity', {}).get('dtype'),
        'raw_response': prediction['raw_response'],
        'parsed_label': label, 'boxes': boxes,
        'model_reused': reused, 'model_load_seconds': loaded - start,
        'inference_seconds': finished - loaded, 'elapsed_seconds': finished - start,
        'synthetic': cfg['model']['backend'] == 'mock',
        'note': 'One window only. No scene refinement, full timeline, or evaluation.'
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


def ground_saved(request, backend_for):
    """Manual boxes: one phrase on saved model inputs, grouped by window, appended to request['output']."""
    from PIL import Image
    cfg = load(request['config'])
    timeline = Timeline(request['timeline'])
    phrase = request['phrase'].strip()
    if not phrase or len(phrase) > 120:
        raise ValueError('Give a phrase of 1-120 characters to box')
    timeline.emit('manual_boxes', phrase=phrase, frames=len(request['items']))
    backend, _ = backend_for(cfg, timeline)
    windows_done = []
    for anchor in sorted({item['anchor'] for item in request['items']}):
        items = []
        for item in (entry for entry in request['items'] if entry['anchor'] == anchor):
            with Image.open(item['file']) as image:
                items.append({'image': image.convert('RGB'), 'slot': item['slot'],
                              'frame_index': item['frame_index'], 'path': item['path']})
        rows = box_images(backend, items, phrase, cfg['boxes']['max_new_tokens'], timeline, anchor=anchor, mode='manual')
        record = {'anchor': anchor, 'mode': 'manual', 'phrase': phrase, 'frames': rows}
        with open(request['output'], 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        windows_done.append(anchor)
    timeline.emit('manual_done', windows=windows_done)
    return {'ok': True, 'boxed': len(request['items']), 'device': backend.device,
            'dtype': getattr(backend, 'model_identity', {}).get('dtype')}


def serve(idle_seconds):
    """Answer one JSON request per stdin line until EOF or idle_seconds without a request."""
    # Model libraries may print while loading. Keep the real stdout for one JSON reply per request.
    replies = os.fdopen(os.dup(1), 'w', buffering=1)
    os.dup2(2, 1)
    loaded = {}

    def backend_for(cfg, timeline):
        if 'backend' in loaded:
            loaded['backend'].configure(cfg)
            timeline.emit('model_reused', device=loaded['backend'].device)
            return loaded['backend'], True
        prepare_environment(cfg)
        loaded['backend'] = load_backend(cfg, timeline)
        return loaded['backend'], False

    def idle(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGALRM, idle)
    while True:
        # The idle alarm covers only the wait for the next request, never a running one.
        signal.setitimer(signal.ITIMER_REAL, idle_seconds)
        line = sys.stdin.readline()
        signal.setitimer(signal.ITIMER_REAL, 0)
        if not line:
            break
        try:
            request = json.loads(line)
            if request.get('kind') == 'ground':
                reply = ground_saved(request, backend_for)
            else:
                result = probe(load(request['config']), request['video'], float(request['anchor_seconds']),
                               request['output'], backend_for)
                reply = {'ok': True, 'parsed_label': result['parsed_label'], 'model_reused': result['model_reused'],
                         'device': result['device'], 'dtype': result['dtype']}
        except Exception:
            reply = {'ok': False, 'error': traceback.format_exc()}
            print(reply['error'], file=sys.stderr, flush=True)
        replies.write(json.dumps(reply) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config')
    ap.add_argument('--video')
    ap.add_argument('--anchor-seconds', type=float)
    ap.add_argument('--output')
    ap.add_argument('--serve', action='store_true', help='Keep the model loaded and read JSON requests from stdin')
    ap.add_argument('--idle-seconds', type=float, default=900, help='Exit --serve after this long without a request')
    args = ap.parse_args()
    if args.serve:
        return serve(args.idle_seconds)
    if None in (args.config, args.video, args.anchor_seconds, args.output):
        ap.error('--config, --video, --anchor-seconds and --output are required without --serve')
    cfg = load(args.config)
    prepare_environment(cfg)
    result = probe(cfg, args.video, args.anchor_seconds, args.output, fresh_backend)
    print(json.dumps({'result': str(Path(args.output).resolve() / 'result.json'),
                      'parsed_label': result['parsed_label']}))


if __name__ == '__main__':
    main()
