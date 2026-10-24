#!/usr/bin/env python3
"""Run one real VERA window through the existing InternVL2 backend.

This is an educational probe. It does not perform whole-video refinement or AUC.
"""
import argparse
import io
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vera_runner.config import load
from vera_runner.data import windows
from vera_runner.backends import BACKENDS, prompt, strict_label


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--video', required=True)
    ap.add_argument('--anchor-seconds', type=float, required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    video = Path(args.video).expanduser().resolve()
    if not video.is_file():
        raise ValueError(f'Video not found: {video}')
    if args.anchor_seconds < 0:
        raise ValueError('anchor-seconds must be nonnegative')
    cfg = load(args.config)
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' if cfg['model']['mps_cpu_fallback'] else '0'
    if cfg['stages'] != ['preprocess', 'infer']:
        raise ValueError('Window probe requires stages=[preprocess,infer]')
    if cfg['model']['backend'] not in ('internvl2', 'mock'):
        raise ValueError('Unknown backend')
    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
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
    target = args.anchor_seconds * actual_fps
    win = min(anchors, key=lambda w: abs(w['anchor'] - target))
    images, decoded, fps = extract(video, win['frame_indices'], cfg['preprocessing']['image_size'])
    frame_dir = out / 'frames'
    frame_dir.mkdir()
    for index, image in enumerate(images):
        image.save(frame_dir / f'{index:02d}_source_{win["frame_indices"][index]:06d}.png')
    (out / 'sample_manifest.json').write_text(json.dumps({
        'video': str(video), 'video_frames': decoded, 'actual_fps': fps,
        'anchor_frame': win['anchor'], 'window_start_frame': win['start'],
        'window_end_frame': win['end'], 'sampled_frame_indices': win['frame_indices'],
        'sampled_times_seconds': [i / fps for i in win['frame_indices']],
        'frame_files': [f'{i:02d}_source_{source:06d}.png'
                        for i, source in enumerate(win['frame_indices'])],
    }, indent=2) + '\n', encoding='utf-8')
    (out / 'prompt.txt').write_text(prompt(cfg), encoding='utf-8')
    start = time.monotonic()
    backend = BACKENDS[cfg['model']['backend']](cfg)
    prediction = backend.predict(images)
    elapsed = time.monotonic() - start
    result = {
        'video': str(video), 'video_frames': decoded, 'actual_fps': fps,
        'requested_anchor_seconds': args.anchor_seconds,
        'anchor_frame': win['anchor'], 'window_start_frame': win['start'],
        'window_end_frame': win['end'], 'sampled_frame_indices': win['frame_indices'],
        'sampled_times_seconds': [i / fps for i in win['frame_indices']],
        'model_backend': cfg['model']['backend'], 'model_path': cfg['model']['path'],
        'device': backend.device, 'raw_response': prediction['raw_response'],
        'parsed_label': strict_label(prediction['raw_response']), 'elapsed_seconds': elapsed,
        'synthetic': cfg['model']['backend'] == 'mock',
        'note': 'One window only. No scene refinement, full timeline, or evaluation.'
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'result': str(out / 'result.json'), 'parsed_label': result['parsed_label']}))


if __name__ == '__main__':
    main()
