"""Dataset names resolve to explicit split roots and optional annotation manifests."""
import hashlib
import json
from pathlib import Path
from .config import path


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def resolve(cfg):
    registry = json.loads(path(cfg['registry']).read_text())
    try:
        dataset = registry[cfg['dataset']]
        split = dataset['splits'][cfg['split']]
    except KeyError as e:
        raise ValueError(f'Unregistered dataset/split: {cfg["dataset"]}/{cfg["split"]}') from e
    root = path(split['videos'])
    if not root.is_dir():
        raise ValueError(f'Video root does not exist: {root}')
    videos = {}
    for p in sorted(root.rglob('*')):
        if p.suffix.lower() in ('.mp4', '.avi', '.mkv', '.mov') and p.is_file():
            if p.stem in videos:
                raise ValueError(f'Duplicate video ID {p.stem}; IDs must be unique within a split')
            videos[p.stem] = p
    annfile = split.get('annotations')
    manifest = None
    if annfile:
        manifest = json.loads(path(annfile).read_text())
        if not isinstance(manifest, list):
            raise ValueError('Annotation file must be a JSON list')
        entries = {}
        for entry in manifest:
            vid = Path(entry['video']).stem
            if vid in entries:
                raise ValueError(f'Duplicate annotation ID {vid}')
            entries[vid] = entry
    else:
        entries = {vid: {'video': p.name} for vid, p in videos.items()}
    selected = cfg['selection']['video_ids'] or sorted(videos.keys() & entries.keys())
    if len(set(selected)) != len(selected):
        raise ValueError('Duplicate selected video IDs')
    for vid in selected:
        if vid not in videos or vid not in entries:
            raise ValueError(f'Video or manifest entry missing: {vid}')
    if cfg['selection']['max_videos']:
        selected = selected[:cfg['selection']['max_videos']]
    if not selected:
        raise ValueError('No matching videos')
    if 'evaluate' in cfg['stages'] and dataset.get('annotation_format', 'vera_frames') != 'vera_frames':
        raise ValueError('Evaluation requires vera_frames annotations; convert other timestamp units explicitly')
    if 'refine' in cfg['stages'] and not split.get('features'):
        raise ValueError('Refinement requires split.features: precomputed segment-aligned .npy vision features')
    result = []
    for vid in selected:
        item = {'id': vid, 'path': str(videos[vid]), 'annotation': entries[vid]}
        if split.get('features'):
            feat = path(split['features']) / (vid + '.npy')
            if 'refine' in cfg['stages'] and not feat.is_file():
                raise ValueError(f'Missing features: {feat}')
            item['features'] = str(feat)
        if 'evaluate' in cfg['stages'] and 'temporal_label' not in entries[vid]:
            raise ValueError(f'No ground truth intervals for {vid}; normal videos need [-1,-1]')
        result.append(item)
    return result


def windows(length, sampling):
    import numpy as np
    half = sampling['fps'] * sampling['window_seconds'] / 2
    for anchor in range(0, length, sampling['stride']):
        start, end = max(anchor - half, 0), min(anchor + half, length)
        indices = np.arange(start, end, (end - start) / sampling['frames_per_window']).astype(int).tolist()
        if len(indices) != sampling['frames_per_window']:
            raise ValueError('Upstream sampling arithmetic did not produce the requested frame count')
        yield {'anchor': anchor, 'start': start, 'end': end, 'frame_indices': indices}


def preprocess(item, directory):
    import cv2
    directory.mkdir()
    cap = cv2.VideoCapture(item['path'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n = 0
    dimensions = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dimensions = [frame.shape[1], frame.shape[0]]
        if not cv2.imwrite(str(directory / f'{n:06d}.jpg'), frame):
            raise RuntimeError('JPEG encoding failed')
        n += 1
    cap.release()
    if not n or (expected > 0 and expected != n):
        raise RuntimeError(f'Incomplete decode for {item["id"]}: {n}/{expected}')
    if 'length' in item['annotation'] and item['annotation']['length'] != n:
        raise ValueError(f'Manifest length differs from decoded length: {item["id"]}')
    return {'decoded_frames': n, 'fps_actual': fps, 'native_size': dimensions,
            'video_sha256': sha256(item['path']), 'opencv_version': cv2.__version__,
            'jpeg_default_quality': True}
