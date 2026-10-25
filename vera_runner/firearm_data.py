"""VISILAB firearm-action dataset: inventory, stable staged names and the runner registry.

Source: Ruiz-Santaquiteria et al., Data in Brief 52 (2024) 110030; the same zip is Mendeley
10.17632/bbzpxhd22j.2 and Zenodo 15387426. Folder names encode the action code, camera (C1 IP
camera 640x480 25 FPS, C2 phone 1920x1080 30 FPS), location P1-P5, subject V1-V4, brightness
HB/LB and take. Every video file is called video.mp4, so each one gets a staged ID
'<Category>__<folder>', a symlink to the read-only original. Nothing is copied or overwritten.
"""
import csv
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from .data import sha256

CATEGORIES = ('Handgun', 'Machine_Gun', 'No_Gun')
NAME = re.compile(r'^(N\d+|P[CA][HW]\d+)_(C[12])_(P\d)_(V\d)_(HB|LB)_(\d+)$')
CAMERA = {'C1': 'cctv_ip', 'C2': 'phone'}
LABEL_FILES = ('label.json', 'label_antiguo.json')


def parse_folder(name):
    match = NAME.match(name)
    if not match:
        raise ValueError(f'Unrecognised video folder name: {name}')
    action, camera, location, subject, brightness, take = match.groups()
    # Repeated takes of one scripted action by one subject at one place: a dependence group.
    return {'action': action, 'camera': camera, 'camera_type': CAMERA[camera], 'location': location,
            'subject': subject, 'brightness': brightness, 'take': int(take),
            'recording_group': '_'.join([action, camera, location, subject, brightness])}


def staged_id(category, folder):
    return f'{category}__{folder}'


def expected_windows(frames, stride=16):
    """Anchors range(0, frames, stride), as data.windows and upstream VERA schedule them."""
    return -(-frames // stride)


def probe(path):
    """Container metadata and a full OpenCV decode count (the decoder data.preprocess uses)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    meta = {'fps': cap.get(cv2.CAP_PROP_FPS), 'frames_meta': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    decoded, shapes = 0, set()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        decoded += 1
        shapes.add((frame.shape[1], frame.shape[0]))
    cap.release()
    return {**meta, 'frames_decoded': decoded, 'decoded_sizes': sorted(shapes), 'opencv_version': cv2.__version__}


def _row(job):
    original_root, category, folder = job
    folder_path = Path(original_root) / category / folder
    video = folder_path / 'video.mp4'
    labels = [name for name in LABEL_FILES if (folder_path / name).is_file()]
    extra = sorted(p.name for p in folder_path.iterdir() if p.name not in ('video.mp4', *LABEL_FILES))
    info = probe(video)
    row = {'id': staged_id(category, folder), 'category': category, 'folder': folder, **parse_folder(folder),
           'source_path': str(video.relative_to(original_root)), 'bytes': video.stat().st_size,
           'sha256': sha256(video), **info,
           'duration_s': info['frames_decoded'] / info['fps'] if info['fps'] else None,
           'expected_windows': expected_windows(info['frames_decoded']),
           'label_file': labels[0] if labels else None, 'label_files': labels,
           'label_sha256': sha256(folder_path / labels[0]) if labels else None, 'unexpected_files': extra}
    return row


def build_inventory(original_root, workers=16):
    original_root = Path(original_root)
    jobs = []
    for category in sorted(p.name for p in original_root.iterdir() if p.is_dir()):
        if category not in CATEGORIES:
            raise ValueError(f'Unexpected category folder: {category}')
        jobs += [(str(original_root), category, folder.name)
                 for folder in sorted((original_root / category).iterdir()) if folder.is_dir()]
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(_row, jobs))
    ids = [row['id'] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Staged IDs are not unique')
    return rows


def stage(rows, original_root, staged_root):
    """One symlink per video; an existing link must already point to the same original."""
    staged_root = Path(staged_root)
    staged_root.mkdir(parents=True, exist_ok=True)
    for row in rows:
        target = (Path(original_root) / row['source_path']).resolve()
        link = staged_root / (row['id'] + '.mp4')
        if link.is_symlink() or link.exists():
            if not link.is_symlink() or Path(os.readlink(link)).resolve() != target:
                raise FileExistsError(f'{link} exists and does not link to {target}')
        else:
            link.symlink_to(target)
        row['staged_path'] = str(link)
    return rows


def write_inventory(rows, directory):
    directory = Path(directory)
    (directory / 'inventory.json').write_text(json.dumps(rows, indent=1) + '\n')
    flat = [{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()} for row in rows]
    with (directory / 'inventory.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)


def write_registry(path, staged_root, features_dir):
    """A runner registry: no annotation manifest (categories are not anomaly labels)."""
    registry = {'firearm_visilab': {'annotation_format': 'none',
                                    'splits': {'test': {'videos': str(staged_root), 'features': str(features_dir)}}}}
    Path(path).write_text(json.dumps(registry, indent=2) + '\n')
    return registry
