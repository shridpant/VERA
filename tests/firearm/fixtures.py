"""Synthetic VISILAB-shaped datasets for the firearm study tests. Nothing here is real data."""
import json
from pathlib import Path
import numpy as np


def write_video(path, frames, size=(64, 48), fps=25, value=None):
    """mp4v video; frame i is a flat grey (i*7 % 255) unless value(i) gives an HxWx3 array."""
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for i in range(frames):
        writer.write(value(i) if value else np.full((size[1], size[0], 3), i * 7 % 255, np.uint8))
    writer.release()


def coco(frames, size, boxes):
    """label.json as the dataset writes it: one image per frame (ids 1..N), boxes {image_id: xywh}."""
    return {'info': {'year': 2023}, 'licenses': [{'id': '1', 'name': 'CC BY NC 3.0'}],
            'categories': [{'id': 1, 'name': 'Handgun'}],
            'images': [{'id': i + 1, 'width': size[0], 'height': size[1], 'file_name': 'video.mp4', 'license': '1'}
                       for i in range(frames)],
            'annotations': [{'id': n + 1, 'image_id': image_id, 'category_id': 1, 'area': b[2] * b[3], 'bbox': b, 'iscrowd': 0}
                            for n, (image_id, b) in enumerate(sorted(boxes.items()))]}


def make_dataset(root, videos):
    """videos: list of (category, folder, frames, size, fps, boxes or None)."""
    root = Path(root)
    for category, folder, frames, size, fps, boxes in videos:
        write_video(root / category / folder / 'video.mp4', frames, size, fps)
        if boxes is not None:
            (root / category / folder / 'label.json').write_text(json.dumps(coco(frames, size, boxes)))
    return root
