"""Existing VISILAB firearm boxes: validation, frame mapping and the model's coordinate system.

label.json is COCO with one image entry per decoded frame (ids 1..N, every file_name 'video.mp4'),
at most one firearm box per image, bbox = [x, y, w, h] in native pixels. The image_id-to-frame
origin and which frame statuses are complete come from the audit (studies/firearm/AUDIT.md),
passed in as `annotation_policy.json`; nothing here assumes them.

The model sees each frame after the runner's non-aspect-preserving PIL resize to 448x448 and
answers on a 0-1000 grid of that input. A per-axis scale preserves normalized coordinates, so a
native box maps to 0-1000 by x/W*1000 and y/H*1000, which equals its 448-input pixel box times
1000/448. IoU is invariant under that per-axis scaling.
"""
import json
from pathlib import Path

SCALE = 1000
STATUSES = ('positive', 'listed_no_box', 'negative_category', 'unknown')


def load_coco(path):
    return json.loads(Path(path).read_text())


def validate_coco(coco, frames, width, height, category):
    """Problems that would make the boxes unusable, as readable strings (empty list = clean)."""
    problems = []
    ids = [im['id'] for im in coco['images']]
    if sorted(ids) != list(range(1, frames + 1)):
        problems.append(f'image ids are not 1..{frames} (n={len(ids)}, min={min(ids, default=None)}, max={max(ids, default=None)})')
    sizes = {(im['width'], im['height']) for im in coco['images']}
    if sizes != {(width, height)}:
        problems.append(f'image sizes {sorted(sizes)} differ from the video {width}x{height}')
    names = {c['id']: c['name'] for c in coco['categories']}
    if set(names.values()) != {category}:
        problems.append(f'categories {names} differ from the folder category {category}')
    seen = set()
    for a in coco['annotations']:
        x, y, w, h = a['bbox']
        if a['category_id'] not in names:
            problems.append(f"annotation {a['id']} has unknown category id {a['category_id']}")
        if a['image_id'] not in ids:
            problems.append(f"annotation {a['id']} refers to missing image {a['image_id']}")
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width + 0.5 or y + h > height + 0.5:
            problems.append(f"annotation {a['id']} box {a['bbox']} is empty or outside {width}x{height}")
        if a['image_id'] in seen:
            problems.append(f"image {a['image_id']} has more than one box")
        seen.add(a['image_id'])
    return problems


def frame_index(image_id, origin):
    """Decoded zero-based frame index of a COCO image id under the audited origin (0 or 1)."""
    if origin not in (0, 1):
        raise ValueError('origin must be 0 or 1')
    return image_id - origin


def to_input_norm(bbox_xywh, width, height):
    """Native xywh pixels -> [x1, y1, x2, y2] on the model's 0-1000 grid, clipped to the image."""
    x, y, w, h = bbox_xywh
    clip = lambda v: min(max(v, 0.0), float(SCALE))
    return [clip(x / width * SCALE), clip(y / height * SCALE), clip((x + w) / width * SCALE), clip((y + h) / height * SCALE)]


def frame_boxes(coco, width, height, origin):
    """{frame_index: [[x1,y1,x2,y2] on 0-1000]} for every annotated frame."""
    boxes = {}
    for a in coco['annotations']:
        boxes.setdefault(frame_index(a['image_id'], origin), []).append(to_input_norm(a['bbox'], width, height))
    return boxes


def frame_statuses(category, frames, coco, origin):
    """Per decoded frame: positive (has a box), listed_no_box (image entry, no box),
    negative_category (No_Gun video) or unknown (no annotation information)."""
    if category == 'No_Gun':
        return ['negative_category'] * frames
    if coco is None:
        return ['unknown'] * frames
    listed = {frame_index(im['id'], origin) for im in coco['images']}
    boxed = {frame_index(a['image_id'], origin) for a in coco['annotations']}
    return ['positive' if i in boxed else 'listed_no_box' if i in listed else 'unknown' for i in range(frames)]


def render_overlay(image, boxes, colors=None, width=3):
    """Copy of a PIL image with 0-1000 boxes drawn in its own pixel space (audit and figures only)."""
    from PIL import ImageDraw
    out = image.convert('RGB').copy()
    draw = ImageDraw.Draw(out)
    for n, (x1, y1, x2, y2) in enumerate(boxes):
        color = (colors or ['red'])[n % len(colors or ['red'])]
        draw.rectangle([x1 / SCALE * out.width, y1 / SCALE * out.height, x2 / SCALE * out.width, y2 / SCALE * out.height],
                       outline=color, width=width)
    return out
