"""Annotation audit for the existing VISILAB boxes: validation, coverage and visual-check montages.

It validates every label file, summarises box coverage per video, and draws montages for the
checks recorded in studies/firearm/AUDIT.md:
  origin/   the box of image_id k on decoded frames k-2, k-1 and k, at the fastest box motion
            and at the first and last boxed frame (an origin-1 label lines up with frame k-1)
  unboxed/  frames that have an image entry but no box (is a firearm visible?)
  no_gun/   frames from No_Gun videos (is any firearm visible?)
Videos and frames are drawn with a fixed seed. The montages are evidence for a person to look at;
this module makes no decision.
"""
import json
import random
from pathlib import Path
from vera_runner.firearm_annotations import load_coco, render_overlay, to_input_norm, validate_coco

SEED = 20261025


def read_frames(path, wanted):
    """{index: PIL RGB} for the wanted zero-based decoded frame indices (sequential OpenCV decode)."""
    import cv2
    from PIL import Image
    wanted, out, cap, i = set(wanted), {}, cv2.VideoCapture(str(path)), 0
    while wanted - set(out):
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            out[i] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    return out


def grid(images, labels, columns, cell=(480, 360)):
    from PIL import Image, ImageDraw
    rows = -(-len(images) // columns)
    sheet = Image.new('RGB', (columns * cell[0], rows * (cell[1] + 18)), 'white')
    draw = ImageDraw.Draw(sheet)
    for n, (im, text) in enumerate(zip(images, labels)):
        x, y = n % columns * cell[0], n // columns * (cell[1] + 18)
        sheet.paste(im.resize(cell), (x, y + 18))
        draw.text((x + 4, y + 3), text, fill='black')
    return sheet


def crop_around(image, box, margin=2.5):
    """Magnified region around a 0-1000 box so a one-frame offset is visible."""
    x1, y1, x2, y2 = [v / 1000 * s for v, s in zip(box, (image.width, image.height) * 2)]
    cx, cy, half = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * margin / 2 + 8
    return image.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))


def coverage(row, coco):
    boxed = sorted(a['image_id'] for a in coco['annotations'])
    gaps = sum(1 for a, b in zip(boxed, boxed[1:]) if b != a + 1)
    areas = [a['bbox'][2] * a['bbox'][3] / (row['width'] * row['height']) for a in coco['annotations']]
    return {'id': row['id'], 'category': row['category'], 'camera': row['camera'], 'frames': row['frames_decoded'],
            'images': len(coco['images']), 'boxed': len(boxed), 'listed_no_box': len(coco['images']) - len(boxed),
            'first_boxed_image_id': boxed[0] if boxed else None, 'last_boxed_image_id': boxed[-1] if boxed else None,
            'boxed_runs': gaps + 1 if boxed else 0, 'area_fractions': areas}


def run_audit(inventory_path, original_root, out_dir, origin_videos=12, unboxed_videos=16, no_gun_videos=12):
    rows = json.loads(Path(inventory_path).read_text())
    out = Path(out_dir)
    for sub in ('origin', 'unboxed', 'no_gun'):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    validation, cover, labelled = {}, [], []
    for row in rows:
        if not row['label_file']:
            continue
        coco = load_coco(Path(original_root) / Path(row['source_path']).parent / row['label_file'])
        validation[row['id']] = {'label_file': row['label_file'],
                                 'problems': validate_coco(coco, row['frames_decoded'], row['width'], row['height'], row['category'])}
        cover.append(coverage(row, coco))
        labelled.append((row, coco))
    (out / 'validation.json').write_text(json.dumps(validation, indent=1) + '\n')
    (out / 'coverage.json').write_text(json.dumps(cover, indent=1) + '\n')

    strata = {}
    for row, coco in labelled:
        strata.setdefault((row['category'], row['camera']), []).append((row, coco))
    chosen = []
    for key in sorted(strata):
        chosen += rng.sample(strata[key], min(len(strata[key]), origin_videos // len(strata)))
    antiguo = [(r, c) for r, c in labelled if r['label_file'] == 'label_antiguo.json']
    chosen += [x for x in antiguo if x not in chosen]
    picks = []
    for row, coco in chosen:
        by_id = {a['image_id']: to_input_norm(a['bbox'], row['width'], row['height']) for a in coco['annotations']}
        ids = sorted(by_id)
        centre = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        motion = sorted(((abs(centre(by_id[k])[0] - centre(by_id[k - 1])[0]) + abs(centre(by_id[k])[1] - centre(by_id[k - 1])[1]), k)
                         for k in ids if k - 1 in by_id), reverse=True)
        keys = [('fast', motion[0][1])] if motion else []
        keys += [('first', ids[0]), ('last', ids[-1])] if ids else []
        frames = read_frames(Path(original_root) / row['source_path'], {k + d for _, k in keys for d in (-2, -1, 0) if 0 <= k + d < row['frames_decoded']})
        for kind, k in keys:
            cells, labels = [], []
            for d in (-2, -1, 0):
                if k + d in frames:
                    cells.append(crop_around(render_overlay(frames[k + d], [by_id[k]], width=2), by_id[k]))
                    labels.append(f'image_id {k} box on frame {k + d} (origin {k - (k + d)})')
            name = f"{row['id']}_{kind}_id{k}.png"
            grid(cells, labels, 3, (360, 360)).save(out / 'origin' / name)
            picks.append({'video': row['id'], 'kind': kind, 'image_id': k, 'file': f'origin/{name}'})

    unboxed = []
    pool = [(r, c) for r, c in labelled]
    for row, coco in rng.sample(pool, min(unboxed_videos, len(pool))):
        boxed = {a['image_id'] for a in coco['annotations']}
        free = [im['id'] for im in coco['images'] if im['id'] not in boxed]
        ids = sorted(rng.sample(free, min(3, len(free))))
        frames = read_frames(Path(original_root) / row['source_path'], {i - 1 for i in ids})
        for i in ids:
            unboxed.append((row['id'], i, frames[i - 1]))
    for n in range(0, len(unboxed), 6):
        part = unboxed[n:n + 6]
        grid([im for _, _, im in part], [f'{v} image_id {i} (no box)' for v, i, _ in part], 3).save(out / 'unboxed' / f'sheet_{n // 6:02d}.png')

    negatives = [r for r in rows if r['category'] == 'No_Gun']
    neg = []
    for row in rng.sample(negatives, min(no_gun_videos, len(negatives))):
        ids = sorted(rng.sample(range(row['frames_decoded']), 2))
        frames = read_frames(Path(original_root) / row['source_path'], ids)
        neg += [(row['id'], i, frames[i]) for i in ids]
    for n in range(0, len(neg), 6):
        part = neg[n:n + 6]
        grid([im for _, _, im in part], [f'{v} frame {i}' for v, i, _ in part], 3).save(out / 'no_gun' / f'sheet_{n // 6:02d}.png')
    manifest = {'seed': SEED, 'origin_checks': picks, 'unboxed_frames': [[v, i] for v, i, _ in unboxed],
                'no_gun_frames': [[v, i] for v, i, _ in neg]}
    (out / 'montages.json').write_text(json.dumps(manifest, indent=1) + '\n')
    return validation, cover, manifest
