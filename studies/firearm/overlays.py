"""Window sheets: the eight exact model inputs with G boxes, AG boxes and the existing boxes drawn.

Ground truth comes from the video's label file under the audited origin; G is red, AG blue,
ground truth green. Used for the pilot's coordinate check, P8 and T11.
"""
import json
from pathlib import Path
from vera_runner.firearm_annotations import frame_boxes, load_coco, render_overlay
from vera_runner.ledger import read_jsonl

COLORS = {'G': 'red', 'AG': 'blue', 'GT': 'lime'}


def ground_truth(row, original_root, origin):
    if not row.get('label_file'):
        return {}
    coco = load_coco(Path(original_root) / Path(row['source_path']).parent / row['label_file'])
    return frame_boxes(coco, row['width'], row['height'], origin)


def latest_complete(run_dir, condition):
    out = {}
    for r in read_jsonl(Path(run_dir) / 'records' / f'{condition}.jsonl')[0]:
        if r['status'] == 'complete':
            out[(r['video'], r['anchor'])] = r
    return out


def window_sheet(run_dir, row, anchor, gt, records, out_path):
    from PIL import Image, ImageDraw
    inputs = Path(run_dir) / 'videos' / row['id'] / 'inputs' / str(anchor)
    schedule = json.loads((Path(run_dir) / 'videos' / row['id'] / 'windows.json').read_text())
    frames = next(w['frame_indices'] for w in schedule if w['anchor'] == anchor)
    cells = []
    for slot in range(8):
        image = Image.open(inputs / f'{slot:02d}.png').convert('RGB')
        for condition in ('G', 'AG'):
            rec = records[condition].get((row['id'], anchor))
            if rec and 'grounding' in rec:
                image = render_overlay(image, rec['grounding']['slots'][slot]['boxes'], [COLORS[condition]], 2)
        image = render_overlay(image, gt.get(frames[slot], []), [COLORS['GT']], 2)
        cells.append((image, slot, frames[slot]))
    sheet = Image.new('RGB', (4 * 448, 2 * (448 + 16)), 'white')
    draw = ImageDraw.Draw(sheet)
    for n, (image, slot, frame) in enumerate(cells):
        x, y = n % 4 * 448, n // 4 * (448 + 16)
        sheet.paste(image, (x, y + 16))
        status = '/'.join(records[c][(row['id'], anchor)]['grounding']['slots'][slot]['status'][:5]
                          for c in ('G', 'AG') if (row['id'], anchor) in records[c] and 'grounding' in records[c][(row['id'], anchor)])
        draw.text((x + 4, y + 2), f'Frame{slot + 1} = frame {frame}  G/AG: {status}', fill='black')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def sheets(run_dir, inventory_path, original_root, origin, out_dir):
    rows = {r['id']: r for r in json.loads(Path(inventory_path).read_text())}
    records = {c: latest_complete(run_dir, c) for c in ('A', 'G', 'AG')}
    made = []
    for video, anchor in sorted(records['G']):
        gt = ground_truth(rows[video], original_root, origin)
        made.append(str(window_sheet(run_dir, rows[video], anchor, gt, records, Path(out_dir) / f'{video}_a{anchor}.png')))
    return made
