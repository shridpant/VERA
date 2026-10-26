"""E2: recognition test (package plan, Phase 1, Ada 5). Can the backbone name the category when it sees the gun?

The VERA learner prompt with Training 1's neutral seed questions, on the 302 non-test videos, under four inputs:
  packed  VERA's call: 8 frames as patches of the first <image> (batch_chat, num_patches_list=[8]), 448 px each;
  frames  each frame its own <image>, 448 px;
  tiles   each frame its own <image> with dynamic 448 tiles (max_num) and a thumbnail;
  crops   frames, plus two enlarged regions ("Zoom1", "Zoom2"): the two largest annotated boxes (x1.5, distinct
          frames when possible), or seeded random regions of a firearm-like size when the video has none
          (every No_Gun video), so the presence of zooms says nothing about the label. Oracle: it uses true boxes.
One run folder per arm (spec.json + records.jsonl), resumable, seeds per video.
"""
import hashlib
import json
from pathlib import Path
import traceback
from vera_runner.ledger import JsonlAppender, now, read_jsonl
from vera_runner.verbalized import LEARNER_FLAG, frame_prefix, parse_answer, sample_indices
from vera_runner.verbalized_variants import rng
from .e1 import code_commit

ZOOM_NOTE = 'Zoom1 and Zoom2 are enlarged regions of the frames.\n'


def prompt_for(template, questions, crops=False):
    data = frame_prefix(8) + ('Zoom1: <image>\nZoom2: <image>\n' + ZOOM_NOTE if crops else '')
    question = template.replace('$Data', data)
    i = question.find(LEARNER_FLAG)
    return question[:i] + questions + question[i:] if i != -1 else question


def videos(cfg):
    data = Path(cfg['data_dir'])
    rows = [r for s in ('train', 'val') for r in json.loads((data / f'{s}.json').read_text())]
    boxes = {k: v for s in ('train', 'val') for k, v in json.loads((data / f'boxes_{s}.json').read_text()).items()}
    return sorted(rows, key=lambda r: r['id']), boxes


def crop_boxes(row, boxes, seed, sizes):
    """Two crop regions on the 0-1000 grid with their source ('gt' or 'random')."""
    idx = sample_indices(row['frames_decoded'], 8)
    b = boxes.get(row['id'], {'statuses': [], 'boxes': {}})
    cands = sorted(((bx[2] - bx[0]) * (bx[3] - bx[1]), k, bx) for k, i in enumerate(idx) if str(i) in b['boxes']
                   for bx in b['boxes'][str(i)])[::-1]
    chosen, used = [], set()
    for _, k, bx in cands:  # largest boxes, distinct frames first
        if k not in used:
            chosen.append((k, bx, 'gt')), used.add(k)
        if len(chosen) == 2:
            break
    for _, k, bx in cands:
        if len(chosen) == 2:
            break
        if (k, bx, 'gt') not in chosen:
            chosen.append((k, bx, 'gt'))
    g = rng(seed, 'crops', row['id'])
    while len(chosen) < 2:
        w, h = sizes[g.integers(len(sizes))]
        x, y = g.uniform(0, 1000 - w), g.uniform(0, 1000 - h)
        chosen.append((int(g.integers(8)), [x, y, x + w, y + h], 'random'))
    out = []
    for k, (x1, y1, x2, y2), source in chosen:  # enlarge 1.5x around the centre, at least 64/1000 a side, clipped
        cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, max(64, 1.5 * (x2 - x1)), max(64, 1.5 * (y2 - y1))
        out.append({'slot': k, 'box': [max(0, cx - w / 2), max(0, cy - h / 2), min(1000, cx + w / 2), min(1000, cy + h / 2)],
                    'source': source})
    return out


def box_sizes(rows, boxes):
    return sorted({(bx[2] - bx[0], bx[3] - bx[1]) for r in rows for bs in boxes.get(r['id'], {}).get('boxes', {}).values() for bx in bs})


def native_frames(cfg, row):
    from PIL import Image
    out = []
    for i in sample_indices(row['frames_decoded'], 8):
        with Image.open(Path(cfg['frames_dir']) / row['id'] / f'{i:06d}.jpg') as im:
            out.append(im.convert('RGB'))
    return out


def inputs(arm, frames, crops):
    """images_per_slot for vlm.InternVL.chat, and whether the call is packed."""
    from vera_runner.vlm import dynamic_preprocess
    square = [im.resize((448, 448)) for im in frames]  # VERA's resize
    if arm['input'] == 'packed':
        return [square], True
    if arm['input'] == 'frames':
        return [[im] for im in square], False
    if arm['input'] == 'tiles':
        return [dynamic_preprocess(im, max_num=arm.get('max_num', 6)) for im in frames], False
    zooms = []
    for c in crops:
        im = frames[c['slot']]
        x1, y1, x2, y2 = c['box']
        zooms.append([im.crop((x1 * im.width / 1000, y1 * im.height / 1000, x2 * im.width / 1000, y2 * im.height / 1000)).resize((448, 448))])
    return [[im] for im in square] + zooms, False


def run_arm(cfg, name, arm, out_dir, root, model=None, limit=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    book = json.loads((Path(root) / cfg['codebook']).read_text())
    template = (Path(root) / cfg['learner_instruct_file']).read_text()
    questions = (Path(root) / cfg['seed_questions_file']).read_text()
    question = prompt_for(template, questions, crops=arm['input'] == 'crops')
    spec = {'experiment': 'package_e2', 'arm': name, **arm, 'model_path': cfg['models'][arm['model']], 'prompt': question,
            'seed': cfg['seed']}
    spec['spec_sha256'] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    if (out / 'spec.json').exists():
        if json.loads((out / 'spec.json').read_text())['spec_sha256'] != spec['spec_sha256']:
            raise ValueError(f'{name}: spec changed since the run started; use a new arm name')
    else:
        (out / 'spec.json').write_text(json.dumps({**spec, 'created_at': now(), 'code_commit': code_commit()}, indent=1) + '\n')
    rows, boxes = videos(cfg)
    sizes = box_sizes(rows, boxes)
    done = {r['video'] for r in read_jsonl(out / 'records.jsonl')[0] if r['status'] != 'failed'}
    todo = [r for r in rows if r['id'] not in done][:limit]
    if model is None:
        from vera_runner.vlm import InternVL
        model = InternVL(cfg['models'][arm['model']], thinking=arm['thinking'], flash=arm.get('flash', False))
    log, failed = JsonlAppender(out / 'records.jsonl'), 0
    try:
        for r in todo:
            seed = int(hashlib.sha256(f"{cfg['seed']}|{r['id']}".encode()).hexdigest()[:8], 16)
            crops = crop_boxes(r, boxes, cfg['seed'], sizes) if arm['input'] == 'crops' else []
            base = {'video': r['id'], 'category': r['category'], 'recording_group': r['recording_group'],
                    'target': [r['category'], r['action_group'], r['action']], 'seed': seed, 'crops': crops, 'started_at': now()}
            try:
                slots, packed = inputs(arm, native_frames(cfg, r), crops)
                result = model.chat(slots, question, arm['max_new_tokens'], seed=seed, packed=packed)
                pred = parse_answer(result['answer'], book) if result['answer'] else {'category': None, 'group': None, 'code': None}
                log.write({**base, 'status': 'complete', 'pred': pred, 'n_patches': sum(len(s) for s in slots), **result,
                           'finished_at': now()})
            except Exception:
                failed += 1
                log.write({**base, 'status': 'failed', 'error': traceback.format_exc(), 'finished_at': now()})
                if failed >= 20:
                    raise
    finally:
        log.close()
    return {'arm': name, 'videos': len(rows), 'done_before': len(done), 'ran': len(todo), 'failed': failed}
