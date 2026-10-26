"""Per-video grounding measurement (M1-M4 of the pre-registration) and the resumable measurement loop.

For one video and one question set: the greedy answer, its parsed category/group/code, the decision
probabilities, the attention lift with the true boxes and with shuffled boxes (another grounding-evaluable
video of the same camera, frame by frame), layer-group and sink-excluded lifts, and, for grounding-evaluable
videos, occlusion reliance: log p_true with control masks minus with box masks, each after regenerating the
answer from the masked frames. Frames whose box has no equal-sized disjoint control position are left out
of both masks and recorded.
"""
import json
import math
import random
import re
import time
from pathlib import Path
import numpy as np
from vera_runner.ledger import JsonlAppender, now, read_jsonl
from vera_runner.reliance import control_box, frame_lifts, lift, mask_image
from vera_runner.verbalized import learner_prompt, load_images, parse_answer, sample_indices, extract_frames

LAYER_GROUPS = {'early': range(0, 11), 'middle': range(11, 22), 'late': range(22, 32)}
WEAPON = re.compile(r'\b(gun|guns|pistol|handgun|rifle|firearm|firearms|weapon|weapons|machine gun|machine-gun)\b', re.I)
SEED = 20261026


def explanation(text):
    cut = re.search(r'category\s*:', text, re.I)
    return text[:cut.start()] if cut else text


def boxed_frames(row, indices, boxes):
    """{slot k: [boxes]} for sampled frames whose audit status is positive."""
    b = boxes.get(row['id'])
    if not b:
        return {}
    return {k: b['boxes'][str(i)] for k, i in enumerate(indices) if b['statuses'][i] == 'positive' and str(i) in b['boxes']}


def shuffled_donors(rows, frames_of):
    """{video: donor video}: a permutation within camera among grounding-evaluable videos, seeded, no fixed points."""
    out = {}
    for cam in sorted({r['camera'] for r in rows}):
        pool = sorted(r['id'] for r in rows if r['camera'] == cam and frames_of[r['id']])
        if len(pool) < 2:
            continue
        shifted = pool[:]
        random.Random(f'{SEED}:{cam}').shuffle(shifted)
        out.update({v: shifted[(i + 1) % len(shifted)] for i, v in enumerate(shifted)})
    return out


def sink_excluded(att):
    a = np.array(att, dtype=float)
    a[np.argmax(a)] = 0.0
    return a / a.sum()


def measure_video(scorer, row, images, question, boxes_true, boxes_shuffled, occlusion=True):
    t0 = time.monotonic()
    gen = scorer.generate(images, question)
    rec = {'video': row['id'], 'category': row['category'], 'camera': row['camera'], 'recording_group': row['recording_group'],
           'target': [row['category'], row['action_group'], row['action']], 'answer': gen['text'],
           'pred': parse_answer(gen['text'], scorer.book), 'generated_tokens': len(gen['ids']),
           'weapon_mentioned': bool(WEAPON.search(explanation(gen['text']))), 'boxed_slots': sorted(boxes_true)}
    s = scorer.score(images, gen['query_ids'], gen['ids'], attention=True)
    if s is None:
        rec.update(decision=None, seconds=time.monotonic() - t0)
        return rec
    rec['decision'] = {'p': s['category_p'], 'logp': s['category_logp'], 'p_true': s['category_p'][row['category']],
                       'frame_mass': s['frame_mass']}
    if boxes_true:
        att = {k: np.array(v) for k, v in s['frame_attention'].items()}
        rec['lift'] = lift(att, boxes_true)
        rec['frame_lifts'] = frame_lifts(att, boxes_true)
        rec['lift_shuffled'] = lift(att, boxes_shuffled) if boxes_shuffled else None
        rec['lift_sink_excluded'] = lift({k: sink_excluded(v) for k, v in att.items()}, boxes_true)
        rec['lift_layers'] = {}
        for name, layers in LAYER_GROUPS.items():
            mean = np.mean([s['layer_rows'][i] for i in layers], axis=0)
            rec['lift_layers'][name] = lift({k: mean[k] / mean[k].sum() for k in boxes_true}, boxes_true)
        rec['lift_per_layer'] = [lift({k: s['layer_rows'][i][k] / s['layer_rows'][i][k].sum() for k in boxes_true}, boxes_true)
                                 for i in sorted(s['layer_rows'])]
    if boxes_true and occlusion:
        controls = {k: [control_box(b) for b in bs] for k, bs in boxes_true.items()}
        usable = {k for k, cs in controls.items() if all(c is not None for c in cs)}
        rec['occlusion_frames'] = sorted(usable)
        rec['occlusion_dropped_frames'] = sorted(set(boxes_true) - usable)
        if usable:
            results = {}
            for name, boxes in (('box', {k: boxes_true[k] for k in usable}), ('control', {k: controls[k] for k in usable})):
                masked = [mask_image(im, boxes[k]) if k in boxes else im for k, im in enumerate(images)]
                g = scorer.generate(masked, question)
                sm = scorer.score(masked, g['query_ids'], g['ids'], attention=False)
                results[name] = {'answer': g['text'], 'pred': parse_answer(g['text'], scorer.book),
                                 'p_true': sm['category_p'][row['category']] if sm else None,
                                 'logp_true': math.log(max(sm['category_p'][row['category']], 1e-300)) if sm else None}
            rec['occlusion'] = results
            b, c = results['box']['logp_true'], results['control']['logp_true']
            rec['reliance'] = (c - b) if b is not None and c is not None else None
    rec['seconds'] = time.monotonic() - t0
    return rec


class Measurer:
    def __init__(self, scorer, data_dir, frames_dir, boxes_dir, learner_template, k=8):
        self.scorer, self.data, self.frames = scorer, Path(data_dir), Path(frames_dir)
        self.learner_template, self.k = learner_template, k
        self.rows = {s: json.loads((self.data / f'{s}.json').read_text()) for s in ('train', 'val', 'test')}
        self.boxes = {}
        for s in ('train', 'val', 'test'):
            self.boxes.update(json.loads((Path(boxes_dir) / f'boxes_{s}.json').read_text()))

    def images(self, row):
        indices = sample_indices(row['frames_decoded'], self.k)
        d = self.frames / row['id']
        if not all((d / f'{i:06d}.jpg').exists() for i in indices):
            n = extract_frames(row['staged_path'], indices, d)
            if n != row['frames_decoded']:
                raise ValueError(f"{row['id']}: decoded {n} frames, expected {row['frames_decoded']}")
        return load_images(d, indices), indices

    def run(self, splits, questions, out_path, occlusion=True, limit=None):
        rows = [r for s in splits for r in self.rows[s]]
        rows = rows[:limit] if limit else rows
        indices = {r['id']: sample_indices(r['frames_decoded'], self.k) for r in rows}
        frames_of = {r['id']: boxed_frames(r, indices[r['id']], self.boxes) for r in rows}
        donors = shuffled_donors(rows, frames_of)
        question = learner_prompt(self.learner_template, questions, self.k)
        done = {r['video'] for r in read_jsonl(out_path)[0]} if Path(out_path).exists() else set()
        out = JsonlAppender(out_path)
        try:
            for r in rows:
                if r['id'] in done:
                    continue
                images, idx = self.images(r)
                own = frames_of[r['id']]
                donor = donors.get(r['id'])
                shuffled = {k: frames_of[donor][k] for k in own if donor and k in frames_of[donor]} if own else {}
                rec = measure_video(self.scorer, r, images, question, own, shuffled, occlusion)
                rec.update(split=next(s for s in splits if r in self.rows[s]), frame_indices=idx, donor=donor, measured_at=now())
                out.write(rec)
        finally:
            out.close()
