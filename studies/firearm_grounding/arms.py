"""Phase 2 training arms of the grounding pre-registration, on Training 1's verbalized-learning loop.

T1  Training 1 unchanged: VERA's optimizer prompt, selection by validation category accuracy.
G   the optimizer prompt also lists, per video, the attention lift on the annotated firearm in each
    boxed frame (training boxes only) with an overall low/typical/high level, and one fixed sentence.
    Selection: among validated steps within 5 points of the arm's best validation category accuracy,
    the highest pooled validation lift, then category accuracy, then the earliest step.
P   placebo: the same block and sentence, each frame lift drawn (seeded per step) from Phase 1's
    training-split frame lifts; the level is computed from the drawn values. Selection as G, on true
    validation lifts.
Attention for the block is read from the learner's own answer (re-tokenized) for the same images.
"""
import json
from pathlib import Path
import numpy as np
from vera_runner.ledger import read_jsonl
from vera_runner.reliance import coverage, frame_lifts, TOKENS_PER_FRAME
from vera_runner.verbalized import Run, learner_prompt, sample_indices
from .measure import boxed_frames

SENTENCE = ('Where there is a firearm, prefer questions that lead the model to look at the firearm itself, '
            'without lowering the accuracy of the category and action.')
BLOCK_HEADER = '** Where the model looked (attention on the annotated firearm per frame, 1.0 = chance): **'


def grounded_template(t1_template):
    """Training 1's optimizer prompt with the grounding section after the targets."""
    anchor = '** The targets: **\n[$GroundTruth]\n'
    if t1_template.count(anchor) != 1:
        raise ValueError('optimizer template must contain the targets block once')
    return t1_template.replace(anchor, anchor + BLOCK_HEADER + '\n[$Grounding]\n' + SENTENCE + '\n', 1)


def level(value, q25, q75):
    return 'low' if value < q25 else ('high' if value > q75 else 'typical')


class GroundedRun(Run):
    def __init__(self, spec, run_dir, data_dir, frames_dir, model, book, scorer, boxes, arm, phase1):
        super().__init__(spec, run_dir, data_dir, frames_dir, model, book)
        if arm not in ('T1', 'G', 'P'):
            raise ValueError(arm)
        self.scorer, self.boxes, self.arm = scorer, boxes, arm
        self.pool = np.array(phase1['train_frame_lifts'], dtype=float)
        self.q25, self.q75 = phase1['train_video_mean_frame_lift_quartiles']
        if arm != 'T1':
            self.optimizer_template = grounded_template(self.optimizer_template)

    def boxes_of(self, row):
        return boxed_frames(row, sample_indices(row['frames_decoded'], self.k), self.boxes)

    def attention(self, row, images, question, answer):
        """Per-frame attention of the decision position for this answer, or None without a Category field."""
        query, _ = self.scorer.query_ids(question, len(images))
        ids = self.scorer.tok(answer, add_special_tokens=False).input_ids
        s = self.scorer.score(images, query[0].tolist(), ids, attention=True)
        return None if s is None else {k: np.array(v) for k, v in s['frame_attention'].items()}

    def optimizer_template_for(self, rows, videos, texts, prompt, step):
        if self.arm == 'T1':
            return self.optimizer_template, {}
        rng = np.random.default_rng([self.spec['training']['seed'], step])
        lines, logged = [], []
        for i, (row, images, text) in enumerate(zip(rows, videos, texts), 1):
            boxes = self.boxes_of(row)
            if not boxes:
                lines.append(f'Video {i}: no firearm in the frames.')
                logged.append({'video': row['id'], 'frames': {}})
                continue
            if self.arm == 'G':
                att = self.attention(row, images, prompt, text)
                values = frame_lifts(att, boxes) if att is not None else None
            else:
                values = {k: float(rng.choice(self.pool)) for k in boxes}
            if not values:
                lines.append(f'Video {i} (target {row["category"]}): no decision to measure (the answer had no Category field).')
                logged.append({'video': row['id'], 'frames': None})
                continue
            mean = float(np.mean(list(values.values())))
            parts = ', '.join(f'Frame{k + 1} {v:.1f}' for k, v in sorted(values.items()))
            lines.append(f'Video {i} (target {row["category"]}): {parts}; overall {level(mean, self.q25, self.q75)}.')
            logged.append({'video': row['id'], 'frames': {str(k): v for k, v in values.items()}, 'mean': mean})
        template = self.optimizer_template.replace('[$Grounding]', '\n'.join(lines))
        return template, {'grounding_block': '\n'.join(lines), 'grounding_values': logged, 'arm': self.arm}

    def evaluate(self, rows, questions, name):
        result = super().evaluate(rows, questions, name)
        if self.arm == 'T1' or name in ('seed-test', 'test'):
            return result
        question = learner_prompt(self.learner_template, questions, self.k)
        by_id = {r['id']: r for r in rows}
        inside = chance = 0.0
        for p in result['predictions']:
            row = by_id[p['video']]
            boxes = self.boxes_of(row)
            if not boxes:
                continue
            images, _ = self.images(row)
            att = self.attention(row, images, question, p['response'])
            if att is None:
                p['lift'] = None
                continue
            vi = vc = 0.0
            for k, bs in boxes.items():
                cov = np.clip(sum(coverage(b) for b in bs), 0, 1).reshape(-1)
                vi += float(np.dot(att[k], cov))
                vc += float(cov.sum() / TOKENS_PER_FRAME)
            p['lift'] = vi / vc if vc else None
            inside, chance = inside + vi, chance + vc
        result['metrics']['pooled_lift'] = inside / chance if chance else None
        result['metrics']['videos_with_lift'] = sum(1 for p in result['predictions'] if p.get('lift') is not None)
        return result

    def select(self):
        vals = [v for v in read_jsonl(self.dir / 'validations.jsonl')[0] if not v['reference_only']]
        if not vals:
            raise ValueError('no validated checkpoint')
        if self.arm == 'T1':
            return super().select()
        best = max(v['metrics']['category_accuracy'] for v in vals)
        eligible = [v for v in vals if v['metrics']['category_accuracy'] >= best - 0.05]
        key = lambda v: (v['metrics'].get('pooled_lift') or 0.0, v['metrics']['category_accuracy'], -v['step'])
        return max(eligible, key=key)


def phase1_summary(records):
    """Frame-lift pool and video quartiles from Phase 1's training-split records (seed questions)."""
    train = [r for r in records if r.get('split') == 'train' and r.get('frame_lifts')]
    frames = [v for r in train for v in r['frame_lifts'].values() if v is not None]
    means = [float(np.mean([v for v in r['frame_lifts'].values() if v is not None])) for r in train
             if any(v is not None for v in r['frame_lifts'].values())]
    if not frames or not means:
        raise ValueError('Phase 1 has no training-split frame lifts')
    return {'train_frame_lifts': frames, 'train_video_mean_frame_lift_quartiles': [float(np.quantile(means, 0.25)), float(np.quantile(means, 0.75))],
            'train_videos': len(means)}
