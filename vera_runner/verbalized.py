"""VERA's verbalized learning (training.py), re-implemented for multi-field answers and resumable runs.

Faithful to the released training.py:
  - learner and optimizer are the same frozen InternVL2; no weight is updated;
  - a batch of 2 videos, 8 frames each at np.arange(0, n, n/8) over the whole video;
  - learner: batch_chat with num_patches_list=[8, 8] and the learner prompt, the current questions
    inserted before "Based on the analysis above";
  - optimizer: chat with both videos' 16 frames, VERA's optimizer prompt assembly (the frame block
    inserted before "** Model Descriptions: **", the questions before "Based on the analysis
    above") and the batch's predictions and targets;
  - new questions = the text after "New Prompt Questions" (+2 characters), cut to five lines when
    longer, kept unchanged when the phrase is missing;
  - greedy, one beam, 1,024 new tokens; validation every 100 training batches within an epoch.
Changed for this task:
  - predictions and targets are [category, action group, action code] per video, parsed strictly
    (an invalid field is the string 'invalid', never a default);
  - the shuffled order is seeded;
  - the seed questions are also validated before training (step 0, reference only);
  - every step and validation is logged and the run resumes from its last complete step.
"""
import hashlib
import json
import os
import re
import time
import traceback
from pathlib import Path
import numpy as np
from .ledger import JsonlAppender, now, read_jsonl

LEARNER_FLAG = 'Based on the analysis above'
OPTIMIZER_INPUT_FLAG = '** Model Descriptions: **'
FIELDS = ('category', 'group', 'code')
LABELS = {'category': r'category', 'group': r'action\s+group', 'code': r'action\s+code'}


def sample_indices(n, k=8):
    """VERA's training/validation sampling (datasets/video_instruction_dataset_VERA.py)."""
    return np.arange(0, n, n / k).astype(int).tolist()


def frame_prefix(k):
    return ''.join(f'Frame{i + 1}: <image>\n' for i in range(k))


def learner_prompt(template, questions, k=8):
    question = template.replace('$Data', frame_prefix(k))
    i = question.find(LEARNER_FLAG)
    return question[:i] + questions + question[i:] if i != -1 else question


def optimizer_prompt(template, questions, predictions, targets, k=8):
    text = template.replace('[$Prediction]', str(predictions)).replace('[$GroundTruth]', str(targets))
    i = text.find(OPTIMIZER_INPUT_FLAG)
    block = '[' + ('[' + frame_prefix(k) + '] ') * len(predictions) + ']\n'
    text = text[:i] + block + text[i - 1:]  # training.py keeps the character before the flag twice
    j = text.find(LEARNER_FLAG)
    return text[:j] + questions + text[j:]


def new_questions(response, current):
    """training.py lines 172-178, verbatim in effect."""
    at = response.find('New Prompt Questions')
    if at == -1:
        return current, 'kept: no "New Prompt Questions" in the answer'
    qs = response[at + len('New Prompt Questions') + 2:]
    if len(qs.split('\n')) > 5:
        qs = '\n'.join(qs.split('\n')[:5]) + '\n'
    return qs, 'updated'


def normalize(field, value, book):
    v = value.strip().strip('`[]()"\'*.,; \t').strip()
    if field == 'category':
        key = re.sub(r'[\s-]+', '_', v.lower())
        return {c.lower(): c for c in book['categories']}.get(key)
    if field == 'group':
        key = re.sub(r'[\s-]+', '_', v.lower())
        return key if key in book['groups'] else None
    key = v.upper().replace(' ', '')
    return key if key in book['codes'] else None


def parse_answer(text, book):
    """{'category', 'group', 'code'}: the last occurrence of each field, value up to the next field."""
    spans = []
    for field, label in LABELS.items():
        hits = list(re.finditer(label + r'\s*:', text, re.I))
        if hits:
            spans.append((hits[-1].start(), hits[-1].end(), field))
    spans.sort()
    out = {f: None for f in FIELDS}
    for n, (_, end, field) in enumerate(spans):
        stop = spans[n + 1][0] if n + 1 < len(spans) else len(text)
        out[field] = normalize(field, text[end:stop].replace('```', ' '), book)
    return out


def as_feedback(parsed):
    return [parsed[f] or 'invalid' for f in FIELDS]


def target_of(row):
    return [row['category'], row['action_group'], row['action']]


def metrics(rows):
    """rows: [{'target': [...], 'pred': {...}}]. Accuracy per field; invalid counts as wrong."""
    n = len(rows)
    out = {'videos': n}
    for k, f in enumerate(FIELDS):
        out[f + '_accuracy'] = sum(r['pred'][f] == r['target'][k] for r in rows) / n if n else None
        out[f + '_invalid'] = sum(r['pred'][f] is None for r in rows)
    return out


def extract_frames(video_path, indices, directory):
    """Decode the whole video with OpenCV (as data.preprocess) and keep only the sampled frames as default JPEGs."""
    import cv2
    directory.mkdir(parents=True, exist_ok=True)
    wanted, cap, n = set(indices), cv2.VideoCapture(str(video_path)), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n in wanted and not cv2.imwrite(str(directory / f'{n:06d}.jpg'), frame):
            raise RuntimeError('JPEG encoding failed')
        n += 1
    cap.release()
    return n


def load_images(directory, indices, size=448):
    from PIL import Image
    images = []
    for i in indices:
        with Image.open(directory / f'{i:06d}.jpg') as im:
            images.append(im.convert('RGB').resize((size, size)))  # the runner's (and VERA's) resize
    return images


class InternVL2Calls:
    """The two calls training.py makes, on the runner's loaded InternVL2 (weights verified at load)."""
    def __init__(self, backend, generation):
        self.b, self.generation = backend, dict(generation)

    def _tokens(self, text):
        return len(self.b.tokenizer(text).input_ids)

    def learn(self, videos, question):
        pixels = self.b.pixels([im for v in videos for im in v])
        with self.b.torch.inference_mode():
            texts = self.b.model.batch_chat(self.b.tokenizer, pixels, num_patches_list=[len(v) for v in videos],
                                            questions=[question] * len(videos), generation_config=dict(self.generation))
        return texts, [self._tokens(t) for t in texts]

    def optimize(self, videos, prompt):
        pixels = self.b.pixels([im for v in videos for im in v])
        with self.b.torch.inference_mode():
            text = self.b.model.chat(self.b.tokenizer, pixels, prompt, dict(self.generation))
        return text, self._tokens(text)


def digest(value):
    return hashlib.sha256((value if isinstance(value, str) else json.dumps(value, sort_keys=True)).encode('utf-8')).hexdigest()


class Run:
    def __init__(self, spec, run_dir, data_dir, frames_dir, model, book):
        self.spec, self.dir, self.data, self.frames, self.model, self.book = spec, Path(run_dir), Path(data_dir), Path(frames_dir), model, book
        self.learner_template = spec['prompts']['learner_instruct']
        self.optimizer_template = spec['prompts']['optimizer_instruct']
        self.k, self.bs = spec['training']['frames_per_video'], spec['training']['batch_size']
        self.split = {s: json.loads((self.data / f'{s}.json').read_text()) for s in ('train', 'val', 'test')}

    def images(self, row):
        indices = sample_indices(row['frames_decoded'], self.k)
        d = self.frames / row['id']
        if not all((d / f'{i:06d}.jpg').exists() for i in indices):
            n = extract_frames(row['staged_path'], indices, d)
            if n != row['frames_decoded']:
                raise ValueError(f"{row['id']}: decoded {n} frames, the inventory says {row['frames_decoded']}")
        return load_images(d, indices), indices

    def evaluate(self, rows, questions, name):
        """Learner on rows in batches of 2; per-video predictions and metrics. A smoke run may limit the rows."""
        limit = self.spec['training'].get('eval_limit')
        rows = rows[:limit] if limit else rows
        out = []
        prompt = learner_prompt(self.learner_template, questions, self.k)
        for b in range(0, len(rows), self.bs):
            batch = rows[b:b + self.bs]
            loaded = [self.images(r) for r in batch]
            begin = time.monotonic()
            texts, tokens = self.model.learn([im for im, _ in loaded], prompt)
            seconds = time.monotonic() - begin
            for r, (_, idx), text, tok in zip(batch, loaded, texts, tokens):
                out.append({'video': r['id'], 'category': r['category'], 'camera': r['camera'], 'recording_group': r['recording_group'],
                            'frame_indices': idx, 'target': target_of(r), 'response': text, 'pred': parse_answer(text, self.book),
                            'generated_tokens': tok, 'batch_seconds': seconds})
        return {'name': name, 'questions': questions, 'questions_sha256': digest(questions), 'prompt_sha256': digest(prompt),
                'metrics': metrics(out), 'predictions': out}

    def order(self, epoch):
        ids = [r['id'] for r in self.split['train']]
        return [ids[i] for i in np.random.default_rng([self.spec['training']['seed'], epoch]).permutation(len(ids))]

    def train(self, session, stop_after=None):
        steps_path, vals_path = self.dir / 'steps.jsonl', self.dir / 'validations.jsonl'
        steps = [s for s in read_jsonl(steps_path)[0] if s['status'] == 'complete']
        vals = {v['step'] for v in read_jsonl(vals_path)[0]}
        step_log, val_log = JsonlAppender(steps_path), JsonlAppender(vals_path)
        questions = steps[-1]['questions_after'] if steps else self.spec['training']['seed_questions']
        done = len(steps)
        by_id = {r['id']: r for r in self.split['train']}
        try:
            if 0 not in vals:
                val_log.write({'step': 0, 'epoch': 0, 'session': session, 'reference_only': True,
                               **self.evaluate(self.split['val'], questions, 'seed'), 'finished_at_utc': now()})
            step, ran = 0, 0
            for epoch in range(self.spec['training']['max_epochs']):
                order = self.order(epoch)
                batches = [order[i:i + self.bs] for i in range(0, len(order), self.bs)]
                due = lambda b: (b + 1) % self.spec['training']['val_check_interval'] == 0
                for b_idx, ids in enumerate(batches):
                    step += 1
                    if step <= done:
                        if due(b_idx) and step not in vals:  # the process ended between a step and its validation
                            val_log.write({'step': step, 'epoch': epoch, 'session': session, 'reference_only': False,
                                           **self.evaluate(self.split['val'], steps[step - 1]['questions_after'], f'step{step}'),
                                           'finished_at_utc': now()})
                            vals.add(step)
                        continue
                    if stop_after is not None and ran >= stop_after:
                        return 'stopped'
                    rows = [by_id[i] for i in ids]
                    loaded = [self.images(r) for r in rows]
                    videos = [im for im, _ in loaded]
                    record = {'step': step, 'epoch': epoch, 'batch_in_epoch': b_idx, 'session': session, 'videos': ids,
                              'frame_indices': [idx for _, idx in loaded], 'questions_before': questions, 'started_at': now()}
                    try:
                        prompt = learner_prompt(self.learner_template, questions, self.k)
                        t0 = time.monotonic()
                        texts, tokens = self.model.learn(videos, prompt)
                        t1 = time.monotonic()
                        preds = [as_feedback(parse_answer(t, self.book)) for t in texts]
                        targets = [target_of(r) for r in rows]
                        opt_prompt = optimizer_prompt(self.optimizer_template, questions, preds, targets, self.k)
                        opt_text, opt_tokens = self.model.optimize(videos, opt_prompt)
                        t2 = time.monotonic()
                    except Exception:
                        step_log.write({**record, 'status': 'failed', 'error': traceback.format_exc(), 'finished_at': now()})
                        raise
                    updated, how = new_questions(opt_text, questions)
                    step_log.write({**record, 'status': 'complete', 'learner_prompt_sha256': digest(prompt), 'learner_responses': texts,
                                    'learner_tokens': tokens, 'predictions': preds, 'targets': targets,
                                    'optimizer_prompt': opt_prompt, 'optimizer_response': opt_text, 'optimizer_tokens': opt_tokens,
                                    'questions_after': updated, 'update': how, 'learner_seconds': t1 - t0,
                                    'optimizer_seconds': t2 - t1, 'finished_at': now()})
                    questions, ran = updated, ran + 1
                    if due(b_idx) and step not in vals:
                        val_log.write({'step': step, 'epoch': epoch, 'session': session, 'reference_only': False,
                                       **self.evaluate(self.split['val'], questions, f'step{step}'), 'finished_at_utc': now()})
                        vals.add(step)
            return 'complete'
        finally:
            step_log.close()
            val_log.close()

    def select(self):
        vals = [v for v in read_jsonl(self.dir / 'validations.jsonl')[0] if not v['reference_only']]
        if not vals:
            raise ValueError('no validated checkpoint')
        key = lambda v: (v['metrics']['category_accuracy'], v['metrics']['group_accuracy'], -v['step'])
        return max(vals, key=key)

    def test(self, session):
        best = self.select()
        seed = self.spec['training']['seed_questions']
        out = JsonlAppender(self.dir / 'test.jsonl')
        done = {t['name'] for t in read_jsonl(self.dir / 'test.jsonl')[0]}
        try:
            for name, qs, source in (('seed', seed, 'seed questions'), ('learned', best['questions'], f"validation step {best['step']}")):
                if name not in done:
                    out.write({**self.evaluate(self.split['test'], qs, name), 'source': source, 'session': session,
                               'selected_step': best['step'] if name == 'learned' else 0, 'finished_at_utc': now()})
        finally:
            out.close()
        return best
