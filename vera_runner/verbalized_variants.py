"""Two declared variants of VERA's verbalized learning (the faithful procedure is verbalized.Run).

Both keep the frozen InternVL2 as learner and optimizer, the learner prompt, the strict parser, the
validation set and the selection rule. Every random choice is derived from (seed, step) so a resumed
run makes the same choices. See studies/firearm_training/LITERATURE.md for the prior art.

FixedRun ('vera_fixed') changes VERA's loop where the faithful run's logs show it failing:
  - each step sees a class-balanced batch (training.videos_per_class per category) instead of 2 i.i.d.
    videos; the learner still answers in pairs with all frames, the optimizer sees every video of the
    batch with training.optimizer_frames evenly spaced frames each, so that its input stays inside the
    model's context (checked before every call);
  - the optimizer prompt lists the last training.history question sets with their check accuracy
    (OPRO's score history);
  - a new question set is kept only if its category accuracy, then action-group accuracy, on a fixed
    check set (training.check_per_class training videos per category, never in a batch) is not
    lower than the current set's (accept-if-not-worse);
  - questions must be answerable with yes/no or a short phrase; a set asking to describe, explain or
    list is rejected without a check.
BankRun ('bank', 'bank_random', 'bank_noscore') learns single yes/no questions:
  - a pass of the learner over the training set (refreshed every training.refresh_every rounds) gives
    the confusions: true category -> predicted category, and true group -> predicted group within a
    correct category;
  - each round samples a confused pair (with probability proportional to its count), shows the
    optimizer one video the learner put in the wrong class and one video of that other class, and asks
    for ONE yes/no question about something visible that differs between them;
  - the question alone is asked on a fixed number of other training videos of each class; it enters
    the bank when its yes/no answers separate the two classes with balanced accuracy at least
    training.keep_threshold (in either orientation);
  - the learner prompt uses the bank's training.top_k best questions, at most training.per_pair_cap per
    pair. Ablations: bank_random draws pairs and videos uniformly; bank_noscore keeps every valid
    question and uses the newest top_k.
"""
import json
import re
import time
import traceback
import numpy as np
from .ledger import JsonlAppender, now, read_jsonl
from .verbalized import (FIELDS, Run, as_feedback, digest, frame_prefix, learner_prompt, metrics, new_questions,
                         parse_answer, target_of)

CATS = ('Handgun', 'Machine_Gun', 'No_Gun')
FORBIDDEN = re.compile(r'\b(describe|explain|list)\b', re.I)


def rng(seed, *parts):
    return np.random.default_rng([seed, *[abs(hash_int(p)) for p in parts]])


def hash_int(value):
    return int(digest(str(value))[:12], 16)


def question_lines(text):
    return [l.strip() for l in text.split('\n') if l.strip() and not l.strip().startswith('```')]


def acceptable(questions):
    """None if the proposed set may be checked, else why it is rejected outright."""
    lines = question_lines(questions)
    if not lines:
        return 'empty'
    if len(lines) > 5:
        return 'more than five questions'
    if FORBIDDEN.search(questions):
        return 'asks to describe, explain or list'
    return None


def score_key(m):
    return (m['category_accuracy'], m['group_accuracy'])


class VariantBase(Run):
    def validate(self, log, step, questions, session, extra=None):
        log.write({'step': step, 'session': session, 'reference_only': step == 0, 'epoch': (extra or {}).get('epoch', 0),
                   **(extra or {}), **self.evaluate(self.split['val'], questions, 'seed' if step == 0 else f'step{step}'),
                   'finished_at_utc': now()})

    def predict_rows(self, rows, questions):
        """Learner on rows (pairs) without the eval limit; [(row, parsed prediction, response, tokens)]."""
        prompt = learner_prompt(self.learner_template, questions, self.k)
        out = []
        for b in range(0, len(rows), self.bs):
            batch = rows[b:b + self.bs]
            texts, tokens = self.model.learn([self.images(r)[0] for r in batch], prompt)
            out += [(r, parse_answer(t, self.book), t, k) for r, t, k in zip(batch, texts, tokens)]
        return out


class FixedRun(VariantBase):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        tr = self.spec['training']
        self.vpc, self.history_n = tr['videos_per_class'], tr['history']
        by_cat = {c: [r for r in self.split['train'] if r['category'] == c] for c in CATS}
        pick = rng(tr['seed'], 'check')
        self.check = [r for c in CATS for r in (by_cat[c][i] for i in sorted(pick.choice(len(by_cat[c]), tr['check_per_class'], replace=False)))]
        check_ids = {r['id'] for r in self.check}
        self.pool = {c: [r for r in by_cat[c] if r['id'] not in check_ids] for c in CATS}
        self.steps_per_epoch = min(len(v) for v in self.pool.values()) // self.vpc
        self.opt_frames = sorted({int(i) for i in np.linspace(0, self.k - 1, tr.get('optimizer_frames') or self.k).round()})

    def batch(self, epoch, j):
        rows = []
        for c in CATS:
            order = rng(self.spec['training']['seed'], 'epoch', epoch, c).permutation(len(self.pool[c]))
            rows += [self.pool[c][order[(j * self.vpc + i) % len(order)]] for i in range(self.vpc)]
        return rows

    def check_score(self, questions):
        out = self.predict_rows(self.check, questions)
        return metrics([{'target': target_of(r), 'pred': p} for r, p, _, _ in out]), [
            {'video': r['id'], 'pred': p, 'tokens': k} for r, p, _, k in out]

    def optimizer_text(self, questions, preds, targets, history):
        hist = '\n'.join(f'Questions:\n{h["questions"].strip()}\nCheck accuracy: category {h["category_accuracy"]:.0%}, '
                         f'action group {h["group_accuracy"]:.0%}' for h in history) or 'none yet'
        text = (self.optimizer_template.replace('[$Prediction]', str(preds)).replace('[$GroundTruth]', str(targets))
                .replace('[$History]', hist))
        block = '[' + ('[' + frame_prefix(len(self.opt_frames)) + '] ') * len(preds) + ']\n'
        i = text.find('** Model Descriptions: **')
        text = text[:i] + block + text[i - 1:]
        j = text.find('Based on the analysis above')
        return text[:j] + questions + text[j:]

    def train(self, session, stop_after=None):
        tr = self.spec['training']
        steps_path, vals_path = self.dir / 'steps.jsonl', self.dir / 'validations.jsonl'
        steps = [s for s in read_jsonl(steps_path)[0] if s['status'] == 'complete']
        vals = {v['step'] for v in read_jsonl(vals_path)[0]}
        step_log, val_log = JsonlAppender(steps_path), JsonlAppender(vals_path)
        try:
            if steps:
                questions, current = steps[-1]['questions_after'], steps[-1]['current_check']
                history = steps[-1]['history_after']
            else:
                questions = tr['seed_questions']
                current, _ = self.check_score(questions)
                history = [{'questions': questions, **{k: current[k] for k in ('category_accuracy', 'group_accuracy')}}]
            if 0 not in vals:
                self.validate(val_log, 0, tr['seed_questions'], session)
            done, ran, step = len(steps), 0, 0
            for epoch in range(tr['max_epochs']):
                for j in range(self.steps_per_epoch):
                    step += 1
                    if step <= done:
                        continue
                    if stop_after is not None and ran >= stop_after:
                        return 'stopped'
                    rows = self.batch(epoch, j)
                    record = {'step': step, 'epoch': epoch, 'batch_in_epoch': j, 'session': session, 'videos': [r['id'] for r in rows],
                              'questions_before': questions, 'started_at': now()}
                    try:
                        t0 = time.monotonic()
                        out = self.predict_rows(rows, questions)
                        preds = [as_feedback(p) for _, p, _, _ in out]
                        targets = [target_of(r) for r in rows]
                        t1 = time.monotonic()
                        prompt = self.optimizer_text(questions, preds, targets, history[-self.history_n:])
                        shown = [[self.images(r)[0][i] for i in self.opt_frames] for r in rows]
                        n_in = self.model.input_tokens(sum(map(len, shown)), prompt) if hasattr(self.model, 'input_tokens') else None
                        new = self.spec.get('runner', {}).get('generation', {}).get('max_new_tokens', 0)
                        if n_in is not None and n_in + new > self.model.context_limit():
                            raise ValueError(f'optimizer input of about {n_in} tokens plus {new} new tokens exceeds the '
                                             f'model context of {self.model.context_limit()}')
                        opt_text, opt_tokens = self.model.optimize(shown, prompt)
                        t2 = time.monotonic()
                        proposed, how = new_questions(opt_text, questions)
                        reason = 'no update' if how != 'updated' else acceptable(proposed)
                        cand, cand_videos = (self.check_score(proposed) if reason is None else (None, None))
                        t3 = time.monotonic()
                    except Exception:
                        step_log.write({**record, 'status': 'failed', 'error': traceback.format_exc(), 'finished_at': now()})
                        raise
                    accepted = cand is not None and score_key(cand) >= score_key(current)
                    if cand is not None:
                        history = (history + [{'questions': proposed, **{k: cand[k] for k in ('category_accuracy', 'group_accuracy')}}])
                    if accepted:
                        questions, current = proposed, cand
                    step_log.write({**record, 'status': 'complete', 'learner_responses': [t for _, _, t, _ in out],
                                    'learner_tokens': [k for _, _, _, k in out], 'predictions': preds, 'targets': targets,
                                    'optimizer_prompt': prompt, 'optimizer_response': opt_text, 'optimizer_tokens': opt_tokens,
                                    'optimizer_frames': self.opt_frames, 'optimizer_input_tokens': n_in, 'proposed': proposed, 'rejected_because': reason, 'candidate_check': cand,
                                    'candidate_check_predictions': cand_videos, 'accepted': accepted, 'questions_after': questions,
                                    'current_check': current, 'history_after': history[-self.history_n:],
                                    'learner_seconds': t1 - t0, 'optimizer_seconds': t2 - t1, 'check_seconds': t3 - t2, 'finished_at': now()})
                    ran += 1
                    if step % tr['val_every_steps'] == 0 and step not in vals:
                        self.validate(val_log, step, questions, session, {'epoch': epoch})
                        vals.add(step)
            return 'complete'
        finally:
            step_log.close()
            val_log.close()


def parse_yes_no(text):
    word = re.match(r'\W*(\w+)', text or '')
    word = word.group(1).lower() if word else ''
    return 'yes' if word == 'yes' else 'no' if word == 'no' else None


def parse_question(text, book):
    hits = list(re.finditer(r'question\s*:', text, re.I))
    if not hits:
        return None, 'no Question field'
    q = text[hits[-1].end():].strip().strip('`').strip().split('\n')[0].strip().strip('"\'*').strip()
    if not q.endswith('?'):
        return None, 'not a question'
    if len(q.split()) > 30:
        return None, 'longer than 30 words'
    names = list(book['categories']) + list(book['codes']) + ['Machine Gun', 'No Gun']
    if any(re.search(rf'\b{re.escape(n)}\b', q, re.I) for n in names):
        return None, 'names a label'
    return q, None


class BankRun(VariantBase):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.method = self.spec['training']['method']
        self.proposer = self.spec['prompts']['bank_proposer']
        self.yes_no = self.spec['prompts']['yes_no']

    @staticmethod
    def label(row, level):
        return row['category'] if level == 'category' else f"{row['category']}/{row['action_group']}"

    def classes(self):
        out = [('category', c) for c in CATS]
        out += [('group', f"{c}/{g}") for c in CATS for g in sorted({r['action_group'] for r in self.split['train'] if r['category'] == c})]
        return out

    def confusions(self, passes):
        """{(level, true, predicted): [video ids]} from the latest training pass."""
        out = {}
        for p in passes:
            t, pc, pg = p['target'], p['pred']['category'], p['pred']['group']
            if pc is not None and pc != t[0]:
                out.setdefault(('category', t[0], pc), []).append(p['video'])
            elif pc == t[0] and pg is not None and pg != t[1]:
                out.setdefault(('group', f'{t[0]}/{t[1]}', f'{t[0]}/{pg}'), []).append(p['video'])
        return out

    def describe(self, level, name):
        if level == 'category':
            return f"{name}: {self.book['categories'][name]}"
        cat, group = name.split('/')
        return f"{cat}, action group {group}: the person {self.book['groups'][group]['meaning']}"

    def proposal(self, rnd, conf, passes):
        """(level, A, B, video of A, video of B, mode); A was predicted as B for 'bank' and 'bank_noscore'."""
        tr, g = self.spec['training'], rng(self.spec['training']['seed'], 'round', rnd)
        by_id = {r['id']: r for r in self.split['train']}
        if self.method == 'bank_random' or not conf:
            pairs = [(l1, a, b) for l1, a in self.classes() for l2, b in self.classes() if l1 == l2 and a != b and
                     (l1 == 'category' or a.split('/')[0] == b.split('/')[0])]
            level, a, b = pairs[g.integers(len(pairs))]
            pool_a = [r for r in self.split['train'] if self.label(r, level) == a]
            va = pool_a[g.integers(len(pool_a))]
            mode = 'random'
        else:
            keys = sorted(conf)
            counts = np.array([len(conf[k]) for k in keys], float)
            level, a, b = keys[g.choice(len(keys), p=counts / counts.sum())]
            ids = conf[(level, a, b)]
            va = by_id[ids[g.integers(len(ids))]]
            mode = 'confusion'
        right = {p['video'] for p in passes if (p['pred']['category'] if level == 'category' else
                                                 f"{p['pred']['category']}/{p['pred']['group']}") == p['target_label'][level]}
        pool_b = [r for r in self.split['train'] if self.label(r, level) == b]
        good_b = [r for r in pool_b if r['id'] in right] if mode == 'confusion' else []
        vb = (good_b or pool_b)[g.integers(len(good_b or pool_b))]
        return level, a, b, va, vb, mode

    def proposer_prompt(self, level, a, b, mode):
        block = '[' + ('[' + frame_prefix(self.k) + '] ') * 2 + ']\n'
        relation = (f'The model wrongly classified Video 1 as {b}.' if mode == 'confusion'
                    else 'The model must tell these two classes apart.')
        return (self.proposer.replace('$Videos', block).replace('$ClassA', self.describe(level, a))
                .replace('$ClassB', self.describe(level, b)).replace('$Relation', relation))

    def score(self, question, level, a, b, rnd, exclude):
        tr, g = self.spec['training'], rng(self.spec['training']['seed'], 'score', rnd)
        sets = {}
        for name in (a, b):
            pool = [r for r in self.split['train'] if self.label(r, level) == name and r['id'] not in exclude]
            idx = sorted(g.choice(len(pool), min(tr['score_per_class'], len(pool)), replace=False))
            sets[name] = [pool[i] for i in idx]
        prompt = self.yes_no.replace('$Data', frame_prefix(self.k)).replace('$Question', question)
        answers = {}
        for name, rows in sets.items():
            got = []
            for i in range(0, len(rows), self.bs):
                texts, _ = self.model.learn([self.images(r)[0] for r in rows[i:i + self.bs]], prompt)
                got += [parse_yes_no(t) for t in texts]
            answers[name] = got
        yes_a = np.mean([x == 'yes' for x in answers[a]])
        no_b = np.mean([x == 'no' for x in answers[b]])
        yes_b = np.mean([x == 'yes' for x in answers[b]])
        no_a = np.mean([x == 'no' for x in answers[a]])
        balanced = max((yes_a + no_b) / 2, (yes_b + no_a) / 2)
        return float(balanced), {'videos': {n: [r['id'] for r in s] for n, s in sets.items()}, 'answers': answers,
                                 'orientation': 'yes means ' + (a if (yes_a + no_b) >= (yes_b + no_a) else b)}

    def current_questions(self, bank):
        tr = self.spec['training']
        if self.method == 'bank_noscore':
            chosen = bank[-tr['top_k']:]
        else:
            chosen, per_pair = [], {}
            for q in sorted(bank, key=lambda q: (-q['score'], q['round'])):
                key = q['pair']
                if per_pair.get(key, 0) < tr['per_pair_cap']:
                    chosen.append(q)
                    per_pair[key] = per_pair.get(key, 0) + 1
                if len(chosen) == tr['top_k']:
                    break
        if not chosen:
            return tr['seed_questions']
        return ''.join(f"{i + 1}. {q['question']}\n" for i, q in enumerate(chosen))

    def training_pass(self, questions):
        out = []
        limit = self.spec['training'].get('eval_limit')  # smoke runs only
        rows = self.split['train'][:limit] if limit else self.split['train']
        for r, p, t, k in self.predict_rows(rows, questions):
            out.append({'video': r['id'], 'target': target_of(r), 'pred': p, 'tokens': k,
                        'target_label': {'category': r['category'], 'group': f"{r['category']}/{r['action_group']}"}})
        return out

    def train(self, session, stop_after=None):
        tr = self.spec['training']
        rounds_path, passes_path, vals_path = self.dir / 'rounds.jsonl', self.dir / 'passes.jsonl', self.dir / 'validations.jsonl'
        rounds = [r for r in read_jsonl(rounds_path)[0] if r['status'] == 'complete']
        passes = read_jsonl(passes_path)[0]
        vals = {v['step'] for v in read_jsonl(vals_path)[0]}
        round_log, pass_log, val_log = JsonlAppender(rounds_path), JsonlAppender(passes_path), JsonlAppender(vals_path)
        bank = [r['entry'] for r in rounds if r.get('kept')]
        try:
            if 0 not in vals:
                self.validate(val_log, 0, tr['seed_questions'], session)
            ran = 0
            for rnd in range(1, tr['rounds'] + 1):
                needs_pass = self.method != 'bank_random' and (rnd - 1) % tr['refresh_every'] == 0
                if needs_pass and not any(p['before_round'] == rnd for p in passes):
                    questions = self.current_questions(bank)
                    begin = time.monotonic()
                    result = self.training_pass(questions)
                    passes.append({'before_round': rnd, 'questions': questions, 'predictions': result,
                                   'metrics': metrics(result), 'seconds': time.monotonic() - begin, 'finished_at': now()})
                    pass_log.write(passes[-1])
                if rnd <= len(rounds):
                    continue
                if stop_after is not None and ran >= stop_after:
                    return 'stopped'
                last = passes[-1]['predictions'] if passes else []
                conf = self.confusions(last)
                level, a, b, va, vb, mode = self.proposal(rnd, conf, last)
                record = {'round': rnd, 'session': session, 'level': level, 'class_a': a, 'class_b': b, 'mode': mode,
                          'video_a': va['id'], 'video_b': vb['id'], 'confusion_counts': {'|'.join(k): len(v) for k, v in conf.items()},
                          'started_at': now()}
                try:
                    prompt = self.proposer_prompt(level, a, b, mode)
                    t0 = time.monotonic()
                    text, tokens = self.model.optimize([self.images(va)[0], self.images(vb)[0]], prompt)
                    t1 = time.monotonic()
                    question, reason = parse_question(text, self.book)
                    scored, details = (None, None)
                    if question is not None and self.method != 'bank_noscore':
                        scored, details = self.score(question, level, a, b, rnd, {va['id'], vb['id']})
                    t2 = time.monotonic()
                except Exception:
                    round_log.write({**record, 'status': 'failed', 'error': traceback.format_exc(), 'finished_at': now()})
                    raise
                kept = question is not None and (self.method == 'bank_noscore' or scored >= tr['keep_threshold'])
                entry = {'question': question, 'pair': f'{level}|{a}|{b}', 'score': scored, 'round': rnd} if kept else None
                if kept:
                    bank.append(entry)
                round_log.write({**record, 'status': 'complete', 'proposer_prompt': prompt, 'proposer_response': text,
                                 'proposer_tokens': tokens, 'question': question, 'rejected_because': reason, 'score': scored,
                                 'score_details': details, 'kept': kept, 'entry': entry, 'questions_after': self.current_questions(bank),
                                 'proposer_seconds': t1 - t0, 'score_seconds': t2 - t1, 'finished_at': now()})
                ran += 1
                if rnd % tr['refresh_every'] == 0 and rnd not in vals:
                    self.validate(val_log, rnd, self.current_questions(bank), session, {'bank_size': len(bank)})
                    vals.add(rnd)
            return 'complete'
        finally:
            round_log.close()
            pass_log.close()
            val_log.close()


VARIANTS = {'vera_fixed': FixedRun, 'bank': BankRun, 'bank_random': BankRun, 'bank_noscore': BankRun}
