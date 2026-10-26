import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from vera_runner.config import ROOT
from vera_runner.ledger import read_jsonl
from vera_runner.verbalized_variants import BankRun, FixedRun, acceptable, parse_question, parse_yes_no
from tests.firearm.fixtures import write_video
from tests.firearm.test_verbalized import BOOK

P = ROOT / 'studies/firearm_training/prompts'
SHADE = {'Handgun': 220, 'Machine_Gun': 200, 'No_Gun': 20}  # the fake learner cannot tell Handgun from Machine_Gun
LABEL = {'Handgun': ('PAH1', 'aim'), 'Machine_Gun': ('PCW1', 'carry'), 'No_Gun': ('N5', 'phone')}


class FakeCalls:
    """Bright first frame -> Handgun/aim/PAH1, dark -> No_Gun/phone/N5; yes/no: bright -> yes."""
    def __init__(self, describe_first=False):
        self.calls, self.describe_first = 0, describe_first

    def learn(self, videos, question):
        out = []
        for v in videos:
            bright = sum(v[0].convert('L').getdata()) / (v[0].width * v[0].height) >= 128
            if 'only yes or no' in question:
                out.append('Yes.' if bright else 'No')
            else:
                out.append('Answers to Prompt Questions: synthetic.\nCategory:\n' +
                           ('Handgun\nAction group:\naim\nAction code:\nPAH1' if bright else 'No_Gun\nAction group:\nphone\nAction code:\nN5'))
        return out, [9] * len(videos)

    def optimize(self, videos, prompt):
        self.calls += 1
        if 'Write ONE question' in prompt:
            return 'Reasoning:\nOne is bright.\nQuestion:\nIs the scene bright?', 7
        if self.describe_first and self.calls == 1:
            return 'Reasoning:\nx\nNew Prompt Questions:\n1. Describe the scene in detail.\n', 7
        return f'Reasoning:\nx\nNew Prompt Questions:\n1. Is it bright? ({self.calls})\n2. Is the person moving?\n', 7


def make_split(td, per_class=(10, 2, 2)):
    rows = {'train': [], 'val': [], 'test': []}
    n = 0
    for split, count in zip(('train', 'val', 'test'), per_class):
        for cat in ('Handgun', 'Machine_Gun', 'No_Gun'):
            for i in range(count):
                vid = f'{cat}__{LABEL[cat][0]}_C1_P1_V1_HB_{n}'
                path = Path(td) / 'videos' / f'{vid}.mp4'
                write_video(path, 16, value=lambda j, s=SHADE[cat]: np.full((48, 64, 3), s, np.uint8))
                rows[split].append({'id': vid, 'category': cat, 'action': LABEL[cat][0], 'action_group': LABEL[cat][1], 'camera': 'C1',
                                    'recording_group': vid, 'frames_decoded': 16, 'staged_path': str(path)})
                n += 1
    data = Path(td) / 'data'
    data.mkdir()
    for s, r in rows.items():
        (data / f'{s}.json').write_text(json.dumps(r))
    return data


def spec(method, **training):
    base = {'batch_size': 2, 'frames_per_video': 8, 'max_epochs': 1, 'val_check_interval': 100, 'seed': 3,
            'seed_questions': '1. Seed?\n', 'eval_limit': None, 'method': method, **training}
    return {'prompts': {'learner_instruct': (P / 'learner_instruct.txt').read_text(),
                        'optimizer_instruct': (P / ('optimizer_fixed.txt' if method == 'vera_fixed' else 'optimizer_instruct.txt')).read_text(),
                        'bank_proposer': (P / 'bank_proposer.txt').read_text(), 'yes_no': (P / 'yes_no.txt').read_text()},
            'training': base}


FIXED = dict(videos_per_class=2, check_per_class=2, history=3, val_every_steps=2)
BANK = dict(rounds=6, refresh_every=3, score_per_class=3, keep_threshold=0.75, top_k=3, per_pair_cap=2)


def read(run, name):
    return read_jsonl(Path(run) / name)[0]


class HelperTests(unittest.TestCase):
    def test_parsers_and_rules(self):
        self.assertEqual([parse_yes_no(x) for x in ('Yes.', ' no, it is not', 'Maybe', '')], ['yes', 'no', None, None])
        self.assertEqual(parse_question('Reasoning: x\nQuestion:\nIs the arm extended?', BOOK), ('Is the arm extended?', None))
        self.assertEqual(parse_question('Question: Is it a Handgun?', BOOK)[1], 'names a label')
        self.assertEqual(parse_question('Question: The arm is extended.', BOOK)[1], 'not a question')
        self.assertIsNone(acceptable('1. Is the arm extended?\n2. Is it dark?\n'))
        self.assertEqual(acceptable('1. Describe the scene.\n'), 'asks to describe, explain or list')
        self.assertEqual(acceptable(''.join(f'{i}. Q{i}?\n' for i in range(1, 7))), 'more than five questions')


class FixedRunTests(unittest.TestCase):
    def test_balanced_batches_check_set_acceptance_and_resume(self):
        with tempfile.TemporaryDirectory() as clean, tempfile.TemporaryDirectory() as td:
            data_c, data = make_split(clean), make_split(td)
            for d in (clean, td):
                (Path(d) / 'run').mkdir()
            ref = FixedRun(spec('vera_fixed', **FIXED), Path(clean) / 'run', data_c, Path(clean) / 'frames', FakeCalls(True), BOOK)
            self.assertEqual(ref.train('001'), 'complete')
            run = FixedRun(spec('vera_fixed', **FIXED), Path(td) / 'run', data, Path(td) / 'frames', FakeCalls(True), BOOK)
            self.assertEqual(run.train('001', stop_after=2), 'stopped')
            run.model = FakeCalls()  # a fresh process: the fake's call counter restarts, as a model would
            self.assertEqual(run.train('002'), 'complete')
            steps = read(Path(td) / 'run', 'steps.jsonl')
            self.assertEqual(len(steps), run.steps_per_epoch)
            self.assertEqual([s['videos'] for s in steps], [s['videos'] for s in read(Path(clean) / 'run', 'steps.jsonl')])
            check = {r['id'] for r in run.check}
            for s in steps:
                cats = [v.split('__')[0] for v in s['videos']]
                self.assertEqual(sorted(cats), sorted(['Handgun', 'Machine_Gun', 'No_Gun'] * 2))
                self.assertFalse(check & set(s['videos']))
            first = read(Path(clean) / 'run', 'steps.jsonl')[0]
            self.assertEqual((first['rejected_because'], first['accepted']), ('asks to describe, explain or list', False))
            self.assertIn('[$History]', run.optimizer_template)
            self.assertIn('Check accuracy', steps[-1]['optimizer_prompt'])
            self.assertEqual([v['step'] for v in read(Path(td) / 'run', 'validations.jsonl')], [0, 2, 4])
            key = lambda m: (m['category_accuracy'], m['group_accuracy'])
            previous = None
            for s in read(Path(clean) / 'run', 'steps.jsonl'):  # never accept a set that checks lower than the current one
                if s['accepted'] and previous is not None:
                    self.assertGreaterEqual(key(s['candidate_check']), key(previous))
                if not s['accepted']:
                    self.assertEqual(s['questions_after'], s['questions_before'])
                previous = s['current_check']
            best = run.test('003')
            self.assertEqual([t['name'] for t in read(Path(td) / 'run', 'test.jsonl')], ['seed', 'learned'])
            self.assertGreater(best['step'], 0)


class BankRunTests(unittest.TestCase):
    def run_bank(self, method, td):
        data = make_split(td)
        (Path(td) / 'run').mkdir()
        run = BankRun(spec(method, **BANK), Path(td) / 'run', data, Path(td) / 'frames', FakeCalls(), BOOK)
        self.assertEqual(run.train('001'), 'complete')
        return run, read(Path(td) / 'run', 'rounds.jsonl'), read(Path(td) / 'run', 'passes.jsonl')

    def test_confusion_targeting_scoring_and_selection(self):
        with tempfile.TemporaryDirectory() as td:
            run, rounds, passes = self.run_bank('bank', td)
            self.assertEqual([p['before_round'] for p in passes], [1, 4])
            self.assertEqual(len(rounds), 6)
            # The only confusion of the fake learner: Machine_Gun videos are called Handgun.
            self.assertTrue(all((r['class_a'], r['class_b'], r['mode']) == ('Machine_Gun', 'Handgun', 'confusion') for r in rounds))
            self.assertTrue(all(r['video_a'].startswith('Machine_Gun') and r['video_b'].startswith('Handgun') for r in rounds))
            # "Is the scene bright?" cannot separate two bright classes: scored 0.5, never kept.
            self.assertTrue(all(r['score'] == 0.5 and not r['kept'] for r in rounds))
            self.assertTrue(all(r['video_a'] not in sum(r['score_details']['videos'].values(), []) for r in rounds))
            self.assertEqual([v['step'] for v in read(Path(td) / 'run', 'validations.jsonl')], [0, 3, 6])
            run.test('002')

    def test_random_pairs_and_no_scoring_ablations(self):
        with tempfile.TemporaryDirectory() as td:
            _, rounds, passes = self.run_bank('bank_random', td)
            self.assertEqual(passes, [])
            self.assertTrue(all(r['mode'] == 'random' for r in rounds))
            kept = [r for r in rounds if r['kept']]
            self.assertTrue(all(r['score'] >= 0.75 for r in kept))
        with tempfile.TemporaryDirectory() as td:
            run, rounds, _ = self.run_bank('bank_noscore', td)
            self.assertTrue(all(r['kept'] and r['score'] is None for r in rounds))
            self.assertEqual(rounds[-1]['questions_after'].count('Is the scene bright?'), 3)

    def test_resume_reproduces_choices(self):
        with tempfile.TemporaryDirectory() as clean, tempfile.TemporaryDirectory() as td:
            _, ref, _ = self.run_bank('bank', clean)
            data = make_split(td)
            (Path(td) / 'run').mkdir()
            run = BankRun(spec('bank', **BANK), Path(td) / 'run', data, Path(td) / 'frames', FakeCalls(), BOOK)
            self.assertEqual(run.train('001', stop_after=4), 'stopped')
            self.assertEqual(run.train('002'), 'complete')
            got = read(Path(td) / 'run', 'rounds.jsonl')
            key = lambda r: (r['round'], r['video_a'], r['video_b'], r['score'])
            self.assertEqual([key(r) for r in got], [key(r) for r in ref])
