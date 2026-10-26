import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from vera_runner.config import ROOT
from vera_runner.verbalized import (Run, learner_prompt, new_questions, optimizer_prompt, parse_answer, sample_indices)
from tests.firearm.fixtures import write_video

BOOK = json.loads((ROOT / 'studies/firearm_training/codebook.json').read_text())
VERA_PRED = '[[$Prediction] [$Prediction] [$Prediction] [$Prediction] [$Prediction] [$Prediction] [$Prediction] [$Prediction] [$Prediction] [$Prediction]]'
VERA_TRUTH = VERA_PRED.replace('Prediction', 'GroundTruth')


def vera_methods():
    tree = ast.parse((ROOT / 'training.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VideoAnomalyDetectionModel')
    fns = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('forward', 'training_step')]
    ns = {'torch': torch, 'tokenizer': None, 'generation_config': {'max_new_tokens': 1024}}
    exec(compile(ast.Module(body=fns, type_ignores=[]), '<training.py>', 'exec'), ns)
    return ns['forward'], ns['training_step']


class Fake:
    def __init__(self, learner_answers, optimizer_answer):
        self.learner_answers, self.optimizer_answer, self.calls = learner_answers, optimizer_answer, []

    def eval(self):
        pass

    def batch_chat(self, tokenizer, pixel_values, num_patches_list, questions, generation_config):
        self.calls.append(('learner', questions, num_patches_list, tuple(pixel_values.shape)))
        return self.learner_answers

    def chat(self, tokenizer, pixel_values, question, generation_config):
        self.calls.append(('optimizer', question, None, tuple(pixel_values.shape)))
        return self.optimizer_answer


class FaithfulnessTests(unittest.TestCase):
    def test_prompts_and_question_update_equal_vera_training_py(self):
        forward, training_step = vera_methods()
        learner = (ROOT / 'VERA_learner_instruct.txt').read_text()
        optimizer = (ROOT / 'VERA_optimizer_instruct.txt').read_text()
        answer = 'Reasoning:\nmore\nNew Prompt Questions:\n1. A?\n2. B?\n3. C?\n4. D?\n5. E?\n6. F?\n'
        fake = Fake(['x\nOutput: 0', 'y\nOutput: 1'], answer)
        qs = '1. Seed one?\n2. Seed two?\n'
        self_ = SimpleNamespace(model=fake, tokenizer=None, model_instruct=learner, optimizer_instruct=optimizer, new_qs=qs,
                                log=lambda *a, **k: None)
        self_.forward = lambda pv, npl, batch_idx=0: forward(self_, pv, npl, batch_idx)
        batch = (torch.zeros(2, 8, 3, 4, 4), [1] * 8, torch.tensor([0, 1]), ['a', 'b'])
        training_step(self_, batch, 3)
        vera_learner, vera_optimizer = fake.calls[0][1][0], fake.calls[1][1]
        self.assertEqual(learner_prompt(learner, qs), vera_learner)
        self.assertEqual(fake.calls[0][2], [8, 8])  # each video's 8 frames in its own patch group
        self.assertEqual(fake.calls[1][3][0], 16)    # the optimizer sees both videos' frames
        ours = optimizer_prompt(optimizer.replace(VERA_PRED, '[$Prediction]').replace(VERA_TRUTH, '[$GroundTruth]'), qs, [0, 1], [0, 1])
        self.assertEqual(ours, vera_optimizer)
        self.assertEqual(new_questions(answer, qs)[0], self_.new_qs)
        self.assertEqual(new_questions('no update here', qs), (qs, 'kept: no "New Prompt Questions" in the answer'))

    def test_sampling_equals_the_training_dataset(self):
        self.assertEqual(sample_indices(250), np.arange(0, 250, 250 / 8).astype(int).tolist())
        self.assertEqual(sample_indices(125), [0, 15, 31, 46, 62, 78, 93, 109])


class ParserTests(unittest.TestCase):
    def test_fields_and_normalisation(self):
        text = 'Answers to Prompt Questions: The category seems clear.\nCategory:\nMachine Gun\nAction group:\nheavy object\nAction code:\n`pcw1`.'
        self.assertEqual(parse_answer(text, BOOK), {'category': 'Machine_Gun', 'group': 'heavy_object', 'code': 'PCW1'})
        self.assertEqual(parse_answer('Category: [No_Gun]\nAction group: phone\nAction code: N7', BOOK),
                         {'category': 'No_Gun', 'group': 'phone', 'code': 'N7'})

    def test_invalid_values_are_none_never_defaults(self):
        self.assertEqual(parse_answer('Category: Rifle\nAction group: shooting\nAction code: PAW7', BOOK),
                         {'category': None, 'group': None, 'code': None})
        self.assertEqual(parse_answer('nothing structured', BOOK), {'category': None, 'group': None, 'code': None})
        self.assertEqual(parse_answer('Category: Handgun', BOOK)['code'], None)


class FakeCalls:
    """Deterministic stand-in for the two InternVL2 calls: bright first frame -> Handgun/aim/PAH1."""
    def __init__(self):
        self.optimizer_calls = 0

    def learn(self, videos, question):
        out = []
        for v in videos:
            bright = sum(v[0].convert('L').getdata()) / (v[0].width * v[0].height) >= 128
            out.append('Answers to Prompt Questions: synthetic.\nCategory:\n' + ('Handgun\nAction group:\naim\nAction code:\nPAH1' if bright
                       else 'No_Gun\nAction group:\nmove\nAction code:\nN1'))
        return out, [10] * len(videos)

    def optimize(self, videos, prompt):
        self.optimizer_calls += 1
        return f'Reasoning:\nsynthetic\nNew Prompt Questions:\n1. Question {self.optimizer_calls}?\n', 5


def make_split(td):
    rows = {'train': [], 'val': [], 'test': []}
    n = 0
    for split, count in (('train', 8), ('val', 4), ('test', 4)):
        for i in range(count):
            bright = (n % 2 == 0)
            vid = f"{'Handgun' if bright else 'No_Gun'}__{'PAH1' if bright else 'N1'}_C1_P1_V1_HB_{n}"
            path = Path(td) / 'videos' / f'{vid}.mp4'
            write_video(path, 20, value=lambda j, b=bright: np.full((48, 64, 3), 220 if b else 20, np.uint8))
            rows[split].append({'id': vid, 'category': 'Handgun' if bright else 'No_Gun', 'action': 'PAH1' if bright else 'N1',
                                'action_group': 'aim' if bright else 'move', 'camera': 'C1', 'recording_group': vid,
                                'frames_decoded': 20, 'staged_path': str(path)})
            n += 1
    data = Path(td) / 'data'
    data.mkdir()
    for s, r in rows.items():
        (data / f'{s}.json').write_text(json.dumps(r))
    return data


def spec():
    return {'prompts': {'learner_instruct': (ROOT / 'studies/firearm_training/prompts/learner_instruct.txt').read_text(),
                        'optimizer_instruct': (ROOT / 'studies/firearm_training/prompts/optimizer_instruct.txt').read_text()},
            'training': {'batch_size': 2, 'frames_per_video': 8, 'max_epochs': 2, 'val_check_interval': 2, 'seed': 1,
                         'seed_questions': '1. Seed?\n'}}


class TrainingRunTests(unittest.TestCase):
    def test_train_validate_select_test_and_resume(self):
        with tempfile.TemporaryDirectory() as clean, tempfile.TemporaryDirectory() as td:
            reference = Run(spec(), Path(clean) / 'run', make_split(clean), Path(clean) / 'frames', FakeCalls(), BOOK)
            (Path(clean) / 'run').mkdir()
            self.assertEqual(reference.train('001'), 'complete')
            data = make_split(td)
            (Path(td) / 'run').mkdir()
            calls = FakeCalls()
            run = Run(spec(), Path(td) / 'run', data, Path(td) / 'frames', calls, BOOK)
            self.assertEqual(run.train('001', stop_after=3), 'stopped')
            self.assertEqual(run.train('002'), 'complete')
            read = lambda d, f: [json.loads(l) for l in (Path(d) / 'run' / f).read_text().splitlines()]
            steps = read(td, 'steps.jsonl')
            self.assertEqual([s['step'] for s in steps], list(range(1, 9)))
            self.assertEqual([s['videos'] for s in steps], [s['videos'] for s in read(clean, 'steps.jsonl')])
            vals = read(td, 'validations.jsonl')
            self.assertEqual([v['step'] for v in vals], [0, 2, 4, 6, 8])
            self.assertEqual(vals[0]['metrics']['category_accuracy'], 1.0)
            self.assertTrue(all(len(v['predictions']) == 4 for v in vals))
            best = run.test('003')
            tests = read(td, 'test.jsonl')
            self.assertEqual([t['name'] for t in tests], ['seed', 'learned'])
            self.assertEqual(tests[1]['questions'], best['questions'])
            self.assertEqual(best['step'], 2)  # ties on accuracy break to the earliest step
            self.assertTrue(all(len(list((Path(td) / 'frames' / r['id']).glob('*.jpg'))) == 8 for r in json.loads((data / 'test.json').read_text())))
