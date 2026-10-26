import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from studies.firearm_package import e1, e2
from studies.firearm_package.manifest import build, select, write
from vera_runner.ledger import read_jsonl

ROOT = Path(__file__).resolve().parents[2]


def make_data(td, per_class=(6, 2)):
    data, frames = Path(td) / 'data', Path(td) / 'frames'
    data.mkdir()
    for split, n in zip(('train', 'val'), per_class):
        rows, boxes = [], {}
        for cat, group, code in (('Handgun', 'aim', 'PAH1'), ('Machine_Gun', 'carry', 'PCW1'), ('No_Gun', 'phone', 'N5')):
            for i in range(n):
                vid = f'{cat}__{code}_C1_P1_V1_HB_{split}{i}'
                rows.append({'id': vid, 'category': cat, 'action_group': group, 'action': code, 'recording_group': f'{code}_{split}{i % 2}',
                             'camera': 'C1', 'frames_decoded': 16, 'width': 64, 'height': 48})
                statuses = ['negative_category'] * 16 if cat == 'No_Gun' else ['positive' if k % 2 == 0 else 'listed_no_box' for k in range(16)]
                boxes[vid] = {'statuses': statuses, 'boxes': {} if cat == 'No_Gun' else {str(k): [[100, 100, 300, 400]] for k in range(0, 16, 2)}}
                (frames / vid).mkdir(parents=True)
                for k in range(0, 16, 2):
                    Image.new('RGB', (64, 48), (200, 200, 200) if cat != 'No_Gun' else (10, 10, 10)).save(frames / vid / f'{k:06d}.jpg')
        (data / f'{split}.json').write_text(json.dumps(rows))
        (data / f'boxes_{split}.json').write_text(json.dumps(boxes))
    cfg = {'data_dir': str(data), 'frames_dir': str(frames), 'seed': 3, 'out_root': str(Path(td) / 'out'),
           'models': {'m': 'fake'}, 'codebook': 'studies/firearm_training/codebook.json',
           'learner_instruct_file': 'studies/firearm_training/prompts/learner_instruct.txt',
           'seed_questions_file': 'studies/firearm_training/prompts/seed_questions.txt',
           'manifest': {'firearm_frames_per_video': 4, 'nogun_frames_per_video': 2, 'subset300': 10, 'subset100': 4}}
    return cfg


class FakeDetector:
    def __init__(self):
        self.calls = []

    def chat(self, images_per_slot, question, max_new_tokens, seed=0, packed=False):
        self.calls.append((len(images_per_slot), seed))
        im = images_per_slot[0][0].convert('L')
        bright = sum(im.getdata()) / (im.width * im.height) > 100
        answer = '<ref>handgun</ref><box>[[100, 100, 300, 400]]</box>' if bright else 'none'
        return {'text': answer, 'thinking': '', 'answer': answer, 'truncated': False, 'tokens': 9, 'seconds': 0.1}


class FakeLearner:
    def __init__(self):
        self.calls = []

    def chat(self, images_per_slot, question, max_new_tokens, seed=0, packed=False):
        self.calls.append(([len(s) for s in images_per_slot], packed, question.count('<image>')))
        im = images_per_slot[0][0].convert('L')
        bright = sum(im.getdata()) / (im.width * im.height) > 100
        answer = ('Answers to Prompt Questions: x\nCategory: Handgun\nAction group: aim\nAction code: PAH1' if bright else
                  'Answers to Prompt Questions: x\nCategory: No_Gun\nAction group: phone\nAction code: N5')
        return {'text': answer, 'thinking': '', 'answer': answer, 'truncated': False, 'tokens': 20, 'seconds': 0.2}


class ManifestTests(unittest.TestCase):
    def test_manifest_positive_frames_subsets_nested_and_stable(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = make_data(td)
            frames = build(cfg)
            self.assertEqual(frames, build(cfg))  # seeded
            gun = [f for f in frames if f['category'] != 'No_Gun']
            self.assertTrue(all(f['gt'] and f['frame_index'] % 2 == 0 for f in gun))
            self.assertEqual(len(gun), 16 * 4 // 4 * 4)  # 16 firearm videos x 4 positive frames
            self.assertEqual(len([f for f in frames if f['category'] == 'No_Gun']), 16)
            s100, s300 = {f['frame_id'] for f in select(frames, 'subset100')}, {f['frame_id'] for f in select(frames, 'subset300')}
            self.assertEqual((len(s100), len(s300)), (4, 10))
            self.assertTrue(s100 <= s300)


class E1Tests(unittest.TestCase):
    def test_detect_resume_scores_and_spec_guard(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = make_data(td)
            mpath = Path(td) / 'out/e1_manifest.json'
            write(cfg, mpath)
            arm = {'model': 'm', 'family': 'internvl', 'thinking': False, 'frames': 'all', 'max_new_tokens': 64, 'max_num': 6}
            fake = FakeDetector()
            first = e1.run_arm(cfg, 'a', arm, mpath, Path(td) / 'run', model=fake, limit=5)
            second = e1.run_arm(cfg, 'a', arm, mpath, Path(td) / 'run', model=fake)
            self.assertEqual((first['ran'], second['done_before'], second['ran']), (5, 5, 75))
            recs = read_jsonl(Path(td) / 'run/records.jsonl')[0]
            gun = [r for r in recs if r['category'] != 'No_Gun']
            self.assertTrue(all(r['score']['tp50'] == 1 for r in gun))
            self.assertTrue(all(r['status'] == 'none' for r in recs if r['category'] == 'No_Gun'))
            with self.assertRaisesRegex(ValueError, 'spec changed'):
                e1.run_arm(cfg, 'a', {**arm, 'max_new_tokens': 65}, mpath, Path(td) / 'run', model=fake)


class E2Tests(unittest.TestCase):
    def test_inputs_and_crops(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = make_data(td)
            for inp, expect in (('packed', ([8], True, 8)), ('frames', ([1] * 8, False, 8)), ('crops', ([1] * 10, False, 10))):
                fake = FakeLearner()
                arm = {'model': 'm', 'input': inp, 'thinking': False, 'max_new_tokens': 64}
                res = e2.run_arm(cfg, inp, arm, Path(td) / inp, ROOT, model=fake)
                self.assertEqual(res['ran'], 24)
                self.assertEqual(fake.calls[0], expect)
                recs = read_jsonl(Path(td) / inp / 'records.jsonl')[0]
                self.assertTrue(all((r['pred']['category'] == 'No_Gun') == (r['category'] == 'No_Gun') for r in recs))
                if inp == 'crops':
                    srcs = {r['category']: {c['source'] for c in r['crops']} for r in recs}
                    self.assertEqual((srcs['Handgun'], srcs['No_Gun']), ({'gt'}, {'random'}))


class Phase1AnalysisTests(unittest.TestCase):
    def test_gates_from_fake_runs(self):
        from studies.firearm_package.analysis_phase1 import analyze_e1, analyze_e2
        with tempfile.TemporaryDirectory() as td:
            cfg = make_data(td)
            out = Path(cfg['out_root'])
            mpath = out / 'e1_manifest.json'
            write(cfg, mpath)
            arm = {'model': 'm', 'family': 'internvl', 'thinking': False, 'frames': 'all', 'max_new_tokens': 64, 'max_num': 6}
            e1.run_arm(cfg, 'good', arm, mpath, out / 'e1/good', model=FakeDetector())
            seeds = Path(td) / 'seed.jsonl'
            seeds.write_text(''.join(json.dumps({'video': f['video'], 'category': f['category'],
                                                 'pred': {'category': 'No_Gun'}}) + '\n' for f in json.loads(mpath.read_text())))
            result, gate = analyze_e1(out, mpath, seeds, ['good'], {'min_recall50': 0.5, 'max_whole_image': 0.1, 'max_nogun_fp': 0.1})
            self.assertEqual(result['good']['own']['recall50']['est'], 1.0)
            self.assertEqual(result['good']['own']['nogun_fp']['est'], 0.0)
            self.assertEqual(result['good']['own']['recall50_handgun_missed_by_seed'], 1.0)
            self.assertEqual(gate['chosen'], 'good')
            for n in ('iv2_packed', 'iv35_packed', 'iv35_frames'):
                e2.run_arm(cfg, n, {'model': 'm', 'input': n.split('_')[1], 'thinking': False, 'max_new_tokens': 64}, out / 'e2' / n,
                           ROOT, model=FakeLearner())
            result, gate = analyze_e2(out, ['iv2_packed', 'iv35_packed', 'iv35_frames'])
            self.assertEqual(result['iv35_packed']['vs_reference_category']['est'], 0.0)
            self.assertEqual((gate['backbone']['chosen'], gate['input']['chosen']), ('internvl2_8b', 'packed'))
