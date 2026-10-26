import json
from pathlib import Path
import tempfile
import unittest
from studies.firearm_training import codebook
from studies.firearm_training.prepare import assign, prepare

HERE = Path('studies/firearm_training')


class CodebookTests(unittest.TestCase):
    def test_committed_files_equal_the_generator(self):
        with tempfile.TemporaryDirectory() as td:
            codebook.write(Path(td))
            for name in ('codebook.json', 'prompts/learner_instruct.txt', 'prompts/optimizer_instruct.txt', 'prompts/seed_questions.txt',
                         'prompts/optimizer_fixed.txt', 'prompts/bank_proposer.txt', 'prompts/yes_no.txt'):
                self.assertEqual((Path(td) / name).read_text(), (HERE / name).read_text(), name)

    def test_codes_groups_and_prompt_structure(self):
        book = codebook.codebook()
        self.assertEqual(len(book['codes']), 41)
        self.assertNotIn('PAW7', book['codes'])
        self.assertEqual({c: book['codes'][c]['group'] for c in ('PCH1', 'PAH9', 'PAW8', 'N4', 'N5', 'N11', 'N12')},
                         {'PCH1': 'carry', 'PAH9': 'aim', 'PAW8': 'aim', 'N4': 'move', 'N5': 'phone', 'N11': 'bottle', 'N12': 'heavy_object'})
        self.assertEqual(book['codes']['PCW8']['category'], 'Machine_Gun')
        learner = (HERE / 'prompts/learner_instruct.txt').read_text()
        self.assertEqual(learner.count('$Data'), 1)
        self.assertEqual(learner.count('Based on the analysis above'), 1)
        optimizer = (HERE / 'prompts/optimizer_instruct.txt').read_text()
        for marker in ('** Model Descriptions: **', 'Based on the analysis above', '[$Prediction]', '[$GroundTruth]', 'New Prompt Questions'):
            self.assertIn(marker, optimizer)
        # No folder names or video IDs leak into the prompts.
        for text in (learner, optimizer):
            self.assertNotIn('_C1_', text)
            self.assertNotIn('__', text)


def row(vid, cat, action, group, n=10):
    return {'id': vid, 'category': cat, 'action': action, 'folder': vid.split('__')[1], 'camera': 'C1', 'camera_type': 'cctv_ip',
            'location': 'P1', 'subject': 'V1', 'brightness': 'HB', 'take': 1, 'recording_group': group, 'width': 64, 'height': 48,
            'fps': 25.0, 'frames_decoded': n, 'duration_s': n / 25, 'staged_path': f'/x/{vid}.mp4', 'sha256': 'x', 'label_file': None}


class SplitTests(unittest.TestCase):
    def test_groups_stay_together_and_every_code_reaches_every_split(self):
        inv = []
        for code, cat in (('PAH1', 'Handgun'), ('N1', 'No_Gun')):
            for g in range(7):
                for take in range(2):
                    inv.append(row(f'{cat}__{code}_C1_P1_V1_HB_{g * 10 + take}', cat, code, f'{code}_g{g}'))
        split = assign(inv, 20261026)
        self.assertEqual(split, assign(inv, 20261026))
        for g in {r['recording_group'] for r in inv}:
            self.assertEqual(len({split[r['id']] for r in inv if r['recording_group'] == g}), 1)
        for code in ('PAH1', 'N1'):
            counts = {s: sum(1 for r in inv if r['action'] == code and split[r['id']] == s) for s in ('train', 'val', 'test')}
            self.assertEqual(counts, {'train': 8, 'val': 2, 'test': 4})

    def test_prepare_writes_labels_and_separate_boxes(self):
        from tests.firearm.fixtures import coco, write_video
        from vera_runner.firearm_data import build_inventory, stage, write_inventory
        with tempfile.TemporaryDirectory() as td:
            orig = Path(td) / 'o'
            for g in range(3):
                write_video(orig / f'Handgun/PAH1_C1_P{g + 1}_V1_HB_1/video.mp4', 6)
                (orig / f'Handgun/PAH1_C1_P{g + 1}_V1_HB_1/label.json').write_text(json.dumps(coco(6, (64, 48), {2: [1, 2, 10, 12]})))
                write_video(orig / f'No_Gun/N1_C2_P3_V{g + 1}_HB_1/video.mp4', 6)
            write_inventory(stage(build_inventory(orig, 1), orig, Path(td) / 'staged'), td)
            policy = Path(td) / 'policy.json'
            policy.write_text(json.dumps({'origin': 1, 'label_files': {}}))
            with self.assertRaises(ValueError):  # the full codebook has codes this inventory lacks
                prepare(Path(td) / 'inventory.json', HERE / 'codebook.json', policy, orig, Path(td) / 'out', 7)
            book = json.loads((HERE / 'codebook.json').read_text())
            book['codes'] = {k: book['codes'][k] for k in ('PAH1', 'N1')}
            (Path(td) / 'book.json').write_text(json.dumps(book))
            result = prepare(Path(td) / 'inventory.json', Path(td) / 'book.json', policy, orig, Path(td) / 'out', 7)
            self.assertEqual(result['groups_spanning_splits'], [])
            self.assertEqual(result['codes_missing_from_a_split'], [])
            self.assertEqual({s: result['splits'][s]['videos'] for s in ('train', 'val', 'test')}, {'train': 2, 'val': 2, 'test': 2})
            train = json.loads((Path(td) / 'out/train.json').read_text())
            self.assertEqual({r['action_group'] for r in train}, {'aim', 'move'})
            self.assertNotIn('boxes', train[0])
            boxes = json.loads((Path(td) / 'out/boxes_train.json').read_text())
            hand = next(v for k, v in boxes.items() if k.startswith('Handgun'))
            self.assertEqual(hand['boxes'], {'1': [[15.625, 41.666666666666664, 171.875, 291.6666666666667]]})
            self.assertEqual(hand['statuses'][1], 'positive')
