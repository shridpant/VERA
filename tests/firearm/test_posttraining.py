import copy
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from studies.firearm_posttraining import data, rewards
from studies.firearm_posttraining.model import target_parts, token_kinds, instruction
from studies.firearm_posttraining.stopping import StopState
from studies.firearm_posttraining.evaluation import metrics
from studies.firearm_posttraining.execute import budget_used, reserve, finish
from studies.firearm_posttraining.verbalized import valid_revision
from studies.firearm_posttraining.protocol import validate, verify_selection

BOOK = json.loads(Path('studies/firearm_training/codebook.json').read_text())


def fixture():
    return {'id': 'fixture', 'category': 'Handgun', 'action_group': 'aim', 'action_code': 'PAH1',
            'views': [{'frame_id': k, 'boxes': [[.2, .3, .4, .5]] if k < 2 else None} for k in range(8)]}


def response(row, grounding=True):
    result = {k: row[k] for k in ('category', 'action_group', 'action_code')}
    if grounding:
        result['grounding'] = [{'frame_id': v['frame_id'], 'boxes': copy.deepcopy(v['boxes'] or [])} for v in row['views']]
    return result


class DataTests(unittest.TestCase):
    def test_group_split_determinism_coverage_and_identity_guard(self):
        rows = [{'id': f'{c}-{g}-{t}', 'sha256': f'{c}-{g}-{t}', 'category': c,
                 'recording_group': f'{c}-{g}'} for c in data.CATEGORIES for g in range(9) for t in range(1 + g % 3)]
        parts = data.split_train(rows, 1)
        self.assertEqual(parts, data.split_train(list(reversed(rows)), 1))
        data.assert_disjoint(parts)
        for split in parts.values():
            self.assertEqual({r['category'] for r in split}, set(data.CATEGORIES))
        broken = copy.deepcopy(parts)
        broken['fit'].append(parts['stop'][0])
        with self.assertRaises(ValueError):
            data.assert_disjoint(broken)
        order = data.balanced_order(parts['fit'], 4, 0)
        self.assertEqual(len(set(order)), len(parts['fit']))
        self.assertEqual(order, data.balanced_order(parts['fit'], 4, 0))
        self.assertNotEqual(order, data.balanced_order(parts['fit'], 4, 1))

    def test_letterbox_roundtrip_and_aspect_ratio(self):
        from PIL import Image
        for w, h in ((640, 480), (1920, 1080), (1080, 1920), (643, 481)):
            g = data.geometry(w, h)
            image = data.letterbox(Image.new('RGB', (w, h)), g)
            self.assertEqual(image.size, (448, 448))
            box = [0, 25, 900, 1000]
            for a, b in zip(box, data.box_from_view(data.box_to_view(box, g), g)):
                self.assertAlmostEqual(a, b, places=10)

    def test_unknown_annotations_are_not_negative_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image
            row = {'id': 'a', 'width': 64, 'height': 48, 'frames_decoded': 8, 'action': 'PAH1'}
            Path(tmp, 'a').mkdir()
            for k in range(8):
                Image.new('RGB', (64, 48)).save(Path(tmp, 'a', f'{k:06d}.jpg'))
            ann = {'statuses': ['listed_no_box', 'unknown', 'negative_category', 'positive'] * 2,
                   'boxes': {'3': [[1, 2, 3, 4]], '7': [[1, 2, 3, 4]]}}
            rich = data.enrich(row, ann, tmp)
            self.assertIsNone(rich['views'][0]['boxes'])
            self.assertIsNone(rich['views'][1]['boxes'])
            self.assertEqual(rich['views'][2]['boxes'], [])
            self.assertEqual(len(rich['views'][3]['boxes']), 1)

    def test_holdout_guard_and_immutable_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp, 'x.json')
            data.write_json(p, {'a': 1}, immutable=True)
            data.write_json(p, {'a': 1}, immutable=True)
            with self.assertRaises(ValueError):
                data.write_json(p, {'a': 2}, immutable=True)
            for name in ('test', 'val', '../test'):
                with self.assertRaises(ValueError):
                    data.load_split(tmp, name)


class RewardTests(unittest.TestCase):
    def test_exact_hierarchy_inconsistent_codes_and_verbosity(self):
        row = fixture()
        p = response(row, False)
        self.assertAlmostEqual(rewards.score(json.dumps(p), row, BOOK)['reward'], 1.)
        p['action_code'] = 'N1'
        self.assertAlmostEqual(rewards.score(json.dumps(p), row, BOOK)['reward'], .9)
        p['action_group'] = 'move'
        self.assertAlmostEqual(rewards.score(json.dumps(p), row, BOOK)['reward'], .7)
        p['category'] = 'No_Gun'
        self.assertEqual(rewards.score(json.dumps(p), row, BOOK)['reward'], 0.)
        p = response(row, False)
        base = rewards.score(json.dumps(p), row, BOOK)
        p['explanation'] = 'A visual explanation. ' * 100
        self.assertEqual(rewards.score(json.dumps(p), row, BOOK), base)

    def test_malformed_duplicate_keys_nonfinite_and_truncation(self):
        row = fixture()
        for text in ('{}', '[]', 'not JSON', '{"category":"Handgun","category":"No_Gun"}',
                     json.dumps({**response(row), 'grounding': [{'frame_id': 0, 'boxes': [[0, 0, float('nan'), 1]]}]})):
            self.assertEqual(rewards.score(text, row, BOOK, grounded=True)['reward'], 0)
        self.assertFalse(rewards.score(json.dumps(response(row)), row, BOOK, True, True)['valid'])

    def test_grounding_matching_extra_missing_duplicate_and_reordering(self):
        row = fixture()
        p = response(row)
        self.assertAlmostEqual(rewards.score(json.dumps(p), row, BOOK, True)['quality'], 1.)
        p['grounding'].reverse()
        self.assertAlmostEqual(rewards.score(json.dumps(p), row, BOOK, True)['reward'], 1.)
        p['grounding'].reverse()
        p['grounding'][0]['boxes'].append([.2, .3, .4, .5])
        self.assertAlmostEqual(rewards.score(json.dumps(p), row, BOOK, True)['quality'], (2 / 3 + 1) / 2)
        p['grounding'][0]['boxes'] = []
        self.assertAlmostEqual(rewards.score(json.dumps(p), row, BOOK, True)['quality'], .5)
        p['grounding'][0]['boxes'] = [[0, 0, 1, 1]]
        self.assertAlmostEqual(rewards.score(json.dumps(p), row, BOOK, True)['quality'], .5)
        p['grounding'][1]['frame_id'] = 0
        self.assertFalse(rewards.score(json.dumps(p), row, BOOK, True)['valid'])

    def test_wrong_frame_boundary_boolean_and_no_gun_false_boxes(self):
        row = fixture()
        for bad in (8, -1, True, '0'):
            p = response(row)
            p['grounding'][0]['frame_id'] = bad
            self.assertFalse(rewards.score(json.dumps(p), row, BOOK, True)['valid'])
        row.update(category='No_Gun', action_group='move', action_code='N1')
        for v in row['views']:
            v['boxes'] = []
        p = response(row)
        self.assertEqual(rewards.score(json.dumps(p), row, BOOK, True)['quality'], 1.)
        p['grounding'][0]['boxes'] = [[0, 0, 1, 1]]
        self.assertEqual(rewards.score(json.dumps(p), row, BOOK, True)['quality'], 7/8)
        p['grounding'][0]['boxes'] = [[0, 0, 1.001, 1]]
        self.assertFalse(rewards.score(json.dumps(p), row, BOOK, True)['valid'])

    def test_unknown_only_has_no_quality_and_wrong_category_cannot_be_rescued(self):
        row = fixture()
        p = response(row)
        p['category'] = 'No_Gun'
        self.assertEqual(rewards.score(json.dumps(p), row, BOOK, True)['reward'], 0.)
        for v in row['views']:
            v['boxes'] = None
        result = rewards.score(json.dumps(response(row)), row, BOOK, True)
        self.assertIsNone(result['quality'])
        self.assertAlmostEqual(result['reward'], 1.)

    def test_unknown_target_spans_and_cross_boundary_tokens_are_masked(self):
        row = fixture()
        target, spans = target_parts(row, True)
        kinds = token_kinds([(i, i+1) for i in range(len(target))], spans)
        null = target.index('null')
        self.assertEqual(kinds[null:null+4], [0] * 4)
        known = target.index('0.2')
        self.assertEqual(kinds[known:known+3], [2] * 3)
        self.assertEqual(token_kinds([(spans[0][1]-1, spans[1][1])], spans), [0])
        self.assertNotIn(row['id'], instruction(BOOK, True))
        self.assertNotIn('null', instruction(BOOK, True))


class StoppingTests(unittest.TestCase):
    def test_anchor_best_ties_and_baseline(self):
        state = StopState()
        def observe(c, g, a, passes, ckpt):
            return state.observe(dict(balanced_category_accuracy=c, action_group_accuracy=g, action_code_accuracy=a), ckpt, passes)
        self.assertTrue(observe(.5, .4, .3, 0, 'base'))
        self.assertEqual(state.checks, 0)
        self.assertTrue(observe(.502, .4, .3, .5, 'a'))
        self.assertEqual(state.anchor, .5)
        self.assertTrue(observe(.504, .4, .3, 1, 'b'))
        self.assertFalse(observe(.504, .4, .3, 1.5, 'c'))
        self.assertEqual(state.reason, 'patience')
        self.assertEqual(state.best_checkpoint, 'b')
        self.assertEqual(StopState(**state.snapshot()).snapshot(), state.snapshot())

    def test_min_delta_resets_and_action_tiebreak(self):
        state = StopState()
        metrics = dict(balanced_category_accuracy=.5, action_group_accuracy=.4, action_code_accuracy=.3)
        state.observe(metrics, 'base', 0)
        state.observe({**metrics, 'balanced_category_accuracy': .506}, 'new', .5)
        self.assertEqual(state.stale, 0)
        state.observe({**metrics, 'balanced_category_accuracy': .506, 'action_group_accuracy': .5}, 'tie', 1)
        self.assertEqual(state.best_checkpoint, 'tie')


class EvaluationAndBudgetTests(unittest.TestCase):
    def test_paired_ids_invalid_and_false_positives(self):
        rows = [{**fixture(), 'id': str(i), 'category': cat} for i, cat in enumerate(data.CATEGORIES)]
        predictions = [{'video': r['id'], 'text': json.dumps(response(r, False)), 'truncated': False} for r in rows]
        measured = metrics(rows, predictions, BOOK)
        self.assertAlmostEqual(measured['balanced_category_accuracy'], 1.)
        predictions[0]['text'] = '{}'
        predictions[2]['text'] = json.dumps(response(fixture(), False))
        measured = metrics(rows, predictions, BOOK)
        self.assertAlmostEqual(measured['balanced_category_accuracy'], 1 / 3)
        self.assertEqual(measured['no_gun_category_false_positive_rate'], 1.)
        self.assertAlmostEqual(measured['invalid_fraction'], 1 / 3)
        with self.assertRaises(ValueError):
            metrics(rows, predictions[:-1], BOOK)
        with self.assertRaises(ValueError):
            metrics(rows, predictions + predictions[:1], BOOK)

    def test_concurrent_reservations_charge_gpu_time_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reserve(root, 'a', 'pilot', 6000, 'gpu6')
            with self.assertRaises(RuntimeError):
                reserve(root, 'b', 'pilot', 6000, 'gpu7')
            finish(root, 'a', 100, 1, 'child_failure')
            reserve(root, 'b', 'pilot', 6000, 'gpu7')
            records = json.loads((root / 'budget.json').read_text())
            self.assertEqual(budget_used(records), 6100)
            with self.assertRaises(RuntimeError):
                reserve(root, 'a', 'pilot', 1, 'gpu6')


class GRPOTests(unittest.TestCase):
    def test_policy_direction_zero_variance_and_reference_kl(self):
        import torch
        from studies.firearm_posttraining.smoke import grpo_loss
        old = torch.tensor([-.5, -.7])
        lp = old.clone().requires_grad_()
        loss, parts = grpo_loss(torch, lp, old, old, torch.tensor(1.))
        loss.backward()
        self.assertTrue(bool((lp.grad < 0).all()))
        self.assertEqual(parts['kl_per_token'], 0.)
        lp = old.clone().requires_grad_()
        loss, _ = grpo_loss(torch, lp, old, old, torch.tensor(0.))
        loss.backward()
        self.assertTrue(bool((lp.grad == 0).all()))
        lp = (old + .1).clone().requires_grad_()
        loss, parts = grpo_loss(torch, lp, old, old, torch.tensor(0.))
        self.assertGreater(parts['kl_per_token'], 0.)
        loss.backward()
        self.assertTrue(bool((lp.grad > 0).all()))


class FreezeAndQuestionTests(unittest.TestCase):
    def test_question_revision_preserves_prior_slots(self):
        old = ['What is held?', 'How are the arms positioned?']
        self.assertTrue(valid_revision(old, ['Which object is held?', old[1]]))
        self.assertTrue(valid_revision(old, old + ['Is the object a firearm?']))
        self.assertFalse(valid_revision(old, old))
        self.assertFalse(valid_revision(old, ['new', 'both changed']))
        self.assertFalse(valid_revision(old, old[:1]))
        self.assertFalse(valid_revision(old, ['x'] * 6))
        self.assertFalse(valid_revision(old, ['x' * 241, old[1]]))

    def test_protocol_and_selection_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td, 'protocol.json')
            data.write_json(p, {'status': 'frozen', 'protocol_hash': 'wrong'})
            with self.assertRaises(ValueError):
                validate(p, td)
            p = Path(td, 'selection.json')
            data.write_json(p, {'status': 'selection_frozen', 'selection_hash': 'wrong'})
            with self.assertRaises(ValueError):
                verify_selection(p, td)

    def test_holdout_guard_runs_before_original_data_is_read(self):
        from studies.firearm_posttraining.holdout import run
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as td:
            p = Path(td, 'bad.json')
            data.write_json(p, {'status': 'selection_frozen'})
            with self.assertRaises(ValueError):
                run(SimpleNamespace(frozen_manifest=str(p), data=td, original_data='/must/not/be/read'))


class StatisticsTests(unittest.TestCase):
    def test_recording_group_bootstrap_and_error_transitions(self):
        import numpy as np
        from studies.firearm_posttraining.statistics import paired_interval, transitions
        rows = [{'category': c, 'recording_group': f'{c}-{g}'} for c in data.CATEGORIES for g in range(30)]
        one, zero = np.ones((3, len(rows))), np.zeros((3, len(rows)))
        result = paired_interval(rows, one, zero, zero, zero, repeats=100)
        self.assertEqual(result['balanced_category_accuracy_difference'], 1.)
        self.assertEqual(result['balanced_category_accuracy_interval'], [1., 1.])
        self.assertEqual(result['no_gun_false_positive_interval'], [0., 0.])
        changed = transitions(rows, [False] * len(rows), [True] * len(rows))
        self.assertEqual(changed['all'], {'recovered': 90, 'broken': 0, 'net': 90})
        with self.assertRaises(ValueError):
            paired_interval(rows, one[:, :-1], zero, zero, zero, repeats=10)


if __name__ == '__main__':
    unittest.main()
