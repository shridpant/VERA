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
from studies.firearm_posttraining.execute import budget_used, reserve, finish, validate_duration
from studies.firearm_posttraining.verbalized import valid_revision, decode_proposal
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

    def test_only_unknown_payload_tokens_are_masked_and_known_schema_boundaries_survive(self):
        row = fixture()
        target, spans = target_parts(row, True)
        kinds = token_kinds([(i, i+1) for i in range(len(target))], spans)
        null = target.index('null')
        self.assertEqual(kinds[null:null+4], [0] * 4)
        known = target.index('0.2')
        self.assertEqual(kinds[known:known+3], [2] * 3)
        self.assertEqual(token_kinds([(spans[0][1]-1, spans[1][1])], spans), [1])
        self.assertEqual(token_kinds([(null-1, null+2)], spans), [0])
        for i, view in enumerate(row['views']):
            frame = target.index('{"frame_id":' + str(view['frame_id']))
            self.assertEqual(kinds[frame], 1)
        self.assertEqual(token_kinds([(0, 2)], [(0, 1, 1), (1, 2, 2)]), [2])
        self.assertEqual(token_kinds([(0, 3)], [(0, 1, 1), (2, 3, 1)]), [0])
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
            reserve(root, 'a', 'pilot', 12000, 'gpu6')
            with self.assertRaises(RuntimeError):
                reserve(root, 'b', 'pilot', 12000, 'gpu7')
            finish(root, 'a', 100, 1, 'child_failure')
            reserve(root, 'b', 'pilot', 12000, 'gpu7')
            records = json.loads((root / 'budget.json').read_text())
            self.assertEqual(budget_used(records), 12100)
            with self.assertRaises(RuntimeError):
                reserve(root, 'a', 'pilot', 60, 'gpu6')

    def test_classification_is_independent_of_grounding_schema(self):
        rows = [{**fixture(), 'id': str(i), 'category': c} for i, c in enumerate(data.CATEGORIES)]
        predictions = []
        for row in rows:
            value = response(row)
            value['grounding'] = 'invalid box schema'
            predictions.append({'video': row['id'], 'text': json.dumps(value), 'truncated': False})
            self.assertEqual(rewards.score(json.dumps(value), row, BOOK, True)['reward'], 0.)
        result = metrics(rows, predictions, BOOK, grounded=True)
        self.assertEqual(result['balanced_category_accuracy'], 1.)
        self.assertEqual(result['invalid_fraction'], 1.)


    def test_real_runs_have_no_time_cap_but_pilots_remain_bounded(self):
        for stage in ('sweep', 'evaluation'):
            validate_duration(stage, 0)
            with self.assertRaises(ValueError):
                validate_duration(stage, 9000)
        for bad in (0, 30, 6 * 3600 + 1):
            with self.assertRaises(ValueError):
                validate_duration('pilot', bad)
        validate_duration('pilot', 4 * 3600)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            reserve(root, 'real', 'sweep', 0, 'gpu6')
            records = json.loads((root / 'budget.json').read_text())
            self.assertIsNone(records['real']['reserved_seconds'])
            self.assertGreaterEqual(budget_used(records), 0)
            finish(root, 'real', 1000 * 3600, 0, 'completed')
            reserve(root, 'next', 'evaluation', 0, 'gpu7')
            # Recorded scientific GPU use cannot exhaust a removed aggregate cap.
            reserve(root, 'pilot', 'pilot', 4 * 3600, 'gpu6')


class GRPOTests(unittest.TestCase):
    def test_sampling_likelihoods_match_full_support_temperature_policy(self):
        import torch
        from types import SimpleNamespace
        from studies.firearm_posttraining.sampling import generation_settings, policy_statistics
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2, bos_token_id=1)
        settings = generation_settings(tokenizer, True, 512)
        self.assertEqual((settings['temperature'], settings['top_p'], settings['top_k']), (.8, 1., 0))
        self.assertFalse(generation_settings(tokenizer, False, 512)['do_sample'])
        logits = torch.tensor([[[0., 1., 2.], [2., 0., -1.]]], requires_grad=True)
        tokens = torch.tensor([[2, 0]])
        selected, entropy = policy_statistics(logits, tokens)
        expected = (logits / settings['temperature']).log_softmax(-1)
        self.assertTrue(torch.allclose(selected, expected.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)))
        self.assertTrue(torch.allclose(entropy, -(expected.exp() * expected).sum(-1)))
        self.assertFalse(torch.allclose(selected, logits.log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)))
        from studies.firearm_posttraining.model import NativeModel
        native = NativeModel.__new__(NativeModel)
        native.logits = lambda prompt, ids: (logits, tokens)
        actual, actual_entropy = native.log_probs({}, [2, 0])
        self.assertTrue(torch.allclose(actual, selected[0]))
        self.assertTrue(torch.allclose(actual_entropy, entropy[0].mean()))
        (-selected.mean()).backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))

    def test_generation_does_not_inherit_checkpoint_sampling_filters(self):
        import torch
        from transformers import GenerationConfig
        from types import SimpleNamespace
        from studies.firearm_posttraining.model import NativeModel
        captured = {}
        def generate(**kwargs):
            captured.update(kwargs)
            return torch.tensor([[1, 3, 4, 2]])
        native = NativeModel.__new__(NativeModel)
        native.torch = torch
        native.model = SimpleNamespace(eval=lambda: None, generate=generate,
            generation_config=GenerationConfig(temperature=.7, top_k=20, top_p=.8, do_sample=True))
        native.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2, bos_token_id=1,
                                           decode=lambda ids, **kwargs: '{}')
        result = native.generate({'input_ids': torch.tensor([[1, 3]])}, sample=True)
        actual = captured['generation_config']
        self.assertEqual((actual.temperature, actual.top_p, actual.top_k, actual.num_beams), (.8, 1., 0, 1))
        self.assertIsNot(actual, native.model.generation_config)
        self.assertEqual(result['ids'], [4, 2])
        self.assertFalse(result['truncated'])

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
        self.assertEqual(decode_proposal(json.dumps({'slot': 0, 'question': 'New question?'}), old), ['New question?', old[1]])
        self.assertIsNone(decode_proposal(json.dumps({'slot': True, 'question': 'New?'}), old))
        self.assertIsNone(decode_proposal(json.dumps({'slot': 3, 'question': 'New?'}), old))

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
        self.assertGreater(result['no_gun_noninferiority_upper_screen'], .05)
        strong = paired_interval(rows, one, zero, zero, one, repeats=100)
        self.assertLess(strong['no_gun_noninferiority_upper_screen'], 0.)
        changed = transitions(rows, [False] * len(rows), [True] * len(rows))
        self.assertEqual(changed['all'], {'recovered': 90, 'broken': 0, 'net': 90})
        with self.assertRaises(ValueError):
            paired_interval(rows, one[:, :-1], zero, zero, zero, repeats=10)

    def test_localization_inverse_view_unknowns_and_false_boxes(self):
        from studies.firearm_posttraining.report import localization
        row = fixture()
        row['geometry'] = data.geometry(640, 480)
        p = response(row)
        pred = [{'video': row['id'], 'text': json.dumps(p), 'truncated': False}]
        result = localization([row], pred, BOOK)
        self.assertEqual(result['precision_IoU50'], 1.)
        self.assertEqual(result['recall_IoU50'], 1.)
        self.assertEqual(result['annotation_coverage'], .25)
        p['grounding'][0]['boxes'] = [[0, 0, 1, 1]]
        pred[0]['text'] = json.dumps(p)
        result = localization([row], pred, BOOK)
        self.assertEqual(result['precision_IoU50'], .5)
        self.assertEqual(result['whole_image_box_rate'], .5)


class OrchestrationTests(unittest.TestCase):
    def test_calibration_grid_is_finite_and_leaves_scientific_continuation(self):
        from studies.firearm_posttraining.calibrate import validate_grid
        self.assertEqual(validate_grid([3.]), (3.,))
        for values in ([], [float('nan')], [float('inf')], [5.], [.25], [1.25], [2., 1.], [1., 1.]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_grid(values)

    def test_idle_cooldown_never_waits_on_an_occupied_gpu(self):
        from unittest.mock import patch
        from studies.firearm_posttraining.execute import idle_state
        idle = {'processes': [], 'memory_used_mib': 4, 'utilization_pct': 0}
        with patch('studies.firearm_posttraining.execute.gpu_state', side_effect=[{**idle, 'utilization_pct': 100}, idle]), \
             patch('studies.firearm_posttraining.execute.time.sleep') as sleep:
            self.assertEqual(idle_state('device'), idle)
            sleep.assert_called_once_with(2)
        with patch('studies.firearm_posttraining.execute.gpu_state', return_value={**idle, 'processes': ['other']}), \
             patch('studies.firearm_posttraining.execute.time.sleep') as sleep:
            with self.assertRaises(RuntimeError):
                idle_state('device')
            sleep.assert_not_called()

    def test_calibration_uses_fit_only_candidates_and_never_scales_failed_reward_gate(self):
        from unittest.mock import patch
        from studies.firearm_posttraining.calibrate import run
        for reject_first_format, reject_reward, prelaunch_failure in ((True, False, False), (False, True, False), (False, False, True), (False, False, False)):
            with self.subTest(format_retry=reject_first_format, reward_failure=reject_reward), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                namespaced = not any((reject_first_format, reject_reward, prelaunch_failure))
                options = {'control_name': 'calibration-v2', 'attempt_offset': 7, 'warmup_grid': [3.], 'warmup_attempt': 8,
                           'grounded_smoke': root / 'smoke-grounded-new/result.json'} if namespaced else {}
                data.write_json(root / 'data/preflight.json', {'model': 'pinned'})
                data.write_json(root / 'data/fit.json', [{}] * 192)
                data.write_json(root / 'data/stop.json', [{}] * 48)
                data.write_json(root / 'data/audit.json', {'source_counts': {'val': 62, 'test_metadata': 96}})
                data.write_json(root / 'pilot-sl-001/result.json', {'status': 'stopped'})
                data.write_json(root / 'pilot-sl-001/stop-initial/metrics.json', {'latency_seconds_per_video': 1.3})
                commands = []
                def fake_run(command, **kwargs):
                    commands.append(command)
                    name = command[command.index('--name') + 1]
                    if prelaunch_failure and name == 'pilot-c0l-002' and sum('--name' in c and c[c.index('--name') + 1] == name for c in commands) == 1:
                        raise RuntimeError('simulated idle check before job creation')
                    directory = Path(command[command.index('--output') + 1])
                    directory.mkdir(parents=True)
                    data.write_json(root / 'jobs' / name / 'status.json', {'exit_code': 0})
                    grounded = '--grounded' in command
                    spec = {'grounded': grounded}
                    for flag in ('--model', '--data', '--adapter'):
                        if flag in command:
                            spec[flag[2:]] = command[command.index(flag) + 1]
                    data.write_json(directory / 'spec.json', spec)
                    if '--warmup-only' in command:
                        passes = float(command[command.index('--warmup-passes') + 1])
                        adapter = directory / 'checkpoints/fixed'
                        data.write_json(adapter / 'state.json', {'cursor': int(192 * passes), 'selection': {'checks': 0}})
                        data.write_json(directory / 'spec.json', {'model': 'pinned', 'warmup_passes': passes, 'grounded': grounded,
                                                               'code_commit': 'audited-reference'})
                        result = {'status': 'stopped', 'stop_reason': 'fixed_warmup_complete', 'fit_passes': passes, 'best_checkpoint': str(adapter), 'elapsed_seconds': 30.}
                    elif 'studies.firearm_posttraining.fit_format' in command:
                        fail = reject_first_format and grounded and name.endswith('002')
                        result = {'status': 'failed_gate' if fail else 'passed', 'elapsed_seconds': 20., 'responses': 96}
                    else:
                        result = {'status': 'failed_gate' if reject_reward and grounded else 'passed', 'elapsed_seconds': 40., 'groups': 24}
                    data.write_json(directory / 'result.json', result)
                def fake_pipeline(*args):
                    data.write_json(root / 'pipeline/status.json', {'stage': 'complete'})
                if namespaced:
                    # Completed SFT artifacts exist before the new probability
                    # validation; no old freeze/forecast has been produced.
                    for kind in ('l', 'g'):
                        name = f'pilot-c0{kind}-008'
                        command = ['python', '--name', name, '--output', str(root / name), '--model', 'pinned',
                                   '--data', str(root / 'data'), '--warmup-only', '--warmup-passes', '3', '--seconds', '1800']
                        if kind == 'g':
                            command.append('--grounded')
                        fake_run(command)
                    data.write_json(root / 'data/target-mask-audit-v3.json', {'status': 'passed',
                                    'sft_training_source_unchanged': True, 'sft_reference_commit': 'audited-reference'})
                    prior_warmups = sum('--warmup-only' in c for c in commands)
                with patch('studies.firearm_posttraining.calibrate.subprocess.run', side_effect=fake_run), \
                     patch('studies.firearm_posttraining.calibrate.freeze', return_value={'protocol_hash': 'frozen'}) as freeze, \
                     patch('studies.firearm_posttraining.calibrate.pipeline', side_effect=fake_pipeline) as pipeline:
                    if prelaunch_failure:
                        with self.assertRaisesRegex(RuntimeError, 'simulated idle'):
                            run(root, 'original', 'frames')
                        run(root, 'original', 'frames', resume=True)
                        self.assertEqual(len(list((root / 'calibration').glob('pilot-c0l-002.prelaunch-*.log'))), 1)
                        self.assertEqual(sum(c[c.index('--name') + 1] == 'pilot-c0g-002' for c in commands), 1)
                    else:
                        run(root, 'original', 'frames', **options)
                    if reject_reward:
                        prior_commands = len(commands)
                        run(root, 'original', 'frames', resume=True)
                        self.assertEqual(len(commands), prior_commands)
                final = json.loads((root / ('calibration-v2' if namespaced else 'calibration') / 'status.json').read_text())
                if namespaced:
                    self.assertFalse((root / 'calibration').exists())
                    self.assertTrue(any(c[c.index('--name') + 1] == 'pilot-c0g-008' for c in commands))
                    self.assertEqual(freeze.call_args.args[3], root / 'smoke-grounded-new/result.json')
                    with self.assertRaisesRegex(ValueError, 'immutable artifact differs'):
                        run(root, 'original', 'frames', resume=True, **{**options, 'warmup_grid': [3.5]})
                    self.assertEqual(sum('--warmup-only' in c for c in commands), prior_warmups)
                    self.assertTrue(any(c[c.index('--name') + 1] == 'format-g-009' for c in commands))
                    self.assertTrue(any(c[c.index('--name') + 1] == 'pilot-rg-009' for c in commands))
                if reject_reward:
                    self.assertEqual(final['stage'], 'failed_real_reward_gate')
                    freeze.assert_not_called()
                    pipeline.assert_not_called()
                else:
                    self.assertEqual(final['stage'], 'complete')
                    self.assertEqual(freeze.call_args.kwargs['warmup_passes'], 3. if namespaced else (1.5 if reject_first_format else 1.))
                    self.assertEqual(freeze.call_args.kwargs['max_passes'], 5.)
                    if reject_first_format:
                        self.assertFalse(any('pilot-rg-002' in command for command in commands))
                self.assertTrue(all(int(c[c.index('--seconds') + 1]) > 0 for c in commands))

    def test_sweep_clones_identical_fixed_warm_starts_and_preserves_scientific_stops(self):
        from unittest.mock import patch
        from studies.firearm_posttraining.sweep import run
        commands = []
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            protocol = root / 'protocol.json'
            protocol.write_text('{}')
            def fake_run(command, **kwargs):
                commands.append(command)
                directory = Path(command[command.index('--output') + 1])
                directory.mkdir(parents=True)
                warm = '--warmup-only' in command
                seed = int(command[command.index('--seed') + 1])
                method = command[command.index('--method') + 1] if '--method' in command else 'VQ'
                checkpoint = directory / 'checkpoints' / 'final-000096'
                checkpoint.mkdir(parents=True)
                (checkpoint / 'adapter_model.safetensors').write_bytes(f'{seed}:{method}'.encode())
                data.write_json(directory / 'result.json', {'status': 'stopped',
                    'stop_reason': 'fixed_warmup_complete' if warm else 'patience',
                    'fit_passes': 1. if warm else 2., 'best_checkpoint': str(checkpoint), 'elapsed_seconds': 20.})
            with patch('studies.firearm_posttraining.sweep.validate', return_value={'model': 'pinned', 'shared_max_fit_passes': 5, 'paired_warmup_fit_passes': 1.}), \
                 patch('studies.firearm_posttraining.sweep.subprocess.run', side_effect=fake_run), \
                 patch('studies.firearm_posttraining.sweep.freeze_selection', return_value={'selection_hash': 'frozen'}) as freeze:
                run(root, protocol)
            self.assertEqual(len(commands), 21)
            self.assertEqual(len(freeze.call_args.args[2]), 15)
            for command in commands:
                self.assertEqual(command[command.index('--seconds') + 1], '0')
                self.assertEqual(command[command.index('--max-seconds') + 1], '0')
                if '--warmup-passes' in command:
                    self.assertEqual(command[command.index('--warmup-passes') + 1], '1.0')
                if '--adapter' in command:
                    method = command[command.index('--method') + 1]
                    seed = int(command[command.index('--seed') + 1])
                    kind = 'G' if method in ('SG', 'RG') else 'L'
                    init = json.loads((root / 'sweep' / f'paired-initializations-{seed}.json').read_text())[kind]
                    self.assertEqual(command[command.index('--adapter') + 1], init['adapter'])
                    self.assertEqual(command[command.index('--max-passes') + 1], '5')

    def test_vq_optimizer_never_receives_stop_rows_or_targets(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from studies.firearm_posttraining.verbalized import run
        from studies.firearm_posttraining.model import DEFAULT_QUESTIONS
        fit = [{**fixture(), 'id': f'fit-{i}', 'category': data.CATEGORIES[i % 3]} for i in range(12)]
        stop = [{**fixture(), 'id': 'STOP_SECRET', 'action_code': 'N12'}]
        optimizer_ids = []
        class FakeNative:
            def __init__(self, *args, **kwargs):
                self.book = BOOK
                self.model = SimpleNamespace(parameters=lambda: [])
            def adapter_hash(self):
                return 'unchanged'
            def prompt(self, row, **kwargs):
                self.assert_fit(row)
                return row, ''
            def assert_fit(self, row):
                assert row['id'].startswith('fit-')
            def generate(self, prompt, **kwargs):
                if prompt == 'optimizer':
                    text = json.dumps({'slot': 0, 'question': 'Describe the held object precisely?'})
                else:
                    text = json.dumps(response(prompt, False))
                return {'text': text, 'ids': [1], 'truncated': False, 'generated_tokens': 1}
            def encode(self, rows, text):
                assert 'STOP_SECRET' not in text
                for row in rows:
                    self.assert_fit(row)
                optimizer_ids.extend(r['id'] for r in rows)
                return 'optimizer', text
        measured = dict(balanced_category_accuracy=.5, action_group_accuracy=.4, action_code_accuracy=.3)
        def fake_evaluate(native, rows, *args, **kwargs):
            self.assertEqual(rows, stop)
            return dict(measured)
        with tempfile.TemporaryDirectory() as td:
            args = SimpleNamespace(frozen_protocol='frozen', data=td, model='pinned', output=str(Path(td, 'run')),
                codebook='studies/firearm_training/codebook.json', seed=20261026, max_seconds=0)
            with patch('studies.firearm_posttraining.verbalized.validate'), \
                 patch('studies.firearm_posttraining.verbalized.provenance'), \
                 patch('studies.firearm_posttraining.verbalized.NativeModel', FakeNative), \
                 patch('studies.firearm_posttraining.verbalized.load_split', side_effect=lambda _, s: fit if s == 'fit' else stop), \
                 patch('studies.firearm_posttraining.verbalized.evaluate', side_effect=fake_evaluate), \
                 patch('studies.firearm_posttraining.verbalized.resource_metrics', return_value={}):
                result = run(args)
            self.assertEqual(result['stop_reason'], 'three_nonimproving_candidates')
            self.assertEqual(result['STOP_checks'], 3)
            self.assertEqual(result['FIT_prediction_queries'], 36)
            self.assertEqual(len(optimizer_ids), 6)


if __name__ == '__main__':
    unittest.main()
