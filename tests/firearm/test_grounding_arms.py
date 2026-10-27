import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from vera_runner.ledger import read_jsonl
from vera_runner.verbalized import sample_indices
from studies.firearm_grounding.arms import BLOCK_HEADER, SENTENCE, GroundedRun, grounded_template, phase1_summary
from tests.firearm.test_verbalized import BOOK, FakeCalls, make_split, spec


class FakeScorer:
    """Attention concentrated on the top-left 4x4 token block of every frame; a Category field always present."""
    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return type('E', (), {'input_ids': [1, 2, 3]})()

    def __init__(self, decides=True):
        self.tok, self.decides = self.Tok(), decides

    def decision(self, ids):
        return (0, ' ', 'Handgun') if self.decides else None

    def query_ids(self, question, n):
        import torch
        return torch.tensor([[0] * 5]), '<|im_end|>'

    def score(self, images, query, ids, attention=True):
        if not self.decides:
            return None
        a = np.zeros((16, 16))
        a[:4, :4] = 1
        a = (a / a.sum()).reshape(-1)
        return {'frame_attention': {k: a.tolist() for k in range(8)}, 'category_p': {}}


def boxes_for(data):
    out = {}
    for s in ('train', 'val', 'test'):
        for r in json.loads((data / f'{s}.json').read_text()):
            if r['category'] == 'Handgun':
                idx = sample_indices(r['frames_decoded'])
                out[r['id']] = {'statuses': ['positive'] * r['frames_decoded'], 'boxes': {str(i): [[0, 0, 250, 250]] for i in idx}}
    return out


PHASE1 = {'train_video_frame_lifts': [[0.5, 1.0], [2.0, 4.0, 8.0]], 'train_frame_lifts': [0.5, 1.0, 2.0, 4.0, 8.0],
          'train_video_mean_frame_lift_quartiles': [1.0, 3.0]}


class ArmTests(unittest.TestCase):
    def run_arm(self, td, arm, scorer=None):
        data = make_split(td)
        run_dir = Path(td) / arm
        run_dir.mkdir()
        run = GroundedRun(spec(), run_dir, data, Path(td) / 'frames', FakeCalls(), BOOK, scorer or FakeScorer(), boxes_for(data), arm, PHASE1)
        self.assertEqual(run.train('001'), 'complete')
        return run, read_jsonl(run_dir / 'steps.jsonl')[0], read_jsonl(run_dir / 'validations.jsonl')[0]

    def test_t1_keeps_veras_optimizer_prompt(self):
        with tempfile.TemporaryDirectory() as td:
            run, steps, _ = self.run_arm(td, 'T1')
            self.assertNotIn(BLOCK_HEADER, steps[0]['optimizer_prompt'])
            self.assertNotIn('grounding_block', steps[0])

    def test_grounded_block_reports_the_measured_lift(self):
        with tempfile.TemporaryDirectory() as td:
            run, steps, vals = self.run_arm(td, 'G')
            block = '\n'.join(s['grounding_block'] for s in steps)
            self.assertIn(BLOCK_HEADER, steps[0]['optimizer_prompt'])
            self.assertIn(SENTENCE, steps[0]['optimizer_prompt'])
            self.assertIn('Frame1 16.0', block)  # all attention inside a box covering 1/16 of the frame
            self.assertIn('overall high', block)
            self.assertIn('no firearm in the frames', block)
            self.assertAlmostEqual(vals[1]['metrics']['pooled_lift'], 16.0)

    def test_placebo_draws_one_training_video_per_video_with_a_seeded_generator(self):
        with tempfile.TemporaryDirectory() as td, tempfile.TemporaryDirectory() as td2:
            _, steps, _ = self.run_arm(td, 'P')
            _, again, _ = self.run_arm(td2, 'P')
            drawn = [set(g['frames'].values()) for s in steps for g in s['grounding_values'] if g['frames']]
            self.assertTrue(drawn)
            self.assertTrue(all(any(d <= set(v) for v in PHASE1['train_video_frame_lifts']) for d in drawn))
            self.assertEqual([s['grounding_block'] for s in steps], [s['grounding_block'] for s in again])

    def test_grounded_and_placebo_report_a_missing_decision_alike(self):
        for arm in ('G', 'P'):
            with tempfile.TemporaryDirectory() as td:
                _, steps, _ = self.run_arm(td, arm, FakeScorer(decides=False))
                block = '\n'.join(s['grounding_block'] for s in steps)
                self.assertIn('no decision to measure', block, arm)
                self.assertNotIn('Frame1', block, arm)

    def test_selection_prefers_lift_among_accuracy_eligible_steps(self):
        with tempfile.TemporaryDirectory() as td:
            run, _, _ = self.run_arm(td, 'G')
            path = Path(td) / 'G' / 'validations.jsonl'
            vals = read_jsonl(path)[0]
            vals[1]['metrics'].update(category_accuracy=0.80, pooled_lift=1.0)
            vals[2]['metrics'].update(category_accuracy=0.77, pooled_lift=3.0)
            vals[3]['metrics'].update(category_accuracy=0.60, pooled_lift=9.0)
            vals[4]['metrics'].update(category_accuracy=0.50, pooled_lift=20.0)  # best lift, but not accuracy-eligible
            path.write_text(''.join(json.dumps(v) + '\n' for v in vals))
            self.assertEqual(run.select()['step'], vals[2]['step'])

    def test_template_and_pool_helpers(self):
        t = grounded_template('x\n** The targets: **\n[$GroundTruth]\nrest')
        self.assertIn('[$Grounding]', t)
        summary = phase1_summary([{'split': 'train', 'frame_lifts': {'0': 1.0, '1': 3.0}}, {'split': 'val', 'frame_lifts': {'0': 9.0}}])
        self.assertEqual(summary['train_frame_lifts'], [1.0, 3.0])
        self.assertEqual(summary['train_video_frame_lifts'], [[1.0, 3.0]])
