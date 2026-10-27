import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from studies.firearm_grounding.gate import gate


def rec(i, lift, shuffled, reliance, correct=True):
    cat = 'Handgun' if i % 2 else 'Machine_Gun'
    return {'video': f'v{i}', 'category': cat, 'recording_group': f'g{i}', 'pred': {'category': cat if correct else 'No_Gun'},
            'decision': {'p_true': 0.5}, 'lift': lift + 5.0, 'lift_shared': lift, 'lift_shuffled': shuffled, 'reliance': reliance}


class GateTests(unittest.TestCase):
    def run_gate(self, records):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'm.jsonl'
            p.write_text(''.join(json.dumps(r) + '\n' for r in records))
            return gate(p)

    def test_specific_and_agreeing_measures_pass(self):
        rng = np.random.default_rng(0)
        recs = [rec(i, 2 + i * 0.1, 1.0 + rng.normal(0, 0.05), 0.1 * i + rng.normal(0, 0.05)) for i in range(30)]
        result = self.run_gate(recs)
        self.assertTrue(result['V1']['passed'] and result['V2']['passed'] and result['gate_passed'])

    def test_v1_pairs_on_the_shared_frames(self):
        rng = np.random.default_rng(2)
        recs = [rec(i, 1.0 + rng.normal(0, 0.1), 1.0 + rng.normal(0, 0.1), 0.1 * i) for i in range(30)]
        result = self.run_gate(recs)  # the all-frame lift is 5 higher, but the shared-frame lift is not
        self.assertFalse(result['V1']['passed'])
        self.assertGreater(result['V1']['mean_lift_true'] - result['V1']['mean_lift_true_shared_frames'], 4.9)

    def test_incomplete_phase1_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'm.jsonl'
            p.write_text(json.dumps(rec(0, 2, 1, 0.1)) + '\n')
            with self.assertRaises(ValueError):
                gate(p, expected=['v0', 'v1'])

    def test_unspecific_attention_fails_the_gate(self):
        rng = np.random.default_rng(1)
        recs = [rec(i, 1.0 + rng.normal(0, 0.1), 1.0 + rng.normal(0, 0.1), rng.normal(0, 1)) for i in range(30)]
        result = self.run_gate(recs)
        self.assertFalse(result['gate_passed'])
