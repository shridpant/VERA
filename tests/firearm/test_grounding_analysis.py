import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from vera_runner.config import ROOT
from studies.firearm_grounding.analysis import analyze
from studies.firearm_grounding.gate import gate

CATS = ('Handgun', 'Machine_Gun', 'No_Gun')


def record(i, split, reliance_shift=0.0, wrong=False):
    cat = CATS[i % 3]
    firearm = cat != 'No_Gun'
    pred = 'No_Gun' if wrong else cat
    r = {'video': f'v{i}', 'split': split, 'category': cat, 'camera': 'C1', 'recording_group': f'g{i // 2}',
         'target': [cat, 'aim', 'PAH1'], 'pred': {'category': pred, 'group': 'aim', 'code': 'PAH1'}, 'answer': 'A gun. Category: x',
         'decision': {'p_true': 0.7}, 'weapon_mentioned': firearm, 'frame_indices': [0, 10, 20, 30, 40, 50, 60, 70],
         'boxed_slots': [1, 2] if firearm else []}
    if firearm:
        lift = 2.0 + 0.1 * i + reliance_shift
        same = {'pred': {'category': pred}}
        r.update(occlusion={'box': {'pred': {'category': 'No_Gun'}}, 'control': same}, lift=lift, lift_shuffled=1.0, lift_sink_excluded=lift * 1.1, frame_lifts={'1': lift, '2': lift},
                 lift_layers={'early': 1.0, 'middle': lift, 'late': lift}, reliance=0.5 + 0.05 * i + reliance_shift)
    return r


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))


class GroundingAnalysisTests(unittest.TestCase):
    def test_tables_and_primary_contrasts_from_fake_runs(self):
        with tempfile.TemporaryDirectory() as td:
            root, data = Path(td) / 'g', Path(td) / 'data'
            phase1 = [record(i, 'train') for i in range(30)] + [record(i, 'test') for i in range(30, 54)]
            write_jsonl(root / 'phase1_seed_measurements.jsonl', phase1)
            gate(root / 'phase1_seed_measurements.jsonl', root / 'gate.json')
            (root / 'phase0').mkdir()
            (root / 'phase0/phase0.json').write_text(json.dumps({'checks': {'c_token_grid_orientation': {'passed': True, 'trials': 10}}}))
            shift = {'T1': 0.0, 'G': 1.0, 'P': 0.0}
            for arm in shift:
                for seed in (1, 2):
                    run = root / 'arms' / f'{arm}_s{seed}'
                    write_jsonl(run / 'test_measurements.jsonl', [record(i, 'test', shift[arm]) for i in range(30, 54)])
                    vals = [{'step': s, 'reference_only': s == 0, 'metrics': {'category_accuracy': 0.5, 'group_accuracy': 0.3,
                                                                               'pooled_lift': 2.0 if arm != 'T1' else None}} for s in (0, 100)]
                    write_jsonl(run / 'validations.jsonl', vals)
                    write_jsonl(run / 'steps.jsonl', [{'step': s, 'status': 'complete', 'update': 'updated'} for s in range(1, 101)])
                    (run / 'selected.json').write_text(json.dumps({'step': 100, 'questions': 'q', 'metrics': vals[1]['metrics']}))
            data.mkdir()
            boxes = {f'v{i}': {'boxes': {'10': [[0, 0, 100, 100]], '20': [[0, 0, 50, 50]]}} for i in range(30, 54)}
            (data / 'boxes_test.json').write_text(json.dumps(boxes))
            out = Path(td) / 'analysis'
            manifest = analyze(root, data, ROOT / 'studies/firearm/annotation_policy.json', out)
            self.assertEqual(manifest['tables'], ['R1', 'R2', 'R2s', 'R3', 'R4', 'R5', 'R6', 'R7'])
            self.assertEqual(set(manifest['figures']), {'F1_phase1_lift_and_reliance', 'F2_arms', 'F3_validation_trajectories'})
            with (out / 'tables/R4.csv').open() as f:
                rows = list(csv.DictReader(f))
            h1, h2 = rows[0], rows[1]
            self.assertAlmostEqual(float(h1['estimate']), 1.0)  # G - P reliance, every video shifted by 1 nat
            self.assertEqual(h1['supported'], 'True')
            self.assertAlmostEqual(float(h2['estimate']), 0.0)
            self.assertEqual(h2['supported'], 'True')
            self.assertEqual(int(h1['videos']), 16)  # No_Gun test videos carry no reliance
            self.assertEqual(int(h2['videos']), 24)
            numbers = json.loads((out / 'numbers.json').read_text())
            self.assertTrue(numbers['gate_passed'])
            self.assertAlmostEqual(numbers['H1_estimate'], 1.0)
            with (out / 'tables/R6.csv').open() as f:
                sizes = list(csv.DictReader(f))
            self.assertEqual({r['size'] for r in sizes if int(r['videos'])}, {'medium'})  # boxes of 1 % and 0.25 %: mean 0.625 %
            with (out / 'tables/R5.csv').open() as f:
                g = next(r for r in csv.DictReader(f) if r['arm'] == 'G')
            self.assertEqual((float(g['flip_box_mask']), float(g['flip_control_mask'])), (1.0, 0.0))
            self.assertIn('G, seed 1 (step 100)', (out / 'questions.tex').read_text())


if __name__ == '__main__':
    unittest.main()
