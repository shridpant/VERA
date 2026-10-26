import csv
import json
from pathlib import Path
import tempfile
import unittest
from vera_runner.config import ROOT
from vera_runner.verbalized import Run
from studies.firearm_training.analysis.run import analyze
from tests.firearm.test_verbalized import BOOK, FakeCalls, make_split, spec


class TrainingAnalysisTests(unittest.TestCase):
    def test_tables_figures_numbers_from_a_fake_run(self):
        with tempfile.TemporaryDirectory() as td:
            data = make_split(td)
            summary = {'videos': 4, 'recording_groups': 4, 'action_codes': 2, 'category': {'Handgun': 2, 'No_Gun': 2},
                       'action_group': {'aim': 2, 'move': 2}, 'camera': {'C1': 4}, 'subject': {'V1': 4}}
            (data / 'splits.json').write_text(json.dumps({'splits': {s: summary for s in ('train', 'val', 'test')}}))
            run_dir = Path(td) / 'run'
            run_dir.mkdir()
            s = spec()
            (run_dir / 'spec.json').write_text(json.dumps({**s, 'spec_sha256': 'x'}))
            run = Run(s, run_dir, data, Path(td) / 'frames', FakeCalls(), BOOK)
            run.train('001')
            run.test('002')
            (run_dir / 'sessions/001').mkdir(parents=True)
            (run_dir / 'sessions/001/session.json').write_text(json.dumps({'model_load_seconds': 20.0, 'gpu': 'GPU-x, RTX 6000 Ada'}))
            out = Path(td) / 'analysis'
            manifest = analyze(run_dir, data, ROOT / 'studies/firearm_training/codebook.json', out)
            self.assertEqual(manifest['tables'], ['S1', 'S10', 'S2', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9'])
            self.assertEqual(set(manifest['figures']), {'F1_validation_trajectory', 'F2_category_confusion', 'F3_accuracy_by_category', 'F4_runtime'})
            with (out / 'tables/S4.csv').open() as f:
                rows = {r['questions']: r for r in csv.DictReader(f)}
            self.assertEqual(float(rows['seed']['category_acc']), 1.0)  # the fake always answers correctly
            self.assertEqual(float(rows['learned - seed']['category_acc']), 0.0)
            with (out / 'tables/S5.csv').open() as f:
                conf = list(csv.DictReader(f))
            self.assertEqual(sum(int(r['videos']) for r in conf if r['questions'] == 'learned'), 4)
            self.assertIn(r'\csname val@test_learned_category_acc_pct', (out / 'numbers.tex').read_text())
            self.assertTrue((out / 'questions_selected.txt').read_text())
            self.assertEqual(len(json.loads((out / 'qualitative.json').read_text())), 2)  # a correct case for each true category present
