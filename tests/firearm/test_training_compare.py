import csv
import json
from pathlib import Path
import tempfile
import unittest
from vera_runner.verbalized import Run
from vera_runner.verbalized_variants import BankRun, FixedRun
from studies.firearm_training.analysis.compare import compare
from tests.firearm.test_verbalized import BOOK
from tests.firearm.test_verbalized_variants import BANK, FIXED, FakeCalls, make_split, spec


class CompareTests(unittest.TestCase):
    def test_compare_faithful_and_variants(self):
        with tempfile.TemporaryDirectory() as td:
            data = make_split(td)
            dirs = []
            for name, method, cls, extra in (('faithful', None, Run, {}), ('fixed', 'vera_fixed', FixedRun, FIXED),
                                             ('bank', 'bank', BankRun, BANK), ('random', 'bank_random', BankRun, BANK)):
                s = spec(method or 'vera', **extra)
                if method is None:
                    s['training'].pop('method')
                    s['training']['val_check_interval'] = 3
                run_dir = Path(td) / name
                run_dir.mkdir()
                (run_dir / 'spec.json').write_text(json.dumps({**s, 'experiment_id': name, 'spec_sha256': 'x'}))
                run = cls(s, run_dir, data, Path(td) / 'frames', FakeCalls(), BOOK)
                run.train('001')
                run.test('002')
                dirs.append(run_dir)
            out = Path(td) / 'compare'
            result = compare(dirs, out)
            self.assertEqual(result['tables'], ['C1', 'C2', 'C3', 'C4'])
            self.assertIn('G1_validation_all_runs', result['figures'])
            with (out / 'tables/C1.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(r['seed_test_disagreements'] == '0' for r in rows))  # deterministic fake
            with (out / 'tables/C4.csv').open() as f:
                c4 = {r['run'].rsplit(' (', 1)[0]: r for r in csv.DictReader(f)}
            self.assertEqual(set(c4), {'VERA (faithful)', 'VERA, fixed', 'Question bank', 'Bank, random pairs'})
            self.assertIn('lstlisting', (out / 'selected_questions.tex').read_text())
