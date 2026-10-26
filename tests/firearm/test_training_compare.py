import csv
import json
from pathlib import Path
import tempfile
import unittest
from vera_runner.verbalized import Run
from vera_runner.verbalized_variants import BankRun, FixedRun
from studies.firearm_training.analysis.compare import compare, decide
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


def fake_run(root, name, method, val, finished=True, seed_val=0.5):
    d = Path(root) / name
    d.mkdir()
    (d / 'spec.json').write_text(json.dumps({'experiment_id': name, 'training': {'method': method, 'seed': 1}}))
    vals = [{'step': 0, 'reference_only': True, 'metrics': {'category_accuracy': seed_val, 'group_accuracy': 0.3}, 'predictions': []}]
    vals += [{'step': 20 * (i + 1), 'reference_only': False, 'metrics': {'category_accuracy': c, 'group_accuracy': g}, 'predictions': []}
             for i, (c, g) in enumerate(val)]
    (d / 'validations.jsonl').write_text(''.join(json.dumps(v) + '\n' for v in vals))
    if finished:
        (d / 'test.jsonl').write_text(json.dumps({'name': 'seed', 'predictions': []}) + '\n' + json.dumps({'name': 'learned', 'predictions': []}) + '\n')
    return d


class DecideTests(unittest.TestCase):
    def test_winner_queue_and_write_once(self):
        with tempfile.TemporaryDirectory() as td:
            fixed = fake_run(td, 'fixed', 'vera_fixed', [(0.5, 0.2), (0.6, 0.1)])
            bank = fake_run(td, 'bank', 'bank', [(0.6, 0.3), (0.4, 0.4)])
            out = Path(td) / 'decision.json'
            d = decide([('c/training1b_fixed.json', fixed), ('c/training1c_bank.json', bank)], out)
            self.assertEqual(d['winner'], 'c/training1c_bank.json')  # tie on category, bank wins on action group
            self.assertTrue(d['successful'])
            self.assertEqual(d['queue'], ['c/training1c_bank_seed2.json', 'c/training1c_bank_seed3.json',
                                          'c/training1b_fixed_seed2.json', 'c/training1b_fixed_seed3.json'])
            (bank / 'validations.jsonl').write_text('')  # a later call must not re-decide
            self.assertEqual(decide([('c/training1b_fixed.json', fixed), ('c/training1c_bank.json', bank)], out), d)

    def test_waits_then_decides_among_finished(self):
        with tempfile.TemporaryDirectory() as td:
            fixed = fake_run(td, 'fixed', 'vera_fixed', [(0.4, 0.2)])
            bank = fake_run(td, 'bank', 'bank', [(0.9, 0.9)], finished=False)
            t = [0.0]
            sleeps = []
            d = decide([('training1b_fixed.json', fixed), ('training1c_bank.json', bank)], Path(td) / 'd.json', deadline=900,
                       clock=lambda: t[0], sleep=lambda s: (sleeps.append(s), t.__setitem__(0, t[0] + s)))
            self.assertEqual(len(sleeps), 3)
            self.assertEqual(d['winner'], 'training1b_fixed.json')
            self.assertFalse(d['successful'])  # 0.4 is below the seed questions' 0.5
            self.assertEqual(d['queue'], ['training1b_fixed_seed2.json', 'training1b_fixed_seed3.json'])
            self.assertFalse(d['candidates'][1]['finished'])
