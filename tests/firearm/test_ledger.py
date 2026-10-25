import json
from pathlib import Path
import tempfile
import unittest
from vera_runner.ledger import JsonlAppender, read_jsonl, seal
from vera_runner.study import reconcile, run_study
from tests.firearm.test_study_run import Recording, make_study


class SealTests(unittest.TestCase):
    def test_a_torn_fragment_is_set_aside_and_appends_stay_readable(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / 'x.jsonl'
            f.write_text('{"a": 1}\n{"b": ')
            out = JsonlAppender(f)
            out.write({'c': 3})
            out.close()
            self.assertEqual(read_jsonl(f)[0], [{'a': 1}, {'c': 3}])
            self.assertEqual((Path(td) / 'x.jsonl.torn.1').read_text(), '{"b": ')

    def test_a_complete_line_without_newline_is_kept(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / 'x.jsonl'
            f.write_text('{"a": 1}')
            self.assertIsNone(seal(f))
            self.assertEqual(f.read_text(), '{"a": 1}\n')

    def test_resume_after_torn_ledger_and_records(self):
        with tempfile.TemporaryDirectory() as td:
            study = make_study(td)
            backend = Recording(study['runner'])
            run_study(study, backend=backend)
            run = Path(td) / 'run'
            with (run / 'ledger.jsonl').open('a') as f:
                f.write('{"t": "2026-10-25", "event": "sta')
            with (run / 'records/G.jsonl').open('a') as f:
                f.write('{"video": "x", "anch')
            self.assertEqual(run_study(study, backend=Recording(study['runner'])), 0)
            self.assertEqual(run_study(study, backend=Recording(study['runner'])), 0)  # and again: no corrupt line
            self.assertEqual(reconcile(study, run)['counts'], {'complete': 15})
            notes = [e for e in read_jsonl(run / 'ledger.jsonl')[0] if e['event'] == 'note']
            self.assertEqual(len(notes), 2)
            self.assertTrue((run / 'records/G.jsonl.torn.1').exists())
