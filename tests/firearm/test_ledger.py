import json
from pathlib import Path
import tempfile
import unittest
from vera_runner.ledger import JsonlAppender, Ledger, read_jsonl, seal


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

    def test_ledger_counts_attempts_across_reopen(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = Ledger(td)
            self.assertEqual(ledger.start('u', 's1'), 1)
            ledger.end('u', 1, 's1', 'failed')
            ledger.close()
            again = Ledger(td)
            self.assertEqual(again.start('u', 's2'), 2)
            again.close()
            self.assertEqual([e['event'] for e in read_jsonl(Path(td) / 'ledger.jsonl')[0]], ['start', 'end', 'start'])
