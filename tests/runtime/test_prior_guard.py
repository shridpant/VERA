import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime,timezone
from tests.real_model.guard import prior_run_blocker

class PriorRunGuardTests(unittest.TestCase):
    def state(self,root,**kwargs):
        data={'pid':123456,'status':'running','heartbeat_at':datetime.now(timezone.utc).isoformat()}
        data.update(kwargs);p=Path(root)/'status.json';p.write_text(json.dumps(data));return p

    def test_dead_process_with_fresh_heartbeat_does_not_block(self):
        with tempfile.TemporaryDirectory() as td,patch('tests.real_model.guard.os.kill',side_effect=ProcessLookupError):
            self.assertIsNone(prior_run_blocker(self.state(td)))

    def test_process_access_denied_is_unknown_not_claimed_running(self):
        with tempfile.TemporaryDirectory() as td,patch('tests.real_model.guard.os.kill',side_effect=PermissionError):
            msg=prior_run_blocker(self.state(td))
            self.assertIn('status is unknown',msg)

    def test_finished_run_does_not_probe_pid(self):
        with tempfile.TemporaryDirectory() as td,patch('tests.real_model.guard.os.kill') as probe:
            self.assertIsNone(prior_run_blocker(self.state(td,status='interrupted')));probe.assert_not_called()

    def test_invalid_pid_is_not_sent_to_os(self):
        with tempfile.TemporaryDirectory() as td,patch('tests.real_model.guard.os.kill') as probe:
            self.assertIn('no valid PID',prior_run_blocker(self.state(td,pid=0)));probe.assert_not_called()
