import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from vera_runner.config import ROOT
from vera_runner.study import run_study
from studies.firearm.overlays import sheets
from tests.firearm.test_study_run import Recording, make_study


class PilotToolTests(unittest.TestCase):
    def test_sheets_and_slot_check_on_a_mock_run(self):
        with tempfile.TemporaryDirectory() as td:
            study = make_study(td)
            run_study(study, backend=Recording(study['runner']))
            made = sheets(Path(td) / 'run', Path(td) / 'inventory.json', Path(td) / 'original', 1, Path(td) / 'sheets')
            self.assertEqual(len(made), 5)
            config = Path(td) / 'study.json'
            config.write_text(json.dumps({'study_id': 'x', 'runner': {'dataset': 'firearm_visilab', 'registry': str(Path(td) / 'registry.json'),
                                                                      'model': {'backend': 'mock'}}}))
            out = Path(td) / 'slot_check.json'
            subprocess.run([sys.executable, str(ROOT / 'studies/firearm/slot_check.py'), '--config', str(config), '--run-dir', str(Path(td) / 'run'),
                            '--windows', 'Handgun__PAH1_C1_P1_V1_HB_1:0', '--keep', '0,7', '--out', str(out)],
                           cwd=ROOT, check=True, capture_output=True)
            results = json.loads(out.read_text())['results']
            # The mock boxes bright slots only: slot 7 of this window is bright, slot 0 dark; grey is dark.
            self.assertEqual([(r['variant'], r['kept_slot'], r['slots_with_boxes']) for r in results],
                             [('keep', 0, []), ('keep', 7, [7]), ('all_grey', None, [])])
