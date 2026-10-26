import json
from pathlib import Path
import tempfile
import unittest
from vera_runner.firearm_data import build_inventory, write_inventory
from studies.firearm.audit import run_audit
from tests.firearm.fixtures import make_dataset


class AuditTests(unittest.TestCase):
    def test_audit_writes_validation_coverage_and_montages(self):
        with tempfile.TemporaryDirectory() as td:
            original = make_dataset(Path(td) / 'o', [
                ('Handgun', 'PAH1_C1_P1_V1_HB_1', 12, (64, 48), 25, {k: [k, 5, 10, 8] for k in range(3, 9)}),
                ('Machine_Gun', 'PAW1_C2_P3_V1_HB_1', 10, (96, 54), 30, {2: [1, 1, 20, 10], 4: [30, 20, 20, 10]}),
                ('No_Gun', 'N1_C1_P2_V2_HB_1', 8, (64, 48), 25, None)])
            write_inventory(build_inventory(original, 1), td)
            validation, cover, manifest = run_audit(Path(td) / 'inventory.json', original, Path(td) / 'audit')
            self.assertEqual({k: v['problems'] for k, v in validation.items()}, {'Handgun__PAH1_C1_P1_V1_HB_1': [], 'Machine_Gun__PAW1_C2_P3_V1_HB_1': []})
            by = {c['id']: c for c in cover}
            self.assertEqual((by['Machine_Gun__PAW1_C2_P3_V1_HB_1']['boxed'], by['Machine_Gun__PAW1_C2_P3_V1_HB_1']['boxed_runs']), (2, 2))
            self.assertEqual(by['Handgun__PAH1_C1_P1_V1_HB_1']['listed_no_box'], 6)
            for pick in manifest['origin_checks']:
                self.assertTrue((Path(td) / 'audit' / pick['file']).is_file())
            self.assertTrue(list((Path(td) / 'audit/no_gun').glob('*.png')))
            self.assertTrue(json.loads((Path(td) / 'audit/montages.json').read_text())['unboxed_frames'])
