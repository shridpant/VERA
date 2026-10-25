import copy
import json
from pathlib import Path
import tempfile
import unittest
import yaml
from vera_runner.config import from_dict, load
from vera_runner.study import SPEC_KEYS, digest, load_study

YAML = Path('studies/firearm/experiment_A_G_AG.annotated.yaml')


class ConvertTests(unittest.TestCase):
    def setUp(self):
        self.raw = yaml.safe_load(YAML.read_text())

    def test_yaml_values_are_the_studys_frozen_settings(self):
        study = load_study(copy.deepcopy(self.raw))
        r = study['runner']
        self.assertEqual(r['model']['revision'], '6fb9ad6924f69424e57fab2ab061d707688f0296')
        self.assertEqual((r['model']['device'], r['model']['dtype'], r['model']['flash_attention']), ('cuda', 'bfloat16', True))
        self.assertEqual(r['generation'], {'num_beams': 1, 'max_new_tokens': 1024, 'do_sample': False})
        self.assertEqual(r['sampling'], {'stride': 16, 'window_seconds': 10.0, 'fps': 30.0, 'frames_per_window': 8})
        self.assertEqual(r['boxes']['mode'], 'off')  # quoted: YAML 1.1 would read a bare off as False
        self.assertEqual(study['conditions']['order'], ['A', 'G', 'AG'])
        self.assertEqual(study['analysis']['bootstrap_seed'], 20261025)

    def test_runner_block_is_a_valid_runner_config(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, 'runner.json').write_text(json.dumps(self.raw['runner']))
            self.assertEqual(load(Path(td, 'runner.json')), from_dict(self.raw['runner']))

    def test_cli_convert_round_trip(self):
        import subprocess, sys
        from vera_runner.config import ROOT
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / 'experiment.json'
            subprocess.run([sys.executable, str(ROOT / 'run_firearm_study.py'), 'convert', '--yaml', str(YAML), '--out', str(out)],
                           cwd=ROOT, check=True, capture_output=True)
            self.assertEqual(json.loads(out.read_text()), self.raw)
            self.assertEqual(json.loads((Path(td) / 'experiment.runner_A.json').read_text()), self.raw['runner'])

    def test_unknown_keys_fail_and_scientific_knobs_change_the_spec(self):
        bad = copy.deepcopy(self.raw)
        bad['execution']['retry_forever'] = True
        with self.assertRaises(ValueError):
            load_study(bad)
        bad = copy.deepcopy(self.raw)
        bad['runner']['generation']['top_k'] = 5
        with self.assertRaises(ValueError):
            load_study(bad)
        base = load_study(copy.deepcopy(self.raw))
        changed = copy.deepcopy(self.raw)
        changed['runner']['generation']['max_new_tokens'] = 2048
        operational = copy.deepcopy(self.raw)
        operational['execution']['max_attempts'] = 5
        spec = lambda s: digest({k: s[k] for k in SPEC_KEYS})
        self.assertNotEqual(spec(base), spec(load_study(changed)))
        self.assertEqual(spec(base), spec(load_study(operational)))
