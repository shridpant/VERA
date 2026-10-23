"""Opt-in acceptance test using real checkpoint weights and an original UCF video."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from datetime import datetime,timezone
from vera_runner.config import ROOT


@unittest.skipUnless(os.environ.get('VERA_TEST_REAL_MODEL') == '1',
                     'Real-model acceptance requires explicit --suite real-model; it is not a mock test')
class RealModelAcceptance(unittest.TestCase):
    def test_real_internvl2_window(self):
        config_path=Path(os.environ.get('VERA_REAL_TEST_CONFIG',str(ROOT/'tests/real_model/config.local.json')))
        self.assertTrue(config_path.is_file(),'Create tests/real_model/config.local.json from config.example.json')
        settings=json.loads(config_path.read_text())
        cfg=settings['experiment']
        self.assertEqual(cfg['model']['backend'],'internvl2')
        self.assertEqual(cfg['selection']['max_windows_per_video'],1)
        self.assertEqual(len(cfg['selection']['video_ids']),1)
        self.assertEqual(cfg['intervention']['mode'],'baseline')
        self.assertEqual(cfg['stages'],['preprocess','infer'])
        # Do not load a second 8B model while the known prior run is still active.
        guard=settings.get('existing_run_status')
        from tests.real_model.guard import prior_run_blocker
        blocker=prior_run_blocker(guard)
        self.assertIsNone(blocker,blocker)
        cfg_file=ROOT/'tests/real_model/resolved.local.json'
        cfg_file.write_text(json.dumps(cfg,indent=2)+'\n')
        output_root=Path(cfg['output_root'])
        if not output_root.is_absolute():output_root=ROOT/output_root
        before=set(output_root.glob('run *'))
        result=subprocess.run([sys.executable,str(ROOT/'run_vera.py'),'--config',str(cfg_file)],
                              cwd=ROOT,capture_output=True,text=True,
                              timeout=cfg['runtime']['timeout_seconds']+30)
        (ROOT/'tests/reports/real_model/last_console.log').write_text(result.stdout+'\n'+result.stderr)
        self.assertEqual(result.returncode,0,'Real runner failed. See tests/reports/real_model/last_console.log and tests/artifacts/real_model/run N/output/diagnostic.log')
        created=set(output_root.glob('run *'))-before
        self.assertEqual(len(created),1)
        output=created.pop()/'output'
        status=json.loads((output/'status.json').read_text())
        self.assertEqual(status['status'],'complete')
        self.assertTrue(status['inference_completed'])
        self.assertFalse(status['synthetic'])
        files=list(output.glob('*/raw_responses.jsonl'));self.assertEqual(len(files),1)
        rows=[json.loads(line) for line in files[0].read_text().splitlines()]
        self.assertEqual(len(rows),1);row=rows[0]
        self.assertFalse(row['synthetic'])
        self.assertTrue(row['raw_response'].strip())
        self.assertIn(row['strict_label'],(0,1),'Generated response did not yield a valid strict anomaly label')
        self.assertTrue(row['input_token_ids'],'Real model input token IDs were not captured')
        self.assertEqual(len(list(output.glob('*/inputs/0/baseline/*.png'))),8)
