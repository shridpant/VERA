import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from PIL import Image
from vera_runner.config import DEFAULTS,ROOT,load
from vera_runner.data import windows
from vera_runner.backends import prompt,intervene,strict_label
from vera_runner.refine import refine


class MockPipelineTests(unittest.TestCase):
    def test_mock_end_to_end_probe_and_refine(self):
        import cv2
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);vid=root/'tiny.avi'
            writer=cv2.VideoWriter(str(vid),cv2.VideoWriter_fourcc(*'MJPG'),30,(32,24))
            for i in range(320):writer.write(np.full((24,32,3),i%255,np.uint8))
            writer.release()
            (root/'ann.json').write_text(json.dumps([{'video':'tiny.avi','length':320,'temporal_label':[100,200]}]))
            np.save(root/'tiny.npy',np.ones((20,4)))
            reg={'sample':{'splits':{'test':{'videos':td,'annotations':str(root/'ann.json'),'features':td}}}}
            (root/'reg.json').write_text(json.dumps(reg))
            cfg={'dataset':'sample','registry':str(root/'reg.json'),'model':{'backend':'mock'},'output_root':str(root/'runs'),
                 'selection':{'max_windows_per_video':1},'intervention':{'mode':'grey','slots':[0,2],'probe':True}}
            (root/'c.json').write_text(json.dumps(cfg))
            subprocess.run([sys.executable,str(ROOT/'run_vera.py'),'--config',str(root/'c.json')],cwd=ROOT,check=True,capture_output=True)
            out=root/'runs/run 1/output';status=json.loads((out/'status.json').read_text())
            self.assertEqual(status['status'],'complete');self.assertFalse(status['inference_completed']);self.assertTrue(status['synthetic'])
            records=[json.loads(x) for x in (out/'tiny/raw_responses.jsonl').read_text().splitlines()]
            self.assertEqual([x['condition'] for x in records],['baseline','grey'])
            self.assertEqual([i for i,(a,b) in enumerate(zip(records[0]['rgb_sha256'],records[1]['rgb_sha256'])) if a!=b],[0,2])
            cfg['selection']['max_windows_per_video']=None;cfg['intervention']={'mode':'baseline'}
            cfg['stages']=['preprocess','infer','refine','evaluate'];(root/'c.json').write_text(json.dumps(cfg))
            subprocess.run([sys.executable,str(ROOT/'run_vera.py'),'--config',str(root/'c.json')],cwd=ROOT,check=True,capture_output=True)
            self.assertTrue((root/'runs/run 2/output/metrics.json').exists())
            self.assertTrue((root/'runs/run 2/output/tiny/frame_scores.npy').exists())
