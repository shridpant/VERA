import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from vera_runner.config import ROOT


class BoxesPipelineTests(unittest.TestCase):
    def test_only_alert_windows_are_boxed_and_every_step_is_on_the_timeline(self):
        import cv2
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);vid=root/'ramp.avi'
            writer=cv2.VideoWriter(str(vid),cv2.VideoWriter_fourcc(*'MJPG'),30,(32,24))
            for i in range(480):writer.write(np.full((24,32,3),min(255,i//2),np.uint8))
            writer.release()
            (root/'reg.json').write_text(json.dumps({'sample':{'splits':{'test':{'videos':td}}}}))
            cfg={'dataset':'sample','registry':str(root/'reg.json'),'model':{'backend':'mock'},'output_root':str(root/'runs'),
                 'boxes':{'mode':'alerts'}}
            (root/'c.json').write_text(json.dumps(cfg))
            subprocess.run([sys.executable,str(ROOT/'run_vera.py'),'--config',str(root/'c.json')],cwd=ROOT,check=True,capture_output=True)
            out=root/'runs/run 1/output'
            labels={json.loads(x)['anchor']:json.loads(x)['strict_label'] for x in (out/'ramp/raw_responses.jsonl').read_text().splitlines()}
            boxed=[json.loads(x) for x in (out/'ramp/boxes.jsonl').read_text().splitlines()]
            # The mock labels bright windows 1; the ramp has both kinds, and only the 1s are boxed.
            self.assertEqual(set(labels.values()),{0,1})
            self.assertEqual([b['anchor'] for b in boxed],sorted(a for a,l in labels.items() if l==1))
            self.assertTrue(all(len(b['frames'])==8 and b['phrase']=='mock anomaly' for b in boxed))
            self.assertEqual(boxed[0]['frames'][0]['status'],'box')
            events=[json.loads(x)['event'] for x in (out/'timeline.jsonl').read_text().splitlines()]
            self.assertEqual(events[:5],['run','video','model_loading','model_ready','prompt'])
            self.assertEqual(events[-1],'done')
            self.assertEqual(events.count('window'),len(labels))
            self.assertEqual(events.count('box'),8*len(boxed))
            self.assertTrue((out/'model_input.txt').read_text().startswith('MOCK INPUT'))

    def test_boxes_off_writes_no_boxes(self):
        import cv2
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);vid=root/'bright.avi'
            writer=cv2.VideoWriter(str(vid),cv2.VideoWriter_fourcc(*'MJPG'),30,(32,24))
            for i in range(320):writer.write(np.full((24,32,3),200,np.uint8))
            writer.release()
            (root/'reg.json').write_text(json.dumps({'sample':{'splits':{'test':{'videos':td}}}}))
            (root/'c.json').write_text(json.dumps({'dataset':'sample','registry':str(root/'reg.json'),'model':{'backend':'mock'},
                                                   'output_root':str(root/'runs')}))
            subprocess.run([sys.executable,str(ROOT/'run_vera.py'),'--config',str(root/'c.json')],cwd=ROOT,check=True,capture_output=True)
            self.assertFalse((root/'runs/run 1/output/bright/boxes.jsonl').exists())
