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


class RunnerTests(unittest.TestCase):
    def test_unknown_setting_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'c.json';p.write_text('{"intervention":{"radius":10}}')
            with self.assertRaises(ValueError):load(p)

    def test_upstream_sampling(self):
        # Independent literal upstream arithmetic, including fractional endpoints.
        for length in [17,300,2159,2580]:
            got=list(windows(length,DEFAULTS['sampling']))
            for row,anchor in zip(got,range(0,length,16)):
                start=max(anchor-.5*30*10,0);end=min(anchor+.5*30*10,length)
                want=np.arange(start,end,(end-start)/8).astype(int).tolist()
                self.assertEqual(row['frame_indices'],want)

    def test_prompt_matches_original_forward(self):
        # Execute the actual upstream forward with a fake model, no weights or script side effects.
        tree=ast.parse((ROOT/'generate_initial_score.py').read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='VideoAnomalyDetectionModelInference')
        fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='forward')
        ns={'torch':__import__('torch'),'generation_config':DEFAULTS['generation']}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original forward>','exec'),ns)
        from types import SimpleNamespace
        import torch
        captured={}
        class Fake:
            def eval(self):pass
            def batch_chat(self,*args,**kw):captured.update(kw);return ['Output:\n0']
        from vera_runner.backends import upstream_questions
        obj=SimpleNamespace(model=Fake(),tokenizer=None,model_instruct=(ROOT/'VERA_learner_instruct.txt').read_text(),new_qs=upstream_questions())
        ns['forward'](obj,torch.zeros(1,8,3,2,2),[1]*8)
        self.assertEqual(prompt(DEFAULTS),captured['questions'][0])

    def test_interventions_exact(self):
        a=Image.fromarray(np.random.default_rng(0).integers(0,256,(448,448,3),dtype=np.uint8))
        iv=copy.deepcopy(DEFAULTS['intervention'])
        self.assertIs(intervene(a,iv),a)
        iv['mode']='grey';out=np.array(intervene(a,iv));self.assertTrue(np.all(out==127))
        from PIL import ImageFilter
        iv['mode']='blur';self.assertEqual(intervene(a,iv).tobytes(),a.filter(ImageFilter.GaussianBlur(17.92)).tobytes())

    def test_parser_does_not_invent_labels(self):
        self.assertIsNone(strict_label('Cannot tell'))
        self.assertIsNone(strict_label('Output: 0 or 1'))
        self.assertEqual(strict_label('Analysis\nOutput:\n1.'),1)

    def test_refinement_matches_upstream(self):
        rng=np.random.default_rng(12);length=700
        rows=list(windows(length,DEFAULTS['sampling']))
        for r in rows:r['strict_label']=int(rng.random()<.5)
        feat=rng.normal(size=(len(rows),16));ours=refine(rows,feat,length,16,DEFAULTS['refinement'])['frame_scores']
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);(p/'Data/segment_level_score').mkdir(parents=True);(p/'Data/vision_features').mkdir()
            (p/'Data/UCF_Eval.json').write_text(json.dumps([{'video':'v.mp4','length':length,'temporal_label':[100,300]}]))
            scores={str(r['anchor']):{'start':r['start'],'end':r['end'],'score':r['strict_label']} for r in rows}
            (p/'Data/segment_level_score/v.json').write_text(json.dumps(scores));np.save(p/'Data/vision_features/v.npy',feat)
            (p/'upstream.py').write_text((ROOT/'compute_auc.py').read_text())
            subprocess.run([sys.executable,'-c',"import runpy,numpy as np; x=runpy.run_path('upstream.py'); np.save('result.npy',x['all_predict_score'])"],cwd=p,check=True,capture_output=True)
            self.assertTrue(np.array_equal(ours,np.load(p/'result.npy')))

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

if __name__=='__main__':unittest.main()
