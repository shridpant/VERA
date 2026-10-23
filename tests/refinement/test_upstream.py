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


class RefinementTests(unittest.TestCase):
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
