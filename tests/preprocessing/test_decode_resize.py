import ast
from pathlib import Path
import runpy
import tempfile
import unittest
import cv2
import numpy as np
from PIL import Image
from vera_runner.config import ROOT
from vera_runner.data import preprocess

class DecodeResizeTests(unittest.TestCase):
    def test_jpeg_decode_and_resize_match_original(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);video=root/'sample.avi'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),30,(64,48))
            for i in range(10):writer.write(np.random.default_rng(i).integers(0,256,(48,64,3),dtype=np.uint8))
            writer.release()
            item={'id':'sample','path':str(video),'annotation':{'length':10}}
            ours=root/'ours';metadata=preprocess(item,ours)
            original=runpy.run_path(str(ROOT/'Preprocessing/extract_frames.py'))
            original['extract_frames'](str(video),str(root/'original'))
            self.assertEqual(metadata['decoded_frames'],10)
            for p in ours.glob('*.jpg'):
                self.assertEqual(p.read_bytes(),(root/'original/sample'/p.name).read_bytes())
            tree=ast.parse((ROOT/'generate_initial_score.py').read_text())
            fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dynamic_preprocess')
            ns={'Image':Image};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original resize>','exec'),ns)
            with Image.open(ours/'000000.jpg') as frame:
                rgb=frame.convert('RGB')
                self.assertEqual(rgb.resize((448,448)).tobytes(),ns['dynamic_preprocess']([rgb],grid_size=1).tobytes())
