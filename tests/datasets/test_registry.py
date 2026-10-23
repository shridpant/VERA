import copy
import json
from pathlib import Path
import tempfile
import unittest
from vera_runner.config import DEFAULTS
from vera_runner.data import resolve

class DatasetRegistryTests(unittest.TestCase):
    def test_named_dataset_respects_manifest_split(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'test_clip.mp4').touch();(root/'train_clip.mp4').touch()
            (root/'annotations.json').write_text(json.dumps([{'video':'frames/X/test_clip.mp4'}]))
            (root/'registry.json').write_text(json.dumps({'example':{'splits':{'test':{'videos':td,'annotations':str(root/'annotations.json')}}}}))
            cfg=copy.deepcopy(DEFAULTS);cfg.update(dataset='example',registry=str(root/'registry.json'))
            self.assertEqual([x['id'] for x in resolve(cfg)],['test_clip'])
            cfg['selection']['video_ids']=['train_clip']
            with self.assertRaises(ValueError):resolve(cfg)

    def test_duplicate_stems_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'a').mkdir();(root/'b').mkdir()
            (root/'a/clip.mp4').touch();(root/'b/clip.avi').touch()
            (root/'registry.json').write_text(json.dumps({'example':{'splits':{'test':{'videos':td}}}}))
            cfg=copy.deepcopy(DEFAULTS);cfg.update(dataset='example',registry=str(root/'registry.json'))
            with self.assertRaises(ValueError):resolve(cfg)
