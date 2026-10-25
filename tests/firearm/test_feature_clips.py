import importlib.util
import unittest
import numpy as np
from vera_runner.imagebind_features import anchors, clip_frame_indices, compare


class ClipIndexTests(unittest.TestCase):
    def test_lavad_clip_arithmetic(self):
        # range(max(c-150,0), min(c+150,n)) then np.linspace(0, t-1, 10, dtype=int); all frames when t <= 10.
        self.assertEqual(clip_frame_indices(16, 40), [0, 4, 8, 13, 17, 21, 26, 30, 34, 39])
        self.assertEqual(clip_frame_indices(0, 7), list(range(7)))
        self.assertEqual(clip_frame_indices(0, 10), list(range(10)))
        self.assertEqual(clip_frame_indices(400, 1000), [250, 283, 316, 349, 382, 416, 449, 482, 515, 549])
        self.assertEqual(clip_frame_indices(224, 225), [74, 90, 107, 124, 140, 157, 174, 190, 207, 224])

    def test_one_row_per_runner_anchor(self):
        from vera_runner.data import windows
        sampling = {'stride': 16, 'window_seconds': 10.0, 'fps': 30.0, 'frames_per_window': 8}
        for n in (8, 16, 17, 224, 225, 450):
            self.assertEqual(anchors(n), [w['anchor'] for w in windows(n, sampling)])

    def test_compare_reports_cosine_and_shape(self):
        a = np.random.default_rng(0).normal(size=(5, 4))
        self.assertAlmostEqual(compare(a, 3 * a)['cosine_min'], 1.0)
        self.assertFalse(compare(a, a[:4])['same_shape'])


@unittest.skipUnless(importlib.util.find_spec('pytorchvideo') and importlib.util.find_spec('imagebind'), 'ImageBind env only')
class ClipTensorTests(unittest.TestCase):
    def test_fifteen_two_frame_crops(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        from vera_runner.imagebind_features import clip_tensor
        with tempfile.TemporaryDirectory() as td:
            paths = []
            for i in range(10):
                Image.new('RGB', (320, 240), (i * 20, 0, 0)).save(Path(td) / f'{i:06d}.jpg')
                paths.append(Path(td) / f'{i:06d}.jpg')
            self.assertEqual(tuple(clip_tensor(paths).shape), (15, 3, 2, 224, 224))
