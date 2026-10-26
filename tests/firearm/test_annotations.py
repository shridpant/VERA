import unittest
import numpy as np
from PIL import Image
from vera_runner.firearm_annotations import (frame_boxes, frame_index, frame_statuses, render_overlay, to_input_norm,
                                             validate_coco)
from tests.firearm.fixtures import coco


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class CoordinateTests(unittest.TestCase):
    def check_geometry(self, size, bbox):
        """A red box drawn at native xywh, resized exactly as the runner resizes model inputs."""
        w, h = size
        arr = np.zeros((h, w, 3), np.uint8)
        x, y, bw, bh = bbox
        arr[y:y + bh, x:x + bw] = (255, 0, 0)
        small = np.asarray(Image.fromarray(arr).convert('RGB').resize((448, 448)))
        red = np.argwhere((small[..., 0] > 128) & (small[..., 1] < 100))
        found = [red[:, 1].min(), red[:, 0].min(), red[:, 1].max() + 1, red[:, 0].max() + 1]
        expected = [v * 448 / 1000 for v in to_input_norm(bbox, w, h)]
        for got, want in zip(found, expected):
            self.assertLessEqual(abs(got - want), 1.5, (size, bbox, found, expected))

    def test_cctv_and_phone_geometry_on_the_448_input(self):
        self.check_geometry((640, 480), (121, 128, 19, 21))
        self.check_geometry((640, 480), (590, 216, 43, 40))
        self.check_geometry((1920, 1080), (900, 400, 120, 60))
        self.check_geometry((1920, 1080), (0, 0, 64, 1080))

    def test_known_values_and_clipping(self):
        self.assertEqual(to_input_norm([64, 48, 64, 48], 640, 480), [100.0, 100.0, 200.0, 200.0])
        self.assertEqual(to_input_norm([1900, 1000, 40, 100], 1920, 1080)[2:], [1000.0, 1000.0])

    def test_iou_is_unchanged_by_the_anisotropic_resize(self):
        a, b = [100, 50, 40, 30], [110, 60, 40, 30]
        native = iou([a[0], a[1], a[0] + a[2], a[1] + a[3]], [b[0], b[1], b[0] + b[2], b[1] + b[3]])
        self.assertAlmostEqual(native, iou(to_input_norm(a, 1920, 1080), to_input_norm(b, 1920, 1080)), places=9)

    def test_overlay_draws_in_image_pixels(self):
        out = np.asarray(render_overlay(Image.new('RGB', (200, 100)), [[100, 100, 500, 500]], width=1))
        self.assertTrue((out[10, 20:100] == (255, 0, 0)).all())


class AnnotationTests(unittest.TestCase):
    def test_origin_mapping_and_statuses(self):
        c = coco(5, (64, 48), {2: [1, 1, 10, 10], 4: [2, 2, 10, 10]})
        self.assertEqual(frame_index(1, 1), 0)
        self.assertEqual(sorted(frame_boxes(c, 64, 48, 1)), [1, 3])
        self.assertEqual(frame_statuses('Handgun', 5, c, 1), ['listed_no_box', 'positive', 'listed_no_box', 'positive', 'listed_no_box'])
        self.assertEqual(frame_statuses('No_Gun', 2, None, 1), ['negative_category'] * 2)
        self.assertEqual(frame_statuses('Handgun', 2, None, 1), ['unknown'] * 2)
        with self.assertRaises(ValueError):
            frame_index(1, 2)

    def test_validation_reports_problems(self):
        self.assertEqual(validate_coco(coco(3, (64, 48), {1: [1, 1, 5, 5]}), 3, 64, 48, 'Handgun'), [])
        bad = coco(3, (64, 48), {1: [60, 1, 10, 5], 2: [1, 1, 5, 5]})
        bad['annotations'].append({'id': 9, 'image_id': 2, 'category_id': 7, 'bbox': [1, 1, 2, 2]})
        problems = ' '.join(validate_coco(bad, 4, 64, 48, 'Machine_Gun'))
        for text in ('not 1..4', 'outside', 'unknown category', 'more than one box', 'differ from the folder'):
            self.assertIn(text, problems)
