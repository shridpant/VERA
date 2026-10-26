import unittest
import numpy as np
from PIL import Image
from vera_runner.reliance import boxes_overlap, control_box, coverage, decision_split, frame_lifts, lift, mask_image

CATS = ['Handgun', 'Machine_Gun', 'No_Gun']


class CoverageTests(unittest.TestCase):
    def test_one_cell_half_cells_and_total_area(self):
        cell = coverage([62.5 * 3, 62.5 * 5, 62.5 * 4, 62.5 * 6])  # x in column 3, y in row 5
        self.assertAlmostEqual(cell[5, 3], 1.0)
        self.assertAlmostEqual(cell.sum(), 1.0)
        half = coverage([0, 0, 31.25, 62.5])
        self.assertAlmostEqual(half[0, 0], 0.5)
        box = [123, 456, 345, 678]
        self.assertAlmostEqual(coverage(box).sum(), (box[2] - box[0]) * (box[3] - box[1]) / 62.5 ** 2)

    def test_lift_is_one_for_uniform_attention_and_grows_inside(self):
        uniform = {0: np.full(256, 1 / 256)}
        box = [100, 100, 300, 250]
        self.assertAlmostEqual(lift(uniform, {0: [box]}), 1.0)
        focused = np.zeros(256)
        focused[coverage(box).reshape(-1) == 1] = 1
        focused /= focused.sum()
        self.assertGreater(lift({0: focused}, {0: [box]}), 5)
        self.assertEqual(set(frame_lifts(uniform, {0: [box]})), {0})


class ControlTests(unittest.TestCase):
    def test_mirror_rules_and_equal_size(self):
        self.assertEqual(control_box([100, 200, 200, 300]), [800, 200, 900, 300])
        center = [450, 100, 550, 200]  # horizontal mirror overlaps itself; vertical does not
        self.assertEqual(control_box(center), [450, 800, 550, 900])
        mid = [400, 400, 600, 600]  # both mirrors overlap: fallback, still equal size and disjoint
        c = control_box(mid)
        self.assertFalse(boxes_overlap(c, mid))
        self.assertAlmostEqual(c[2] - c[0], 200)
        self.assertAlmostEqual(c[3] - c[1], 200)
        self.assertIsNone(control_box([300, 300, 700, 700]))  # no equal-sized disjoint position exists

    def test_mask_fills_exactly_the_box_pixels(self):
        image = Image.new('RGB', (448, 448), (0, 0, 0))
        out = np.asarray(mask_image(image, [[250, 500, 500, 750]]))
        grey = np.argwhere(out[..., 0] == 127)
        self.assertEqual((grey[:, 1].min(), grey[:, 0].min(), grey[:, 1].max() + 1, grey[:, 0].max() + 1), (112, 224, 224, 336))


class SplitTests(unittest.TestCase):
    def test_decision_token_after_the_last_category_field(self):
        pieces = ['', 'Answers', ': the category is clear.', '\n', 'Category', ':\n', 'Hand', 'gun', '\n', 'Action']
        self.assertEqual(decision_split(pieces, CATS), (6, ''))
        pieces = ['Category', ':', ' Machine', '_G', 'un']
        self.assertEqual(decision_split(pieces, CATS), (2, ' '))

    def test_no_field_no_name_or_merged_token(self):
        self.assertIsNone(decision_split(['no', ' field'], CATS))
        self.assertIsNone(decision_split(['Category', ':'], CATS))
        self.assertIsNone(decision_split(['Category', ':Hand', 'gun'], CATS))
