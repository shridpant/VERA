import itertools
import unittest
from vera_runner.localization import iou, match, prf, score_instance, size_bin


class LocalizationTests(unittest.TestCase):
    def test_iou_values(self):
        self.assertAlmostEqual(iou([0, 0, 100, 100], [50, 0, 150, 100]), 5000 / 15000)
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        self.assertEqual(iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_duplicate_prediction_is_a_false_positive(self):
        r = score_instance('boxes', [[0, 0, 100, 100], [5, 5, 100, 100]], [[0, 0, 100, 100]], 0.5, True)
        self.assertEqual((r['tp'], r['fp'], r['fn']), (1, 1, 0))
        self.assertEqual(r['matched_ious'], [1.0])

    def test_one_to_one_maximizes_matches_then_iou(self):
        # Greedy best-IoU would pair p0 with g0 and leave g1 unmatched; the assignment matches both.
        preds = [[0, 0, 100, 100], [60, 0, 160, 100]]
        gts = [[30, 0, 130, 100], [100, 0, 200, 100]]
        pairs = match(preds, gts, 0.25)
        self.assertEqual([(p, g) for p, g, _ in pairs], [(0, 0), (1, 1)])

    def test_ties_resolve_identically_under_permutation(self):
        preds = [[0, 0, 100, 100], [0, 0, 100, 100]]
        gts = [[0, 0, 100, 100]]
        results = set()
        for perm in itertools.permutations(range(2)):
            pairs = match([preds[i] for i in perm], gts, 0.5)
            results.add(tuple(sorted(tuple(preds[perm[p]]) for p, _, _ in pairs)))
        self.assertEqual(len(results), 1)

    def test_thresholds_and_whole_image_boxes(self):
        whole = [[0, 0, 1000, 1000]]
        gt = [[400, 400, 450, 450]]
        r = score_instance('boxes', whole, gt, 0.5, True)
        self.assertEqual((r['tp'], r['fp'], r['fn']), (0, 1, 1))  # kept, and it misses a small firearm
        near = [[400, 400, 460, 455]]
        self.assertEqual(score_instance('boxes', near, gt, 0.5, True)['tp'], 1)
        self.assertEqual(score_instance('boxes', near, gt, 0.75, True)['tp'], 1)
        self.assertEqual(score_instance('boxes', [[400, 400, 480, 470]], gt, 0.75, True)['tp'], 0)

    def test_unanswered_slots_miss_every_object_and_precision_needs_completeness(self):
        for status in ('malformed', 'unknown', 'missing', 'duplicate', 'truncated'):
            r = score_instance(status, [], [[1, 1, 2, 2]], 0.5, True)
            self.assertEqual((r['answered'], r['tp'], r['fp'], r['fn']), (False, 0, None, 1))
        r = score_instance('boxes', [[1, 1, 50, 50]], [], 0.5, False)
        self.assertIsNone(r['fp'])  # e.g. a listed_no_box frame
        r = score_instance('boxes', [[1, 1, 50, 50]], [], 0.5, True)
        self.assertEqual(r['fp'], 1)  # e.g. a No_Gun frame
        self.assertEqual(score_instance('empty', [], [], 0.5, True)['fp'], 0)

    def test_prf_and_size_bins(self):
        self.assertEqual(prf(1, 1, 1), (0.5, 0.5, 0.5))
        self.assertEqual(prf(0, None, 2), (None, 0.0, None))
        self.assertEqual(prf(0, 0, 0), (None, None, None))
        edges = [0.003359375, 0.0109716796875]
        self.assertEqual(size_bin([0, 0, 50, 50], edges), 'small')
        self.assertEqual(size_bin([0, 0, 80, 80], edges), 'medium')
        self.assertEqual(size_bin([0, 0, 200, 200], edges), 'large')


class CardinalityTests(unittest.TestCase):
    def test_more_matches_always_beat_higher_total_iou(self):
        # Reviewer's counterexample: a diagonal at IoU 0.5 (5 matches) against four off-diagonal pairs at 1.0.
        import numpy as np
        from unittest import mock
        import vera_runner.localization as L
        m = np.zeros((5, 5))
        np.fill_diagonal(m, 0.5)
        for i in range(4):
            m[i, i + 1] = 1.0
        boxes = [[i, 0, i + 1, 1] for i in range(5)]
        with mock.patch.object(L, 'iou', lambda a, b: m[int(a[0]), int(b[0])]):
            self.assertEqual(len(L.match(boxes, boxes, 0.5)), 5)
