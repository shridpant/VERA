"""Synthetic checks of scientific safeguards, not evidence of detection performance."""
import copy
import math
import unittest
from core import corroborate, select_anchors, support_score


def clear_views():
    return {name: {"delete_evidence": .2, "delete_controls": [.7, .8, .9],
                   "retain_evidence": .9, "retain_controls": [.1, .2, .3]}
            for name in ("blur", "mean")}


class ScientificSafeguards(unittest.TestCase):
    def test_uncorroborated_persistent_evidence_is_not_suppressed(self):
        views = clear_views()
        for view in views.values():
            view["delete_evidence"] = .9
        out = corroborate(.8, .8, views)
        self.assertEqual(out["score"], .8)
        self.assertEqual(out["status"], "unresolved")

    def test_selection_has_no_label_dependency(self):
        records = [{"anchor": i, "score": .7, "label": i % 2} for i in range(20)]
        first = select_anchors(records)
        for r in records:
            r["label"] = 1-r["label"]
        self.assertEqual(first, select_anchors(records))
        self.assertEqual(first, [0, 1])

    def test_one_bad_operator_blocks_boost(self):
        views = clear_views()
        views["mean"]["retain_evidence"] = .1
        self.assertEqual(corroborate(.8, .8, views)["score"], .8)

    def test_missing_control_is_fallback_not_normality(self):
        views = clear_views()
        views["blur"]["delete_controls"] = [.8]
        out = corroborate(.8, .8, views)
        self.assertEqual(out["score"], .8)
        self.assertEqual(out["status"], "unverified")

    def test_sampling_shift_blocks_boost(self):
        self.assertEqual(corroborate(.8, .3, clear_views())["reason"], "sampling_instability")

    def test_score_bounds_and_no_input_mutation(self):
        views = clear_views()
        before = copy.deepcopy(views)
        for p in [0, .1, .5, .8, 1]:
            score = corroborate(p, p, views)["score"]
            self.assertGreaterEqual(score, p)
            self.assertLessEqual(score, 1)
        self.assertEqual(views, before)

    def test_nonfinite_view_falls_back(self):
        views = clear_views()
        views["mean"]["delete_evidence"] = math.nan
        self.assertEqual(corroborate(.8, .8, views)["score"], .8)

    def test_extreme_logits_remain_finite(self):
        self.assertEqual(support_score(10000, 0), 1)
        self.assertEqual(support_score(-10000, 0), 0)
        self.assertEqual(support_score(0, 0), .5)


if __name__ == "__main__":
    unittest.main()
