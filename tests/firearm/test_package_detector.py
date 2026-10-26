import unittest
from vera_runner.detector import iou, label_of, match, parse, parse_internvl, parse_qwen, says_none, score_frame
from vera_runner.vlm import dynamic_preprocess, split_thinking


class ParserTests(unittest.TestCase):
    def test_internvl_tags_labels_and_malformed(self):
        text = '<ref>handgun</ref><box>[[100, 200, 300, 400]]</box> and <ref>long gun</ref><box>[[10, 20, 30, 40], [5, 5, 4, 9]]</box>'
        found, bad = parse_internvl(text)
        self.assertEqual([f['box'] for f in found], [[100, 200, 300, 400], [10, 20, 30, 40]])
        self.assertEqual([f['label'] for f in found], ['handgun', 'long gun'])
        self.assertEqual(bad, 1)
        found, bad = parse_internvl('the firearm<box>[[0, 0, 1000, 1000]')  # cut at the token limit
        self.assertEqual(found[0]['label'], 'firearm')

    def test_qwen_json_fences_and_regex_fallback(self):
        found, bad = parse_qwen('```json\n[{"bbox_2d": [1, 2, 30, 40], "label": "pistol"}]\n```')
        self.assertEqual((found[0]['box'], found[0]['label'], bad), ([1, 2, 30, 40], 'handgun', 0))
        found, _ = parse_qwen('[{"bbox_2d": [1, 2, 30, 40], "label": "rifle"}, {"bbox_2d": [1, 2, 30, 40] "label": "x"}]')
        self.assertEqual([f['label'] for f in found], ['long gun', 'firearm'])  # invalid JSON -> regex
        self.assertEqual(parse_qwen('[]'), ([], 0))
        self.assertTrue(says_none('[]') and says_none('None.') and not says_none('a gun'))

    def test_thinking_split_and_truncation(self):
        self.assertEqual(split_thinking('<think>look</think>\n[]<|im_end|>', True), ('look', '[]', False))
        self.assertEqual(split_thinking('looking and looking', True), ('looking and looking', '', True))
        self.assertEqual(split_thinking('<s>plain', False), ('', 'plain', False))
        r = parse('qwen', {'answer': '', 'truncated': True, 'text': 'x'})
        self.assertEqual(r['status'], 'truncated')
        r = parse('internvl', {'answer': '<ref>handgun</ref><box>[[1, 1, 9, 9]]</box>', 'truncated': False, 'text': ''})
        self.assertEqual(r['status'], 'boxes')

    def test_labels(self):
        self.assertEqual([label_of(t) for t in ('Handgun', 'a pistol', 'Long gun', 'machine gun', 'gun')],
                         ['handgun', 'handgun', 'long gun', 'long gun', 'firearm'])


class ScoreTests(unittest.TestCase):
    def test_iou_match_whole_image(self):
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 0, 15, 10]), 50 / 150)
        gts = [[100, 100, 200, 200], [500, 500, 600, 600]]
        preds = [[0, 0, 1000, 1000], [105, 100, 205, 200], [100, 100, 200, 200]]
        self.assertEqual(match(preds, gts, 0.5), 1)  # one-to-one; the whole-image box never matches
        s = score_frame([{'box': p} for p in preds], gts)
        self.assertEqual((s['n_pred'], s['n_whole_image'], s['tp50'], s['n_gt']), (3, 1, 1, 2))
        self.assertEqual(score_frame([], gts)['best_iou'], None)


class TilingTests(unittest.TestCase):
    def test_dynamic_preprocess_counts(self):
        from PIL import Image
        tiles = dynamic_preprocess(Image.new('RGB', (1920, 1080)), max_num=6)
        self.assertEqual(len(tiles), 3)  # 2x1 tiles (closest ratio with <= 6 tiles) + thumbnail
        self.assertEqual(len(dynamic_preprocess(Image.new('RGB', (448, 448)), max_num=6)), 1)
