import unittest
from vera_runner.grounding import clean_phrase, parse_boxes


# Tagged decodes as InternVL2-8B produced them in grounding pilot WI-01 (spaces around special tokens).
CRASH = '<s> <ref> Red truck and train crossing the road simultaneously </ref> <box> [[353, 123, 1000, 715]] </box>  <|im_end|>'
WHOLE = '<s> <ref> Visible damage to the train and the truck </ref> <box> [[0, 0, 1000, 1000]] </box>  <|im_end|>'


class GroundingParserTests(unittest.TestCase):
    def test_real_pilot_outputs(self):
        crash=parse_boxes(CRASH)
        self.assertEqual((crash['status'],crash['boxes'],crash['tag_format']),('box',[[353,123,1000,715]],'internvl_ref_box'))
        self.assertAlmostEqual(crash['area_fraction'],0.647*0.592,places=4)
        self.assertEqual(parse_boxes(WHOLE)['status'],'full_frame')
        # Near-full but below 95 % stays a box; its area is reported so it can be judged.
        near=parse_boxes('<ref> x </ref> <box> [[0, 204, 1000, 1000]] </box>')
        self.assertEqual((near['status'],near['area_fraction']),('box',0.796))

    def test_missing_cut_and_bad_boxes(self):
        self.assertEqual(parse_boxes('There is no such region.')['status'],'no_box')
        self.assertEqual(parse_boxes('<ref> x </ref> <box> [[12, 30')['status'],'malformed')
        self.assertEqual(parse_boxes('<ref> x </ref> <box> [[500, 30, 100, 90]] </box>')['status'],'malformed')
        self.assertEqual(parse_boxes('<ref> x </ref> <box> [[10, 20, 300, 400]] [[0, 0, 5000, 5]] </box>')['boxes'],[[10,20,300,400]])
        cut=parse_boxes('<ref> x </ref> <box> [[10, 20, 300, 400]]')  # Token limit reached before </box>.
        self.assertEqual(cut['status'],'box')

    def test_phrase_cleanup(self):
        self.assertEqual(clean_phrase('"A red truck hitting a train."\nMore text'),'A red truck hitting a train')
        self.assertEqual(clean_phrase('  '),'')
