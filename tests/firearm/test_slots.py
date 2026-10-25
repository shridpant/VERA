import unittest
from vera_runner.backends import strict_label
from vera_runner.slots import final_output_label, legacy_label, parse_slots, split_answer

G_OK = '''Firearm boxes:
Frame1: [[412, 230, 468, 281]]
Frame2: []
Frame3: [[10, 20, 30, 40], [500, 500, 600, 650]]
Frame4: unknown
Frame5: <ref>firearm</ref><box>[[100, 100, 200, 200]]</box>
Frame6: [0, 0, 1000, 1000]
Frame7: [[0, 0, 1000, 990]]
Frame8: []'''


def statuses(parsed):
    return [s['status'] for s in parsed['slots']]


class LabelTests(unittest.TestCase):
    def test_final_output_field_is_isolated(self):
        self.assertEqual(final_output_label('Answers to Prompt Questions: ...\nOutput:\n1'), 1)
        self.assertEqual(final_output_label('Output:\n```\n0\n```'), 0)
        ag = 'Answers...\nFirearm boxes:\nFrame1: [[100, 200, 300, 400]]\nOutput:\n1'
        self.assertEqual(final_output_label(ag), 1)
        self.assertEqual(legacy_label(ag), 1)

    def test_box_digits_after_output_corrupt_only_the_legacy_rule(self):
        ag = 'Answers...\nOutput: 1\nFirearm boxes:\nFrame1: [[100, 200, 300, 400]]'
        self.assertIsNone(final_output_label(ag))
        self.assertEqual(legacy_label(ag), 0)  # '0' inside the box digits

    def test_two_output_fields_use_the_last_and_the_runner_rule_refuses(self):
        text = 'Output: maybe\nmore\nOutput:\n0'
        self.assertEqual(final_output_label(text), 0)
        self.assertIsNone(strict_label(text))

    def test_invalid_labels_are_missing_not_zero(self):
        for text in ('no output here', 'Output:\n', 'Output:\n2', 'Output: 1 because', 'Output:\nYes'):
            self.assertIsNone(final_output_label(text), text)

    def test_split_answer(self):
        before, field = split_answer('a\nOutput:\n1')
        self.assertEqual((before, field), ('a\n', 'Output:\n1'))
        self.assertEqual(split_answer('a'), ('a', None))


class SlotTests(unittest.TestCase):
    def test_every_status_and_note(self):
        p = parse_slots(G_OK)
        self.assertEqual(statuses(p), ['boxes', 'empty', 'boxes', 'unknown', 'boxes', 'boxes', 'boxes', 'empty'])
        self.assertEqual(p['slots'][2]['boxes'], [[10, 20, 30, 40], [500, 500, 600, 650]])
        self.assertIn('internvl_tags', p['slots'][4]['notes'])
        self.assertIn('flat_box', p['slots'][5]['notes'])
        self.assertEqual(p['slots'][5]['whole_image'], [True])
        self.assertEqual(p['slots'][6]['whole_image'], [True])   # 99 % area
        self.assertEqual(p['slots'][0]['whole_image'], [False])
        self.assertTrue(p['heading'])
        self.assertEqual(p['response_status'], 'ok')

    def test_missing_duplicate_out_of_order_and_extra(self):
        p = parse_slots('Frame3: []\nFrame1: [[1, 2, 3, 4]]\nFrame3: unknown\nFrame9: []')
        self.assertEqual(statuses(p)[:4], ['boxes', 'missing', 'duplicate', 'missing'])
        self.assertEqual(p['extra_frame_lines'], ['Frame9: []'])

    def test_malformed_contents(self):
        for content in ('none', 'No firearm', '[[1, 2, 3]]', '[[1, 2, 3, 4001]]', '[[50, 2, 40, 4]]',
                        '[[1, 2, 3, 4], [9, 9, 1, 1]]', '[[x1, y1, x2, y2]]', '[[-1, 2, 3, 4]]', '[[1, 2, 3, 4] extra'):
            self.assertEqual(statuses(parse_slots('Frame1: ' + content))[0], 'malformed', content)

    def test_floats_kept_and_unit_scale_flagged(self):
        p = parse_slots('Frame1: [[0.1, 0.2, 0.3, 0.4]]')
        self.assertEqual(p['slots'][0]['boxes'], [[0.1, 0.2, 0.3, 0.4]])
        self.assertTrue(p['scale_suspect'])
        self.assertFalse(parse_slots(G_OK)['scale_suspect'])

    def test_truncated_answers_are_never_empty_boxes(self):
        p = parse_slots('Frame1: [[1, 2, 3, 4]]\nFrame2: [[5, 6', truncated=True)
        self.assertEqual(set(statuses(p)), {'truncated'})
        self.assertEqual(p['response_status'], 'truncated')

    def test_no_slot_lines(self):
        p = parse_slots('I cannot help with that.')
        self.assertEqual((p['response_status'], set(statuses(p))), ('no_slots', {'missing'}))

    def test_markdown_decoration_and_special_tokens(self):
        p = parse_slots('<s> - **Frame1:** [[1, 2, 3, 4]]\n  Frame 2 : []<|im_end|>')
        self.assertEqual(statuses(p)[:2], ['boxes', 'empty'])

    def test_slot_lines_after_the_output_field_are_counted(self):
        p = parse_slots('Output: 1\nFrame1: [[1, 2, 3, 4]]')
        self.assertEqual(p['slot_lines_after_output'], 1)
        self.assertEqual(parse_slots('Frame1: []\nOutput:\n1')['slot_lines_after_output'], 0)
