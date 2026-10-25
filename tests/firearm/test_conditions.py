import unittest
from vera_runner.backends import prompt
from vera_runner.conditions import MARKER, OUTPUT_MARKER, build_prompt, frame_prefix, load_texts, prompt_texts
from vera_runner.config import DEFAULTS
import copy

PROMPTS = 'studies/firearm/prompts'


class ConditionPromptTests(unittest.TestCase):
    def setUp(self):
        self.cfg = copy.deepcopy(DEFAULTS)
        self.texts = prompt_texts(self.cfg, PROMPTS)

    def test_a_is_vera_unchanged(self):
        self.assertEqual(self.texts['A'], prompt(self.cfg))
        self.assertEqual(self.texts['A'].count(MARKER), 1)

    def test_ag_is_a_plus_two_declared_insertions(self):
        a, ag = self.texts['A'], self.texts['AG']
        texts = load_texts(PROMPTS)
        start = a.index(MARKER)
        insert = ag[start:ag.index(MARKER)]
        self.assertTrue(a[:start].endswith('5. Are there any unusual sounds or noises in the video that suggest an anomaly?\n'))
        self.assertTrue(insert.startswith('6. Firearm localization:') and insert.endswith('\n'))
        lines = texts['format']
        self.assertEqual(ag.count(lines), 1)
        self.assertTrue(ag.index(lines) > ag.index('Answers to Prompt Questions:'))
        self.assertTrue(ag.endswith(a[a.index(OUTPUT_MARKER):]))  # the final Output field and closing text unchanged
        self.assertEqual(ag.replace(insert, '', 1).replace(lines, '', 1), a)

    def test_g_and_ag_share_the_grounding_paragraph_and_packing(self):
        core = load_texts(PROMPTS)['core'].strip('\n')
        for c in ('G', 'AG'):
            self.assertEqual(self.texts[c].count(core), 1)
            self.assertEqual(self.texts[c].count(load_texts(PROMPTS)['format']), 1)
            self.assertIn('** Input: **\n[' + frame_prefix(8) + ']', self.texts[c])
            self.assertEqual(self.texts[c].count('<image>'), 8)
        self.assertNotIn('anomaly in the video', self.texts['G'])

    def test_no_dataset_information_reaches_any_prompt(self):
        # 'handguns and long guns' is the task's own wording; dataset class names and labels are not.
        for c, text in self.texts.items():
            for word in ('Machine_Gun', 'No_Gun', 'Handgun__', 'annotation', 'label.json', 'category', 'ground truth', 'PAH', 'PCW'):
                self.assertNotIn(word.lower(), text.lower(), (c, word))

    def test_unknown_condition_and_missing_marker_fail(self):
        with self.assertRaises(ValueError):
            build_prompt(self.cfg, 'B', PROMPTS)
        cfg = copy.deepcopy(self.cfg)
        cfg['prompt']['guiding_questions'] = ['1. Q?']
        self.assertTrue(build_prompt(cfg, 'AG', PROMPTS).count(MARKER) == 1)
