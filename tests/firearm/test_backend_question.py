import copy
import unittest
from types import SimpleNamespace
import torch
from PIL import Image
from vera_runner.backends import InternVL2, Mock, generated, prompt
from vera_runner.conditions import prompt_texts
from vera_runner.config import DEFAULTS
from vera_runner.slots import final_output_label, parse_slots


class Tokenizer:
    bos_token_id = 1
    def convert_tokens_to_ids(self, token):
        return {'<|im_end|>': 92542}[token]
    def decode(self, ids, skip_special_tokens=True):
        return ' '.join(map(str, ids))


def fake_backend(cfg, output_ids):
    """InternVL2 without weights: batch_chat records its arguments, as generate() would record ids."""
    backend = InternVL2.__new__(InternVL2)
    calls = []
    def batch_chat(tokenizer, pixel_values, num_patches_list, questions, generation_config):
        calls.append({'pixels': pixel_values.clone(), 'num_patches_list': num_patches_list, 'questions': questions,
                      'generation_config': dict(generation_config)})
        backend.last_ids, backend.last_output_ids = [[5, 6]], output_ids
        return ['Output:\n0']
    backend.__dict__.update(cfg=cfg, torch=torch, device='cpu', dtype=torch.float32, tokenizer=Tokenizer(),
                            question=prompt(cfg), model=SimpleNamespace(batch_chat=batch_chat))
    return backend, calls


class QuestionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = copy.deepcopy(DEFAULTS)
        self.images = [Image.new('RGB', (448, 448), (i * 30, 0, 0)) for i in range(8)]

    def test_a_call_is_unchanged_and_g_uses_the_same_images_and_packing(self):
        backend, calls = fake_backend(self.cfg, [1, 7, 92542])
        backend.predict(self.images)
        g = prompt_texts(self.cfg, 'studies/firearm/prompts')['G']
        backend.predict(self.images, question=g)
        self.assertEqual(calls[0]['questions'], [prompt(DEFAULTS)])
        self.assertEqual(calls[1]['questions'], [g])
        self.assertEqual(calls[0]['num_patches_list'], [8])
        self.assertEqual(calls[1]['num_patches_list'], [8])
        self.assertTrue(torch.equal(calls[0]['pixels'], calls[1]['pixels']))
        self.assertEqual(calls[0]['generation_config'], calls[1]['generation_config'])
        self.assertEqual(calls[0]['generation_config'], DEFAULTS['generation'])

    def test_eos_and_token_count(self):
        tok = Tokenizer()
        done = generated(tok, [1, 7, 8, 92542], 0.5, None)
        self.assertEqual((done['eos'], done['generated_tokens']), (True, 3))
        cut = generated(tok, [1] + [7] * 1024, 0.5, None)
        self.assertEqual((cut['eos'], cut['generated_tokens']), (False, 1024))

    def test_mock_answers_each_condition_in_its_own_format(self):
        mock = Mock(self.cfg)
        texts = prompt_texts(self.cfg, 'studies/firearm/prompts')
        bright = [Image.new('RGB', (8, 8), (200, 200, 200))] * 4 + [Image.new('RGB', (8, 8))] * 4
        a, g, ag = (mock.predict(bright, question=texts[c])['raw_response'] for c in ('A', 'G', 'AG'))
        self.assertEqual(final_output_label(a), 1)
        self.assertNotIn('Frame1', a)
        self.assertIsNone(final_output_label(g))
        self.assertEqual([s['status'] for s in parse_slots(g)['slots']], ['boxes'] * 4 + ['empty'] * 4)
        self.assertEqual((final_output_label(ag), parse_slots(ag)['slots'][0]['status']), (1, 'boxes'))
