"""Scorer on a fake InternVL2/InternLM2 (Phase 0(a)): the frame-token attention rows, the eager switch, the
category scores through the KV-cache continuation, and the decision exclusions, against independent
references (full forwards without a cache; attention weights kept by the layers themselves)."""
import math
import types
import unittest
import numpy as np
import torch
import torch.nn as nn
from vera_runner.reliance import Scorer, eager_attention

D, H, L = 16, 2, 3
SPECIAL = {'<s>': 1, '<IMG_CONTEXT>': 2, '<|im_end|>': 3}
WORDS = ['Category', ':', ' Hand', 'gun', ' Machine', '_G', 'un', ' No', 'Answers', ' x', '\n', 'Hand', 'No', 'Machine', ' y',
         '<img>', '</img>', 'Q', ' **', '**', 'g']
VOCAB = {**SPECIAL, **{w: i + 4 for i, w in enumerate(WORDS)}}
INV = {i: w for w, i in VOCAB.items()}
CATS = ['Handgun', 'Machine_Gun', 'No_Gun']


class Tok:
    bos_token_id = 1

    def convert_tokens_to_ids(self, t):
        return VOCAB[t]

    def encode(self, text):
        ids, i, keys = [], 0, sorted(VOCAB, key=len, reverse=True)
        while i < len(text):
            k = next(k for k in keys if text.startswith(k, i))
            ids.append(VOCAB[k])
            i += len(k)
        return ids

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = self.encode(text)
        return types.SimpleNamespace(input_ids=torch.tensor([ids]) if return_tensors == 'pt' else ids)

    def decode(self, ids, skip_special_tokens=True):
        return ''.join('' if (skip_special_tokens and i in SPECIAL.values()) else INV[i] for i in ids)


class InternLM2Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv, self.o, self.last_w = nn.Linear(D, 3 * D), nn.Linear(D, D), None

    def forward(self, h, past_key_value=None, output_attentions=False, use_cache=False, offset=0):
        b, n, _ = h.shape
        q, k, v = self.qkv(h).split(D, -1)
        pos = torch.arange(offset, offset + n, dtype=torch.float32)[None, :, None]
        q, k = q + torch.sin(pos * 0.37), k + torch.cos(pos * 0.21)
        if past_key_value is not None:
            k, v = torch.cat([past_key_value[0], k], 1), torch.cat([past_key_value[1], v], 1)
        cache = (k, v) if use_cache else None
        qh, kh, vh = (t.view(b, -1, H, D // H).transpose(1, 2) for t in (q, k, v))
        s = qh @ kh.transpose(-1, -2) / math.sqrt(D // H)
        m = k.shape[1]
        s = s.masked_fill(torch.arange(m)[None, :] > (torch.arange(n)[:, None] + (m - n)), float('-inf'))
        w = s.softmax(-1)
        self.last_w = w.detach().clone()
        return self.o((w @ vh).transpose(1, 2).reshape(b, n, D)), (w if output_attentions else None), cache


class InternLM2FlashAttention2(InternLM2Attention):
    def forward(self, h, past_key_value=None, output_attentions=False, use_cache=False, offset=0):
        out, _, cache = super().forward(h, past_key_value, False, use_cache, offset)
        return out, None, cache


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention, self.mlp = InternLM2FlashAttention2(), nn.Linear(D, D)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer() for _ in range(L)])
        self.config = types.SimpleNamespace(attn_implementation='flash_attention_2')

    def forward(self, inputs_embeds, past_key_values=None, output_attentions=False, use_cache=False, return_dict=True):
        off = past_key_values[0][0].shape[1] if past_key_values else 0
        h, cache = inputs_embeds, []
        for i, layer in enumerate(self.layers):
            o = layer.attention(h, past_key_value=past_key_values[i] if past_key_values else None,
                                output_attentions=output_attentions, use_cache=use_cache, offset=off)
            h = h + o[0]
            h = h + torch.tanh(layer.mlp(h))
            cache.append(o[2])
        return types.SimpleNamespace(last_hidden_state=h, past_key_values=tuple(cache) if use_cache else None)


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model, self.output, self.emb = Model(), nn.Linear(D, len(VOCAB), bias=False), nn.Embedding(len(VOCAB), D)

    def get_input_embeddings(self):
        return self.emb


class InternVLChatModel(nn.Module):
    template, system_message, num_image_token = 't', 's', 256

    def __init__(self):
        super().__init__()
        self.language_model, self.proj, self.img_context_token_id = LM(), nn.Linear(3, D), None

    def extract_feature(self, px):
        return self.proj(px.view(px.shape[0], 256, 3))


def get_conv_template(name):
    raise NotImplementedError


class ScorerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        backend = types.SimpleNamespace(torch=torch, model=InternVLChatModel().eval(), tokenizer=Tok(), device='cpu',
                                        pixels=lambda images: torch.stack([torch.as_tensor(im) for im in images]))
        self.sc, self.tok = Scorer(backend, {}, CATS), Tok()
        self.images = [torch.randn(256 * 3) for _ in range(2)]
        self.query = self.tok.encode('Q') + [VOCAB['<img>']] + [2] * 512 + [VOCAB['</img>']] + self.tok.encode('Q\n')

    def reference_logp(self, prefix, lead):
        lm, out = self.sc.lm, {}
        for name in CATS:
            cand = self.tok.encode(lead + name)
            with torch.no_grad():
                hidden = lm.model(inputs_embeds=self.sc._embeds(self.images, torch.tensor([prefix + cand]))).last_hidden_state[0]
                lp = torch.log_softmax(hidden.float() @ lm.output.weight.float().T, -1)
            out[name] = float(sum(lp[len(prefix) - 1 + i, c] for i, c in enumerate(cand)))
        return out

    def test_scores_and_attention_match_independent_references(self):
        lm = self.sc.lm
        for answer, lead, written in (('Answers x y\nCategory: Machine_Gun\n', ' ', 'Machine_Gun'),
                                      ('Answers x\nCategory:\nHandgun', '', 'Handgun'), ('Category: No_Gun', ' ', 'No_Gun')):
            gen = [1] + self.tok.encode(answer) + [3]
            for eager in (True, False):
                r = self.sc.score(self.images, self.query, gen, attention=eager, eager=eager)
                self.assertEqual((r['lead'], r['written']), (lead, written))
                self.assertEqual([type(layer.attention).__name__ for layer in lm.model.layers], ['InternLM2FlashAttention2'] * L)
                self.assertEqual(lm.model.config.attn_implementation, 'flash_attention_2')
                prefix = self.query + [t for t in gen[:r['decision_token']] if t != 1]
                self.assertEqual(INV[gen[r['decision_token']]].strip(), written[:len(INV[gen[r['decision_token']]].strip())])
                ref = self.reference_logp(prefix, lead)
                self.assertLess(max(abs(ref[c] - r['category_logp'][c]) for c in CATS), 1e-4)
                self.assertAlmostEqual(r['candidate_mass'], sum(math.exp(v) for v in ref.values()), places=4)
                if eager:
                    with torch.no_grad(), eager_attention(lm):
                        lm.model(inputs_embeds=self.sc._embeds(self.images, torch.tensor([prefix])), output_attentions=True)
                    img = [i for i, t in enumerate(prefix) if t == 2]
                    rows = np.mean([layer.attention.last_w[0, :, -1, img[0]:img[-1] + 1].mean(0).numpy() for layer in lm.model.layers], 0)
                    rows = rows.reshape(2, 256)
                    got = np.array([r['frame_attention'][k] for k in range(2)])
                    np.testing.assert_allclose(got, rows / rows.sum(1, keepdims=True), atol=1e-6)

    def test_exclusions(self):
        bold = [1] + self.tok.encode('Category: **Handgun**') + [3]
        self.assertIsNone(self.sc.score(self.images, self.query, bold))
        self.assertEqual(self.sc.last_exclusion, 'no_name')
        split = [1] + self.tok.encode('Category:') + [VOCAB[' Hand'], VOCAB['g'], VOCAB['un'], 3]  # ' Hand' 'g' 'un' != ' Hand' 'gun'
        self.assertIsNone(self.sc.score(self.images, self.query, split))
        self.assertEqual(self.sc.last_exclusion, 'token_mismatch')
        self.assertIsNotNone(self.sc.score(self.images, self.query, [1] + self.tok.encode('Category: Handgun'), attention=False, eager=False))
        self.assertIsNone(self.sc.last_exclusion)


if __name__ == '__main__':
    unittest.main()
