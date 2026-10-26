"""Grounding measures on the frozen InternVL2: where the decision looks, and what it relies on.

Pre-registered in docs/superpowers/specs/2026-10-26-firearm-grounding-design.md.

- coverage(box): the fraction of each 16x16 image-token cell inside a 0-1000 box. InternVL2-8B
  turns a 448x448 tile into 32x32 patches of 14 px and pixel-shuffles them (ps_version v2) into a
  16x16 grid; token (r, c) covers rows [62.5r, 62.5(r+1)) and columns [62.5c, 62.5(c+1)).
- control_box(box): an equal-sized box at the horizontal mirror, else the vertical mirror, else
  the non-overlapping position closest to the centre (scan order fixed, no randomness).
- decision_split(...): the answer cut where the category name starts after the last 'Category:'.
- Scorer: replays prompt + the model's own answer prefix with eager attention in every layer and
  returns the decision position's attention to each frame's 256 image tokens (mean over heads and
  selected layers) and the normalised probabilities of the three category names, read with an FP32
  head on the final hidden states. Generation itself is unchanged (FlashAttention, greedy).
"""
import contextlib
import importlib
import math
import re
import numpy as np

GRID = 16
TOKENS_PER_FRAME = GRID * GRID
CELL = 1000.0 / GRID
CATEGORY_FIELD = re.compile(r'category\s*:', re.I)


def coverage(box):
    """16x16 array: share of each token cell (rows = y, columns = x) covered by the box."""
    x1, y1, x2, y2 = box
    edges = np.arange(GRID + 1) * CELL
    over = lambda a, b: np.clip(np.minimum(edges[1:], b) - np.maximum(edges[:-1], a), 0, None) / CELL
    return np.outer(over(y1, y2), over(x1, x2))


def boxes_overlap(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def control_box(box):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    candidates = [[1000 - x2, y1, 1000 - x1, y2], [x1, 1000 - y2, x2, 1000 - y1]]
    for c in candidates:
        if not boxes_overlap(c, box):
            return c
    best = None
    for top in np.arange(0, 1000 - h + 1e-9, CELL / 2):
        for left in np.arange(0, 1000 - w + 1e-9, CELL / 2):
            c = [float(left), float(top), float(left + w), float(top + h)]
            if boxes_overlap(c, box):
                continue
            d = (left + w / 2 - 500) ** 2 + (top + h / 2 - 500) ** 2
            if best is None or d < best[0]:
                best = (d, c)
    return best[1] if best else None


def mask_image(image, boxes, rgb=(127, 127, 127)):
    """Copy of a 448x448 PIL image with each 0-1000 box filled grey (all pixels whose centres lie inside)."""
    from PIL import ImageDraw
    out = image.copy()
    draw = ImageDraw.Draw(out)
    s = out.width / 1000.0
    for x1, y1, x2, y2 in boxes:
        left, top = math.ceil(x1 * s - 0.5), math.ceil(y1 * s - 0.5)
        right, bottom = math.ceil(x2 * s - 0.5) - 1, math.ceil(y2 * s - 0.5) - 1
        if right >= left and bottom >= top:
            draw.rectangle([left, top, right, bottom], fill=tuple(rgb))
    return out


def lift(frame_attention, boxes_by_frame):
    """Pooled lift over boxed frames: sum(inside) / sum(chance). frame_attention: {k: (256,) normalised}."""
    inside = chance = 0.0
    for k, boxes in boxes_by_frame.items():
        cov = np.clip(sum(coverage(b) for b in boxes), 0, 1).reshape(-1)
        inside += float(np.dot(frame_attention[k], cov))
        chance += float(cov.sum() / TOKENS_PER_FRAME)
    return inside / chance if chance > 0 else None


def frame_lifts(frame_attention, boxes_by_frame):
    out = {}
    for k, boxes in boxes_by_frame.items():
        cov = np.clip(sum(coverage(b) for b in boxes), 0, 1).reshape(-1)
        chance = cov.sum() / TOKENS_PER_FRAME
        out[k] = float(np.dot(frame_attention[k], cov) / chance) if chance > 0 else None
    return out


def decision_split(pieces, categories):
    """pieces: decoded text of each generated token. Returns (j, leading_whitespace) where token j starts the
    category name after the last 'Category:' field, or None when there is no such field or name."""
    text = ''.join(pieces)
    fields = list(CATEGORY_FIELD.finditer(text))
    if not fields:
        return None
    end = fields[-1].end()
    pos = 0
    for j, piece in enumerate(pieces):
        nxt = pos + len(piece)
        if nxt > end and piece[max(0, end - pos):].strip():
            if pos < end:  # one token holds both the colon and the name: no clean decision position
                return None
            stripped = piece.lstrip()
            return j, piece[:len(piece) - len(stripped)]
        pos = nxt
    return None


@contextlib.contextmanager
def eager_attention(language_model):
    """InternLM2 with eager attention in every layer (same weights); restores FlashAttention on exit."""
    model = language_model.model
    base = next(c for c in type(model.layers[0].attention).__mro__ if c.__name__ == 'InternLM2Attention')
    saved = (model.config.attn_implementation, [layer.attention.__class__ for layer in model.layers])
    model.config.attn_implementation = 'eager'
    for layer in model.layers:
        layer.attention.__class__ = base
    try:
        yield
    finally:
        model.config.attn_implementation = saved[0]
        for layer, cls in zip(model.layers, saved[1]):
            layer.attention.__class__ = cls


class Scorer:
    """Generation and measurement on a loaded runner InternVL2 backend (one video at a time)."""
    def __init__(self, backend, generation, categories):
        self.b, self.torch = backend, backend.torch
        self.model, self.tok = backend.model, backend.tokenizer
        self.generation = dict(generation)
        self.categories = list(categories)
        self.lm = self.model.language_model
        module = importlib.import_module(type(self.model).__module__)
        self.get_conv_template = module.get_conv_template
        self.img_context = self.tok.convert_tokens_to_ids('<IMG_CONTEXT>')

    def query_ids(self, question, n_frames):
        """Exactly batch_chat's query for one video: all frames' tokens at the first <image>."""
        template = self.get_conv_template(self.model.template)
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        image_tokens = '<img>' + '<IMG_CONTEXT>' * self.model.num_image_token * n_frames + '</img>'
        query = query.replace('<image>', image_tokens, 1)
        return self.tok(query, return_tensors='pt').input_ids, template.sep.strip()

    def generate(self, images, question):
        """Greedy answer and its token ids (equivalent to batch_chat for one video)."""
        ids, sep = self.query_ids(question, len(images))
        pixels = self.b.pixels(images)
        generation = {**self.generation, 'eos_token_id': self.tok.convert_tokens_to_ids(sep)}
        with self.torch.inference_mode():
            out = self.model.generate(pixel_values=pixels, input_ids=ids.to(self.b.device),
                                      attention_mask=self.torch.ones_like(ids).to(self.b.device), **generation)
        gen = out[0].tolist()
        text = self.tok.decode(gen, skip_special_tokens=True).split(sep)[0].strip()
        return {'text': text, 'ids': gen, 'query_ids': ids[0].tolist()}

    def _embeds(self, images, ids):
        with self.torch.inference_mode():
            vit = self.model.extract_feature(self.b.pixels(images))
            emb = self.lm.get_input_embeddings()(ids)
            flat = emb.reshape(-1, emb.shape[-1])
            sel = (ids.reshape(-1) == self.img_context)
            flat[sel] = vit.reshape(-1, flat.shape[-1]).to(flat.dtype)
        return flat.reshape(emb.shape)

    def score(self, images, query_ids, gen_ids, layers=None, attention=True, eager=True):
        """p over the categories at the decision position, and per-frame attention from that position.

        Returns None when the answer has no 'Category:' field followed by a name."""
        torch = self.torch
        pieces = [self.tok.decode(gen_ids[:i + 1], skip_special_tokens=True)[len(self.tok.decode(gen_ids[:i], skip_special_tokens=True)):]
                  for i in range(len(gen_ids))]
        split = decision_split(pieces, self.categories)
        if split is None:
            return None
        j, lead = split
        prefix = query_ids + [t for t in gen_ids[:j] if t != self.tok.bos_token_id]
        ids = torch.tensor([prefix], device=self.b.device)
        img = [i for i, t in enumerate(prefix) if t == self.img_context]
        n_frames = len(img) // TOKENS_PER_FRAME
        rows = {}
        hooks = []
        q = len(prefix) - 1

        def keep(index):
            def hook(module, args, output):
                rows[index] = output[1][0, :, q, img[0]:img[-1] + 1].float().mean(0).cpu().numpy()
                return (output[0], None, output[2])
            return hook
        with torch.inference_mode(), (eager_attention(self.lm) if eager or attention else contextlib.nullcontext()):
            if attention:
                hooks = [layer.attention.register_forward_hook(keep(i)) for i, layer in enumerate(self.lm.model.layers)]
            try:
                out = self.lm.model(inputs_embeds=self._embeds(images, ids), output_attentions=attention, use_cache=True, return_dict=True)
            finally:
                for h in hooks:
                    h.remove()
            head = self.lm.output.weight.float()
            first = torch.log_softmax(out.last_hidden_state[0, -1].float() @ head.T, -1)
            logp = []
            for name in self.categories:
                cand = self.tok(lead + name, add_special_tokens=False).input_ids
                total = float(first[cand[0]])
                if len(cand) > 1:
                    cont = self.lm.model(inputs_embeds=self.lm.get_input_embeddings()(torch.tensor([cand[:-1]], device=self.b.device)),
                                         past_key_values=out.past_key_values, use_cache=False, return_dict=True)
                    lp = torch.log_softmax(cont.last_hidden_state[0].float() @ head.T, -1)
                    total += float(sum(lp[i, cand[i + 1]] for i in range(len(cand) - 1)))
                logp.append(total)
        logp = np.array(logp)
        probs = np.exp(logp - np.logaddexp.reduce(logp))
        result = {'decision_token': j, 'lead': lead, 'category_logp': dict(zip(self.categories, logp.tolist())),
                  'category_p': dict(zip(self.categories, probs.tolist()))}
        if attention:
            use = layers if layers is not None else sorted(rows)
            mean = np.mean([rows[i] for i in use], axis=0)
            per = mean.reshape(n_frames, TOKENS_PER_FRAME)
            result['frame_attention'] = {k: (per[k] / per[k].sum()).tolist() for k in range(n_frames)}
            result['frame_mass'] = per.sum(1).tolist()
            result['layer_rows'] = {i: rows[i].reshape(n_frames, TOKENS_PER_FRAME) for i in rows}
        return result
