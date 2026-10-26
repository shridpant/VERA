"""Model access for the firearm package: InternVL (2 and 3.5, their own remote code) and Qwen (Qwen3-VL, Qwen3.5).

Every call returns the decoded text with special tokens kept, the generated token count, the seconds and
whether a thinking trace was cut by the token limit. torch/transformers are imported only when a model is
loaded, so the parsers and the runners' bookkeeping run (and are tested) without them.

Thinking (package plan, "Models"): off = greedy; on = the model's recommended sampling with a seed per call.
InternVL3.5 thinks when its system message is the R1 protocol (model card); Qwen3.5 through the chat
template's enable_thinking; Qwen3-VL through its separate Thinking checkpoint.
"""
import time

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# InternVL3.5 model card (README.md of revision 9bb6a56a), verbatim.
R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()
END_TOKENS = ('<|im_end|>', '<|endoftext|>', '</s>', '<|end_of_text|>')


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """InternVL model card."""
    best_ratio_diff, best_ratio, area = float('inf'), (1, 1), width * height
    for ratio in target_ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_ratio_diff:
            best_ratio_diff, best_ratio = diff, ratio
        elif diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    """InternVL model card: split into up to max_num 448 tiles at the closest aspect ratio, plus a thumbnail."""
    width, height = image.size
    ratios = sorted({(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
                     if min_num <= i * j <= max_num}, key=lambda x: x[0] * x[1])
    ratio = find_closest_aspect_ratio(width / height, ratios, width, height, image_size)
    tw, th, blocks = image_size * ratio[0], image_size * ratio[1], ratio[0] * ratio[1]
    resized = image.resize((tw, th))
    tiles = [resized.crop(((i % (tw // image_size)) * image_size, (i // (tw // image_size)) * image_size,
                           ((i % (tw // image_size)) + 1) * image_size, ((i // (tw // image_size)) + 1) * image_size))
             for i in range(blocks)]
    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))
    return tiles


def split_thinking(text, thinking):
    """(thinking trace, answer, truncated). A thinking call whose trace never closed has no answer."""
    for token in END_TOKENS:
        text = text.replace(token, '')
    text = text.strip()
    text = text[len('<s>'):] if text.startswith('<s>') else text
    if '</think>' in text:
        trace, answer = text.rsplit('</think>', 1)
        return trace.replace('<think>', '').strip(), answer.strip(), False
    if thinking or '<think>' in text:
        return text.replace('<think>', '').strip(), '', True
    return '', text.strip(), False


class InternVL:
    """InternVL2 or InternVL3.5 through its remote code. device_map='auto' shards a large model over the visible GPUs."""

    def __init__(self, path, thinking=False, flash=False, device_map=None, temperature=0.6):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.thinking, self.temperature = torch, thinking, temperature
        kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True, use_flash_attn=flash)
        if device_map:
            kwargs['device_map'] = device_map
            if device_map == 'auto':
                # A16 automatic placement otherwise fills cards to their capacity and
                # can OOM while loading shards. Reserve vision/KV/allocator headroom.
                gib = 1024 ** 3
                kwargs['max_memory'] = {
                    i: min(int(torch.cuda.mem_get_info(i)[0] * 0.9), (12 if i == 0 else 13) * gib)
                    for i in range(torch.cuda.device_count())
                }
        self.model = AutoModel.from_pretrained(path, **kwargs).eval()
        if not device_map:
            self.model = self.model.to('cuda')
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        if thinking:
            self.model.system_message = R1_SYSTEM_PROMPT
        self.device = next(self.model.vision_model.parameters()).device
        original = self.model.generate

        def captured(*args, **kwargs):
            out = original(*args, **kwargs)
            self.last_ids = out[0].detach().cpu().tolist()
            return out
        self.model.generate = captured

    def pixels(self, images):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        transform = T.Compose([T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC), T.ToTensor(),
                               T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        return self.torch.stack([transform(im.convert('RGB')) for im in images]).to(device=self.device, dtype=self.torch.bfloat16)

    def generation(self, max_new_tokens):
        if self.thinking:
            return dict(do_sample=True, temperature=self.temperature, max_new_tokens=max_new_tokens)
        return dict(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens)

    def chat(self, images_per_slot, question, max_new_tokens, seed=0, packed=False):
        """images_per_slot: a list of image lists, one per <image> in question (tiles of one image, or one image each).
        packed=True reproduces VERA: every image as patches of the FIRST <image> (batch_chat, num_patches_list=[n])."""
        flat = [im for slot in images_per_slot for im in slot]
        pixels = self.pixels(flat)
        self.torch.manual_seed(seed)
        begin = time.monotonic()
        with self.torch.inference_mode():
            if packed:
                self.model.batch_chat(self.tokenizer, pixels, num_patches_list=[len(flat)], questions=[question],
                                      generation_config=self.generation(max_new_tokens))
            else:
                self.model.chat(self.tokenizer, pixels, question, self.generation(max_new_tokens),
                                num_patches_list=[len(slot) for slot in images_per_slot])
        seconds = time.monotonic() - begin
        text = self.tokenizer.decode(self.last_ids, skip_special_tokens=False)
        trace, answer, truncated = split_thinking(text, self.thinking)
        return {'text': text, 'thinking': trace, 'answer': answer, 'truncated': truncated,
                'tokens': len(self.last_ids), 'seconds': seconds}

    def input_tokens(self, question, n_patches):
        return len(self.tokenizer(question).input_ids) + n_patches * (self.model.num_image_token + 2)


class Qwen:
    """Qwen3-VL (Instruct or Thinking checkpoint) or Qwen3.5 (enable_thinking) with transformers' native classes."""

    def __init__(self, path, thinking=False, family='qwen3vl', max_pixels=1024 * 1024):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch, self.thinking, self.family, self.max_pixels = torch, thinking, family, max_pixels
        self.model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='sdpa',
                                                                 device_map='cuda').eval()
        self.processor = AutoProcessor.from_pretrained(path)
        self.sampling = {k: getattr(self.model.generation_config, k, None) for k in ('do_sample', 'temperature', 'top_p', 'top_k')}

    def fit(self, image):
        w, h = image.size
        scale = min(1.0, (self.max_pixels / (w * h)) ** 0.5)
        return image.convert('RGB') if scale == 1.0 else image.convert('RGB').resize((round(w * scale), round(h * scale)))

    def chat(self, images, question, max_new_tokens, seed=0):
        content = [{'type': 'image', 'image': self.fit(im)} for im in images] + [{'type': 'text', 'text': question}]
        kwargs = {'enable_thinking': self.thinking} if self.family == 'qwen35' else {}
        inputs = self.processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors='pt', **kwargs).to(self.model.device)
        gen = dict(max_new_tokens=max_new_tokens)
        if self.thinking:  # the checkpoint's own recommended sampling (generation_config.json)
            gen.update({k: v for k, v in self.sampling.items() if v is not None})
            gen['do_sample'] = True
        else:
            gen.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        self.torch.manual_seed(seed)
        begin = time.monotonic()
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, **gen)
        seconds = time.monotonic() - begin
        ids = out[0][inputs['input_ids'].shape[1]:].tolist()
        text = self.processor.decode(ids, skip_special_tokens=False)
        trace, answer, truncated = split_thinking(text, self.thinking)
        return {'text': text, 'thinking': trace, 'answer': answer, 'truncated': truncated, 'tokens': len(ids),
                'seconds': seconds, 'input_tokens': int(inputs['input_ids'].shape[1])}
