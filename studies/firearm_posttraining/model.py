"""Native differentiable Qwen path, completion masks, and an explicit LoRA reference."""
import hashlib
import json
from pathlib import Path

from .data import letterbox

DEFAULT_QUESTIONS = ('What is the person in the video holding?',
                     'How is the person moving or positioning their arms and body?')

def instruction(book, grounded=False, questions=None):
    if questions is None:
        questions = DEFAULT_QUESTIONS
    labels = '\n'.join(f'{code}: {v["category"]}, {v["group"]}, {v["meaning"]}' for code, v in sorted(book['codes'].items()))
    text = ('Classify the person and action in these eight chronological video frames. '
            'Return only one JSON object with string fields category, action_group, action_code. '
            'Use these exact category names: Handgun (pistol), Machine_Gun (long gun), No_Gun. '
            'Use an action code from the following codebook; the category and group must agree with it.\n' + labels)
    if grounded:
        text += ('\nAlso output grounding: an array of eight objects, each with frame_id (integer 0 through 7) '
                 'and boxes (list of [x1,y1,x2,y2] firearm boxes in that displayed frame). '
                 'Coordinates are in [0,1] relative to the entire displayed square, including black padding. '
                 'Use an empty list when no firearm is visible. Include every frame exactly once.')
    if questions:
        if len(questions) > 5:
            raise ValueError('maximum five guiding questions')
        text += '\nUse these visual checks before choosing the labels:\n' + '\n'.join(questions)
    return text


def target_parts(row, grounded=False):
    """Character spans carry label=1, known-box=2, unknown=0 supervision."""
    pieces = [(json.dumps({k: row[k] for k in ('category', 'action_group', 'action_code')}, separators=(',', ':'))[:-1], 1)]
    if grounded:
        pieces.append((',"grounding":[', 1))
        for i, v in enumerate(row['views']):
            boxes = None if v['boxes'] is None else [[round(x, 5) for x in b] for b in v['boxes']]
            field = json.dumps({'frame_id': v['frame_id'], 'boxes': boxes}, separators=(',', ':'))
            pieces.append(((',' if i else '') + field, 0 if boxes is None else 2))
        pieces.append((']', 1))
    pieces.append(('}', 1))
    text, spans = '', []
    for piece, kind in pieces:
        spans.append((len(text), len(text) + len(piece), kind))
        text += piece
    return text, spans


def token_kinds(offsets, spans):
    kinds = []
    for start, end in offsets:
        kinds.append(next((kind for a, b, kind in spans if start >= a and end <= b), 0))
    return kinds


class NativeModel:
    def __init__(self, model_path, book, adapter=None, seed=20261026):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed
        from peft import LoraConfig, get_peft_model, PeftModel
        if torch.cuda.device_count() != 1:
            raise RuntimeError('exactly one visible CUDA device required per experiment')
        set_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.torch, self.book, self.path = torch, book, model_path
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = 'left'
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, attn_implementation='sdpa',
            local_files_only=True, device_map={'': 0})
        targets = [n for n, m in base.named_modules() if '.language_model.' in n and
                   n.rsplit('.', 1)[-1] in ('q_proj', 'k_proj', 'v_proj', 'o_proj') and isinstance(m, torch.nn.Linear)]
        if not targets or any('visual' in n for n in targets):
            raise RuntimeError('cannot isolate language attention LoRA targets')
        self.targets = targets
        config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0., bias='none',
                            target_modules=targets, task_type='CAUSAL_LM')
        self.model = (PeftModel.from_pretrained(base, adapter, is_trainable=True) if adapter else get_peft_model(base, config))
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.model.enable_input_require_grads()
        self.has_reference = False
        self.assert_trainable()

    def assert_trainable(self):
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]
        if not names or any('lora_' not in n or '.default.' not in n or '.language_model.' not in n for n in names):
            raise RuntimeError(f'unexpected trainable parameters: {names[:12]}')
        return {'names': names, 'count': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                'target_modules': self.targets}

    def adapter_hash(self, name='default'):
        from peft import get_peft_model_state_dict
        h = hashlib.sha256()
        for key, value in sorted(get_peft_model_state_dict(self.model, adapter_name=name).items()):
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().view(self.torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def add_reference(self):
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        import copy
        state = {k: v.detach().clone() for k, v in get_peft_model_state_dict(self.model).items()}
        self.model.add_adapter('reference', copy.deepcopy(self.model.peft_config['default']))
        set_peft_model_state_dict(self.model, state, adapter_name='reference')
        self.model.set_adapter('default')
        for n, p in self.model.named_parameters():
            if '.reference.' in n:
                p.requires_grad_(False)
        self.has_reference = True
        self.reference_hash = self.adapter_hash('reference')
        if self.reference_hash != self.adapter_hash():
            raise RuntimeError('reference differs from actual warm start')
        self.assert_trainable()

    def prompt(self, row, grounded=False, questions=None):
        return self.encode([row], instruction(self.book, grounded, questions))

    def encode(self, rows, text_instruction):
        from PIL import Image
        images = []
        content = []
        for number, row in enumerate(rows):
            if len(rows) > 1:
                content.append({'type': 'text', 'text': f'Video {number}: '})
            for k, v in enumerate(row['views']):
                with Image.open(v['path']) as im:
                    images.append(letterbox(im, row['geometry']))
                content.extend([{'type': 'text', 'text': f'Frame {k}: '}, {'type': 'image'}])
        content.append({'type': 'text', 'text': text_instruction})
        messages = [{'role': 'user', 'content': content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        # Never place IDs, paths, labels, or audited annotations in user content.
        inputs = self.processor(text=[text], images=images, do_resize=False, return_tensors='pt')
        if inputs['input_ids'].shape[1] + 512 > 8192:
            raise ValueError('context overflow; image-token truncation forbidden')
        image_id = self.model.config.image_token_id
        if int((inputs['input_ids'] == image_id).sum()) <= 0 or len(inputs['image_grid_thw']) != len(images):
            raise RuntimeError('lost image tokens')
        return inputs.to('cuda:0'), text

    def completion(self, row, grounded=False):
        text, spans = target_parts(row, grounded)
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = encoded['input_ids'] + [self.tokenizer.eos_token_id]
        kinds = token_kinds(encoded['offset_mapping'], spans) + [1]
        if len(ids) > 512:
            raise ValueError(f'target longer than common completion cap: {len(ids)}')
        return ids, kinds, text

    def joined(self, prompt, ids):
        torch = self.torch
        inputs = dict(prompt)
        extra = torch.tensor([ids], dtype=torch.long, device='cuda:0')
        inputs['input_ids'] = torch.cat([prompt['input_ids'], extra], dim=1)
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids'])
        # Qwen3-VL uses modality IDs to derive multimodal RoPE positions.
        if 'mm_token_type_ids' in inputs:
            inputs['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'], torch.zeros_like(extra)], dim=1)
        return inputs, extra

    def logits(self, prompt, ids):
        inputs, extra = self.joined(prompt, ids)
        out = self.model(**inputs, use_cache=False, logits_to_keep=len(ids) + 1)
        return out.logits[:, :-1, :].float(), extra

    def sft_loss(self, prompt, ids, kinds):
        torch = self.torch
        logits, tokens = self.logits(prompt, ids)
        losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tokens, reduction='none')[0]
        kind = torch.tensor(kinds, device=losses.device)
        labels = losses[kind == 1].mean()
        box = losses[kind == 2].mean() if (kind == 2).any() else labels * 0
        return labels + .5 * box, {'label_loss': float(labels.detach()), 'box_loss': float(box.detach()),
                                  'label_tokens': int((kind == 1).sum()), 'box_tokens': int((kind == 2).sum()),
                                  'masked_tokens': int((kind == 0).sum())}

    def log_probs(self, prompt, ids, reference=False):
        if reference and not self.has_reference:
            raise RuntimeError('explicit warm-start reference missing')
        if reference:
            self.model.set_adapter('reference')
        try:
            logits, tokens = self.logits(prompt, ids)
            selected = logits.log_softmax(-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)[0]
            # Exact categorical entropy, reported separately from sampled-token surprisal.
            entropy = -(logits.softmax(-1) * logits.log_softmax(-1)).sum(-1)[0].mean()
            return selected, entropy
        finally:
            if reference:
                self.model.set_adapter('default')
                for n, p in self.model.named_parameters():
                    if '.reference.' in n:
                        p.requires_grad_(False)

    def generate(self, prompt, sample=False, max_new_tokens=512):
        torch = self.torch
        self.model.eval()
        with torch.no_grad():
            settings = {'do_sample': sample, 'max_new_tokens': max_new_tokens,
                        'use_cache': True, 'pad_token_id': self.tokenizer.pad_token_id,
                        'eos_token_id': self.tokenizer.eos_token_id}
            if sample:
                settings.update(temperature=.8, top_p=.95)
            full = self.model.generate(**prompt, **settings)
        ids = full[0, prompt['input_ids'].shape[1]:].tolist()
        truncated = not ids or ids[-1] != self.tokenizer.eos_token_id
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        return {'ids': ids, 'text': text, 'truncated': truncated, 'generated_tokens': len(ids)}

    def save_adapter(self, path):
        self.model.save_pretrained(path, selected_adapters=['default'], safe_serialization=True)
