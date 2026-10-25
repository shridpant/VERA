"""Backends return raw text. Dataset and interventions remain backend-independent."""
import ast
import hashlib
import json
import re
import time
from .config import ROOT, path
from . import grounding


def upstream_questions():
    tree = ast.parse((ROOT / 'generate_initial_score.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'new_qs' for t in n.targets))
    return eval(compile(ast.Expression(node.value), '<upstream questions>', 'eval'), {'__builtins__': {}})


def prompt(cfg):
    text = path(cfg['prompt']['instruction_file']).read_text()
    marker = "Based on the analysis above, please conclude your answer to 'Is there any anomaly in the video?' in 'Yes, there is an anomaly' or 'No, there is no anomaly'."
    qs = cfg['prompt']['guiding_questions']
    if qs == 'upstream_ucf':
        qs = upstream_questions()
    elif isinstance(qs, list) and all(isinstance(q, str) for q in qs):
        qs = '\n'.join(qs) + '\n'
    else:
        raise ValueError('guiding_questions must be upstream_ucf or an explicit list')
    if marker not in text or '$Data' not in text:
        raise ValueError('Instruction file must contain upstream insertion marker and $Data')
    prefix = ''.join(f'Frame{i+1}: <image>\n' for i in range(cfg['sampling']['frames_per_window']))
    return text.replace('$Data', prefix).replace(marker, qs + marker, 1)


def strict_label(text):
    fields = list(re.finditer(r'output\s*:', text, flags=re.I))
    if len(fields) != 1:
        return None
    tail = text[fields[0].end():].replace('```', ' ').strip(' \t\r\n`[]()\"\'*._')
    return int(tail) if tail in ('0', '1') else None


def intervene(image, spec):
    from PIL import Image, ImageFilter
    if spec['mode'] == 'baseline':
        return image
    if spec['mode'] == 'blur':
        return image.filter(ImageFilter.GaussianBlur(spec['blur_radius_fraction'] * image.width))
    return Image.new('RGB', image.size, tuple(spec['grey_rgb']))


class TextProgress:
    """Generation streamer: reports the decoded text so far. It only observes; tokens are unchanged."""
    def __init__(self, tokenizer, report, every=0.25):
        self.tokenizer, self.report, self.every = tokenizer, report, every
        self.ids, self.prompt_seen, self.reported = [], False, 0.0

    def put(self, value):
        if not self.prompt_seen:  # generate() first passes the prompt ids (empty with inputs_embeds).
            self.prompt_seen = True
            return
        self.ids.extend(value.reshape(-1).tolist())
        if time.monotonic() - self.reported >= self.every:
            self.reported = time.monotonic()
            self.report(self.tokenizer.decode(self.ids, skip_special_tokens=True), len(self.ids), False)

    def end(self):
        self.report(self.tokenizer.decode(self.ids, skip_special_tokens=True), len(self.ids), True)


def install_progress_hooks(model, report):
    """Observe vision and language forwards without changing tensors or generation."""
    counts={'vision':0,'language':0}
    handles=[]
    for name,module in [('vision',model.vision_model),('language',model.language_model)]:
        def before(mod,args,name=name):
            counts[name]+=1
            report(name+'_forward_begin',{'forward_index':counts[name]})
        def after(mod,args,result,name=name):
            report(name+'_forward_end',{'forward_index':counts[name]})
        handles.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
    return handles


def generated(tokenizer, output_ids, seconds, peak_gpu_bytes):
    """Telemetry of one generation. With inputs_embeds, generate() returns a leading <s> before the
    new tokens; EOS is InternVL2's <|im_end|>, which batch_chat sets as eos_token_id."""
    ids = list(output_ids or [])
    new = ids[1:] if ids and ids[0] == tokenizer.bos_token_id else ids
    return {'output_ids': ids, 'tagged_text': tokenizer.decode(ids, skip_special_tokens=False),
            'eos': tokenizer.convert_tokens_to_ids('<|im_end|>') in new, 'generated_tokens': len(new),
            'seconds': seconds, 'peak_gpu_bytes': peak_gpu_bytes}


DTYPE_FOR_DEVICE = {'cuda': 'bfloat16', 'mps': 'float16', 'cpu': 'float32'}


def resolve_checkpoint(settings):
    """Return a local checkpoint directory, or a Hub ID only when downloads are allowed."""
    source = settings['path']
    if path(source).is_dir():
        return str(path(source))
    if not settings['local_files_only']:
        return source
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError
    try:
        # A cached Hub snapshot is already local; local_files_only only forbids network access.
        return snapshot_download(source, revision=settings['revision'], local_files_only=True)
    except LocalEntryNotFoundError as exc:
        raise ValueError('local_files_only requires a model directory or a cached Hub snapshot of this revision') from exc


class InternVL2:
    def __init__(self, cfg):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.cfg, self.torch = cfg, torch
        settings = cfg['model']
        self.device = settings['device']
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        if self.device == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError('MPS requested but unavailable to this process')
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        flash = settings['flash_attention']
        if flash == 'auto':
            import importlib.util
            flash = self.device == 'cuda' and importlib.util.find_spec('flash_attn') is not None
        if flash and self.device != 'cuda':
            raise ValueError('FlashAttention requires CUDA')
        dtype = DTYPE_FOR_DEVICE[self.device] if settings['dtype'] == 'auto' else settings['dtype']
        self.dtype = getattr(torch, dtype)
        torch.manual_seed(cfg['seed'])
        source = resolve_checkpoint(settings)
        self.model_identity = {'repo_or_path': settings['path'], 'resolved_source': source, 'revision': settings['revision'],
                               'device': self.device, 'dtype': dtype, 'flash_attention': flash, 'files': {}}
        if path(source).is_dir():
            from .data import sha256
            directory = path(source)
            prov_path = directory / 'MODEL_PROVENANCE.json'
            provenance = json.loads(prov_path.read_text()) if prov_path.exists() else None
            if provenance and provenance.get('revision') != settings['revision']:
                raise ValueError('Local model provenance revision differs from config')
            for file in sorted(directory.iterdir()):
                if file.is_file() and file.suffix in ('.py','.json','.model','.safetensors','.bin'):
                    self.model_identity['files'][file.name] = sha256(file)
            if provenance:
                for name, expected in provenance['files'].items():
                    file=directory/name
                    if not file.is_file() or file.stat().st_size != expected['bytes']:
                        raise ValueError('Missing or truncated model file: '+name)
                    if 'sha256' in expected:
                        got=self.model_identity['files'].get(name) or sha256(file)
                        if got != expected['sha256']:raise ValueError('Model checksum mismatch: '+name)
                    else:
                        got=hashlib.sha1(b'blob %d\0' % expected['bytes']+file.read_bytes()).hexdigest()
                        if got != expected['git_blob_sha1']:raise ValueError('Model metadata checksum mismatch: '+name)
        self.model = AutoModel.from_pretrained(source, revision=settings['revision'],
            torch_dtype=self.dtype, low_cpu_mem_usage=True, trust_remote_code=True,
            use_flash_attn=flash, local_files_only=settings['local_files_only']).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=settings['revision'],
            trust_remote_code=True, local_files_only=settings['local_files_only'])
        self.model_identity['resolved_commit'] = getattr(self.model.config, '_commit_hash', None)
        self.question = prompt(cfg)
        self.progress = lambda stage,details: None
        def report(stage,details):
            import resource
            details=dict(details)
            details['maxrss_raw']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            details['maxrss_unit']='bytes' if __import__('sys').platform=='darwin' else 'KiB'
            if self.device=='mps':
                for key,func in [('mps_allocated_bytes',torch.mps.current_allocated_memory),('mps_driver_bytes',torch.mps.driver_allocated_memory)]:
                    try:details[key]=func()
                    except Exception:details[key]=None
            self.progress(stage,details)
        self._progress_handles=install_progress_hooks(self.model,report)
        # Retain token IDs passed to and generated by the language model, not only the decoded text.
        original = self.model.generate
        def captured(*args, **kwargs):
            ids = kwargs.get('input_ids')
            self.last_ids = ids.detach().cpu().tolist() if ids is not None else None
            output = original(*args, **kwargs)
            self.last_output_ids = output[0].detach().cpu().tolist()
            return output
        self.model.generate = captured

    def configure(self, cfg):
        """Reuse the loaded weights with another request's prompt and generation settings."""
        if cfg['model'] != self.cfg['model']:
            raise ValueError('Model settings changed; load the new model in a new process')
        self.cfg, self.question = cfg, prompt(cfg)

    def pixels(self, images):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        transform = T.Compose([T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
        return self.torch.stack([transform(im) for im in images]).to(device=self.device,dtype=self.dtype)

    def predict(self, images, on_text=None, question=None):
        """VERA's call: all images and the prompt in one batch_chat. on_text(text, tokens, done) streams it.

        question=None sends the configured VERA prompt. The firearm study passes its G/AG prompt here,
        so every condition gets the same images, packing and generation settings. last_call keeps
        the generated ids, their decode with special tokens, EOS and peak GPU memory."""
        generation = dict(self.cfg['generation'])
        if on_text is not None:
            generation['streamer'] = TextProgress(self.tokenizer, on_text)
        cuda = self.device == 'cuda'
        if cuda:
            self.torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        with self.torch.inference_mode():
            answer = self.model.batch_chat(self.tokenizer, self.pixels(images), num_patches_list=[len(images)],
                questions=[self.question if question is None else question], generation_config=generation)[0]
        self.last_call = generated(self.tokenizer, self.last_output_ids, time.monotonic() - begin,
                                   self.torch.cuda.max_memory_allocated() if cuda else None)
        return {'raw_response': answer, 'input_token_ids': self.last_ids}

    def rendered_input(self):
        """The last language-model input as text, special tokens kept and image-token runs counted."""
        if not self.last_ids:
            return ''
        text = self.tokenizer.decode(self.last_ids[0], skip_special_tokens=False)
        # The decode separates special tokens with spaces, so runs are matched with optional whitespace.
        return re.sub(r'<IMG_CONTEXT>(?:\s*<IMG_CONTEXT>)*', lambda m: f'<IMG_CONTEXT>×{m.group(0).count("<IMG_CONTEXT>")}', text)

    def name_anomaly(self, images, response):
        """Follow-up in the same conversation: the model names the anomaly it described."""
        begin = time.monotonic()
        generation = {'num_beams': 1, 'max_new_tokens': 32, 'do_sample': False}
        with self.torch.inference_mode():
            text = self.model.chat(self.tokenizer, self.pixels(images), grounding.NAME_PROMPT, generation,
                                   history=[(self.question, response)], num_patches_list=[len(images)])
        return {'prompt': grounding.NAME_PROMPT, 'raw_text': text, 'phrase': grounding.clean_phrase(text),
                'seconds': time.monotonic() - begin}

    def ground(self, image, phrase, max_new_tokens=100):
        """One image, the model-card grounding prompt; parsed from the decode that keeps <ref>/<box>."""
        begin = time.monotonic()
        question = grounding.GROUND_PROMPT.format(phrase)
        generation = {'num_beams': 1, 'max_new_tokens': max_new_tokens, 'do_sample': False}
        with self.torch.inference_mode():
            text = self.model.chat(self.tokenizer, self.pixels([image]), question, generation)
        tagged = self.tokenizer.decode(self.last_output_ids, skip_special_tokens=False)
        return {'prompt': question, 'raw_text': text, 'tagged_text': tagged, **grounding.parse_boxes(tagged),
                'seconds': time.monotonic() - begin}


class Mock:
    """Plumbing tests only. Never a substitute for model inference or a research result.

    Its label is 1 when the first image is bright (mean RGB >= 128), so tests can tell alert
    windows from normal ones; its one box is fixed. All of it is visibly synthetic.
    """
    def __init__(self, cfg):
        self.question = prompt(cfg)
        self.device = 'mock'
    def configure(self, cfg):
        self.question = prompt(cfg)
    def predict(self, images, on_text=None, question=None):
        question = self.question if question is None else question
        bright = [int(sum(im.convert('L').getdata()) / (im.width * im.height) >= 128) for im in images]
        text = 'MOCK OUTPUT — NOT A MODEL PREDICTION\n'
        if 'Firearm localization' in question:  # firearm study G/AG: a fixed box on every bright slot
            text += 'Firearm boxes:\n' + ''.join(f'Frame{i+1}: ' + ('[[100, 200, 600, 700]]' if b else '[]') + '\n'
                                                 for i, b in enumerate(bright))
        if 'Is there any anomaly' in question:
            text += f'Output:\n{bright[0]}'
        if on_text is not None:
            on_text(text, 1, True)
        self.last_call = {'output_ids': [], 'tagged_text': text, 'eos': True, 'generated_tokens': len(text.split()),
                          'seconds': 0.0, 'peak_gpu_bytes': None}
        return {'raw_response': text, 'input_token_ids': None}
    def rendered_input(self):
        return 'MOCK INPUT — NO TOKENIZER\n' + self.question
    def name_anomaly(self, images, response):
        return {'prompt': grounding.NAME_PROMPT, 'raw_text': 'mock anomaly', 'phrase': 'mock anomaly', 'seconds': 0.0}
    def ground(self, image, phrase, max_new_tokens=100):
        tagged = f'<ref> {phrase} </ref> <box> [[100, 200, 600, 700]] </box>'
        return {'prompt': grounding.GROUND_PROMPT.format(phrase), 'raw_text': tagged, 'tagged_text': tagged,
                **grounding.parse_boxes(tagged), 'seconds': 0.0}


BACKENDS = {'internvl2': InternVL2, 'mock': Mock}
