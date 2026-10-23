"""Backends return raw text. Dataset and interventions remain backend-independent."""
import ast
import hashlib
import json
import re
from .config import ROOT, path


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
        if settings['flash_attention'] and self.device != 'cuda':
            raise ValueError('FlashAttention requires CUDA')
        self.dtype = getattr(torch, settings['dtype'])
        torch.manual_seed(cfg['seed'])
        source = settings['path']
        self.model_identity = {'repo_or_path': source, 'revision': settings['revision'], 'files': {}}
        if path(source).is_dir():
            source = str(path(source))
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
        elif settings['local_files_only']:
            raise ValueError('local_files_only requires an existing model path')
        self.model = AutoModel.from_pretrained(source, revision=settings['revision'],
            torch_dtype=self.dtype, low_cpu_mem_usage=True, trust_remote_code=True,
            use_flash_attn=settings['flash_attention'], local_files_only=settings['local_files_only']).eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(source, revision=settings['revision'],
            trust_remote_code=True, local_files_only=settings['local_files_only'])
        self.question = prompt(cfg)
        # Retain token IDs passed to the language model, not only the high-level prompt.
        original = self.model.generate
        def captured(*args, **kwargs):
            ids = kwargs.get('input_ids')
            self.last_ids = ids.detach().cpu().tolist() if ids is not None else None
            return original(*args, **kwargs)
        self.model.generate = captured

    def predict(self, images):
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        transform = T.Compose([T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(), T.Normalize((.485,.456,.406),(.229,.224,.225))])
        pixels = self.torch.stack([transform(im) for im in images]).to(device=self.device,dtype=self.dtype)
        with self.torch.inference_mode():
            answer = self.model.batch_chat(self.tokenizer, pixels, num_patches_list=[len(images)],
                questions=[self.question], generation_config=dict(self.cfg['generation']))[0]
        return {'raw_response': answer, 'input_token_ids': self.last_ids}


class Mock:
    """Plumbing tests only. Never a substitute for model inference or a research result."""
    def __init__(self, cfg):
        self.question = prompt(cfg)
        self.device = 'mock'
    def predict(self, images):
        return {'raw_response': 'MOCK OUTPUT — NOT A MODEL PREDICTION\nOutput:\n0', 'input_token_ids': None}


BACKENDS = {'internvl2': InternVL2, 'mock': Mock}
