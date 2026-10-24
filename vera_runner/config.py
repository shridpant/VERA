import copy
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {
    'schema_version': 1, 'dataset': 'ucf_crime', 'split': 'test',
    'registry': 'configs/datasets.local.json', 'output_root': 'runs/vanilla',
    'stages': ['preprocess', 'infer'], 'seed': 0,
    'selection': {'video_ids': [], 'max_videos': None, 'max_windows_per_video': None},
    'preprocessing': {'image_size': 448},
    'sampling': {'stride': 16, 'window_seconds': 10.0, 'fps': 30.0, 'frames_per_window': 8},
    'model': {'backend': 'internvl2', 'path': 'OpenGVLab/InternVL2-8B',
              'revision': '6fb9ad6924f69424e57fab2ab061d707688f0296',
              'device': 'auto', 'dtype': 'bfloat16', 'flash_attention': False,
              'mps_cpu_fallback': False, 'local_files_only': True},
    'generation': {'num_beams': 1, 'max_new_tokens': 1024, 'do_sample': False},
    'prompt': {'instruction_file': 'VERA_learner_instruct.txt', 'guiding_questions': 'upstream_ucf'},
    'intervention': {'mode': 'baseline', 'slots': [], 'blur_radius_fraction': 0.04,
                     'grey_rgb': [127, 127, 127], 'probe': False},
    'refinement': {'neighbor_fraction': 0.15, 'similarity_temperature': 10.0,
                   'smoothing_size': 15, 'smoothing_sigma': 10.0,
                   'round_decimals': 1, 'position_weighting': True},
    'runtime': {'timeout_seconds': 3600},
}


def merge(base, incoming, prefix=''):
    if not isinstance(incoming, dict):
        raise ValueError(f'{prefix or "config"} must be an object')
    result = copy.deepcopy(base)
    for key, value in incoming.items():
        if key not in base:
            raise ValueError(f'Unknown setting: {prefix}{key}')
        if isinstance(base[key], dict):
            result[key] = merge(base[key], value, prefix + key + '.')
        else:
            result[key] = value
    return result


def path(value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def prepare_environment(cfg):
    """Set process variables that torch and Hugging Face read at import time."""
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' if cfg['model']['mps_cpu_fallback'] else '0'
    os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))


def positive(value, name, integer=False):
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    if integer and not isinstance(value, int):
        raise ValueError(f'{name} must be an integer')


def load(filename, overrides=None):
    cfg = merge(DEFAULTS, json.loads(Path(filename).read_text()))
    for key, value in (overrides or {}).items():
        if value is not None:
            cfg[key] = value
    if cfg['schema_version'] != 1:
        raise ValueError('Unsupported schema_version')
    if cfg['split'] not in ('train', 'test', 'val'):
        raise ValueError('split must be train, val or test; train selects inputs only, never trains weights')
    if not isinstance(cfg['stages'], list) or cfg['stages'] not in (
            ['preprocess'], ['preprocess', 'infer'], ['preprocess', 'infer', 'refine'],
            ['preprocess', 'infer', 'refine', 'evaluate']):
        raise ValueError('stages must be an ordered prefix of preprocess,infer,refine,evaluate')
    for key in ('stride', 'frames_per_window'):
        positive(cfg['sampling'][key], 'sampling.' + key, True)
    for key in ('window_seconds', 'fps'):
        positive(cfg['sampling'][key], 'sampling.' + key)
    positive(cfg['preprocessing']['image_size'], 'image_size', True)
    positive(cfg['runtime']['timeout_seconds'], 'timeout_seconds')
    positive(cfg['generation']['max_new_tokens'], 'max_new_tokens', True)
    positive(cfg['generation']['num_beams'], 'num_beams', True)
    for name in ('do_sample',):
        if type(cfg['generation'][name]) is not bool:
            raise ValueError(name + ' must be boolean')
    sel = cfg['selection']
    if not isinstance(sel['video_ids'], list) or not all(isinstance(x, str) for x in sel['video_ids']):
        raise ValueError('video_ids must be a list of strings')
    for key in ('max_videos', 'max_windows_per_video'):
        if sel[key] is not None:
            positive(sel[key], key, True)
    m = cfg['model']
    if m['backend'] not in ('internvl2', 'mock'):
        raise ValueError('Unknown backend; implement/register an adapter in backends.py first')
    if m['backend'] == 'internvl2' and cfg['preprocessing']['image_size'] != 448:
        raise ValueError('InternVL2 adapter is validated for 448px tiles only')
    # dtype 'auto' resolves after the device: CUDA BF16, MPS FP16, CPU FP32.
    if m['device'] not in ('auto', 'cpu', 'mps', 'cuda') or m['dtype'] not in ('auto', 'bfloat16', 'float16', 'float32'):
        raise ValueError('Unsupported device or dtype')
    if m['revision'] is not None and not isinstance(m['revision'], str):
        raise ValueError('model.revision must be a string or null (null uses the Hub default branch)')
    # flash_attention 'auto' is on only for CUDA with flash-attn installed, as upstream VERA ran.
    if type(m['flash_attention']) is not bool and m['flash_attention'] != 'auto':
        raise ValueError('model.flash_attention must be boolean or "auto"')
    for key in ('mps_cpu_fallback', 'local_files_only'):
        if type(m[key]) is not bool:
            raise ValueError('model.' + key + ' must be boolean')
    iv = cfg['intervention']
    if iv['mode'] not in ('baseline', 'blur', 'grey') or type(iv['probe']) is not bool:
        raise ValueError('intervention mode must be baseline, blur or grey; probe must be boolean')
    slots = iv['slots']
    if not isinstance(slots, list) or any(type(i) is not int or i < 0 or i >= cfg['sampling']['frames_per_window'] for i in slots) or len(set(slots)) != len(slots):
        raise ValueError('slots must be unique zero-based frame positions within a window')
    if iv['mode'] == 'baseline' and (slots or iv['probe']):
        raise ValueError('baseline must have empty slots and probe=false')
    if iv['mode'] != 'baseline' and not slots:
        raise ValueError('Choose slots explicitly; no implicit perturbation of every frame')
    positive(iv['blur_radius_fraction'], 'blur_radius_fraction')
    if not isinstance(iv['grey_rgb'], list) or len(iv['grey_rgb']) != 3 or any(type(x) is not int or x < 0 or x > 255 for x in iv['grey_rgb']):
        raise ValueError('grey_rgb must contain three integers in [0,255]')
    ref = cfg['refinement']
    if not 0 < ref['neighbor_fraction'] <= 1:
        raise ValueError('neighbor_fraction must be in (0,1]')
    positive(ref['similarity_temperature'], 'similarity_temperature')
    positive(ref['smoothing_size'], 'smoothing_size', True)
    positive(ref['smoothing_sigma'], 'smoothing_sigma')
    if type(ref['round_decimals']) is not int or ref['round_decimals'] < 0:
        raise ValueError('round_decimals must be nonnegative integer')
    if type(ref['position_weighting']) is not bool:
        raise ValueError('position_weighting must be boolean')
    if 'refine' in cfg['stages'] and (sel['max_windows_per_video'] is not None or iv['probe']):
        raise ValueError('Refinement requires all windows and one condition per window; probe or partial windows cannot be refined')
    return cfg
