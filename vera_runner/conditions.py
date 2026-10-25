"""Prompts of the firearm study's three conditions, all sent through VERA's eight-frame packing.

A   VERA's own prompt, exactly backends.prompt(cfg).
G   a grounding-only prompt laid out like VERA's: instructions, then the same
    '** Input: **\\n[Frame1: <image>\\n...Frame8: <image>\\n]' block, then an output format.
AG  A with one paragraph inserted immediately before VERA's concluding instruction; every other
    byte of A, including its final Output format, is unchanged.

G and AG share the grounding paragraph (prompts/grounding_core.txt) verbatim. InternVL2's
batch_chat with num_patches_list=[8] puts all eight images at the first <image>; the other seven
'<image>' strings stay literal text. G and AG keep that packing, so slot references are text
labels over one image block (see studies/firearm/PROMPT_LOG.md for the interpretability check).
"""
import hashlib
from pathlib import Path
from .backends import prompt
from .config import path

CONDITIONS = ('A', 'G', 'AG')
MARKER = ("Based on the analysis above, please conclude your answer to 'Is there any anomaly in the video?' "
          "in 'Yes, there is an anomaly' or 'No, there is no anomaly'.")
FILES = {'core': 'grounding_core.txt', 'G': 'G_template.txt', 'AG': 'AG_insert_template.txt'}


def frame_prefix(n):
    """The runner's frame list (backends.prompt builds the same string)."""
    return ''.join(f'Frame{i+1}: <image>\n' for i in range(n))


def load_texts(prompt_dir):
    return {key: (path(prompt_dir) / name).read_text() for key, name in FILES.items()}


def build_prompt(cfg, condition, prompt_dir):
    if condition == 'A':
        return prompt(cfg)
    texts = load_texts(prompt_dir)
    core = texts['core'].strip('\n')
    if condition == 'G':
        text = texts['G'].replace('$Grounding', core).replace('$Data', frame_prefix(cfg['sampling']['frames_per_window']))
    elif condition == 'AG':
        base = prompt(cfg)
        if base.count(MARKER) != 1:
            raise ValueError('The A prompt must contain the concluding instruction exactly once')
        text = base.replace(MARKER, texts['AG'].replace('$Grounding', core) + MARKER, 1)
    else:
        raise ValueError(f'Unknown condition {condition}')
    if '$' in text:
        raise ValueError(f'Unreplaced placeholder in the {condition} prompt')
    return text


def prompt_texts(cfg, prompt_dir):
    return {c: build_prompt(cfg, c, prompt_dir) for c in CONDITIONS}


def sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
