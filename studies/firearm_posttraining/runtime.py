"""Immutable run provenance, JSONL ledgers, and exact training checkpoint state."""
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from .data import digest, write_json


def append(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps({'time_unix': time.time(), **row}, allow_nan=False, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())


def provenance(out, spec):
    import importlib.metadata as metadata
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True)
    if dirty:
        raise RuntimeError('committed clean checkout required for experiment execution')
    actual = {**spec, 'code_commit': commit, 'command': sys.argv, 'host': platform.node(),
              'python': sys.version, 'executable': sys.executable,
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'packages': {p.metadata['Name']: p.version for p in metadata.distributions()}}
    write_json(Path(out) / 'spec.json', actual, immutable=True)
    return actual


def save_checkpoint(native, optimizer, scheduler, path, state):
    import numpy as np
    torch = native.torch
    path = Path(path)
    if path.exists():
        raise ValueError(f'checkpoint exists: {path}')
    temp = path.with_name(path.name + '.partial')
    temp.mkdir(parents=True)
    native.save_adapter(temp)
    torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict() if scheduler else None,
                'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'state': state}, temp / 'training.pt')
    write_json(temp / 'state.json', {**state, 'adapter_hash': native.adapter_hash()})
    write_json(temp / 'hashes.json', {p.name: digest(p) for p in temp.iterdir() if p.is_file()})
    temp.rename(path)


def restore_checkpoint(native, optimizer, scheduler, path):
    import numpy as np
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    torch = native.torch
    path = Path(path)
    for name, expected in json.loads((path / 'hashes.json').read_text()).items():
        if digest(path / name) != expected:
            raise RuntimeError(f'corrupted checkpoint: {name}')
    set_peft_model_state_dict(native.model, load_file(path / 'adapter_model.safetensors'))
    # Only our locally created, hashed checkpoints may enter this path.
    saved = torch.load(path / 'training.pt', map_location='cpu', weights_only=False)
    optimizer.load_state_dict(saved['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(saved['scheduler'])
    random.setstate(saved['python_rng'])
    np.random.set_state(saved['numpy_rng'])
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    if native.adapter_hash() != json.loads((path / 'state.json').read_text())['adapter_hash']:
        raise RuntimeError('adapter reload hash mismatch')
    return saved['state']


def resource_metrics():
    import torch
    return {'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'device': torch.cuda.get_device_name(0)}
