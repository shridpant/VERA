#!/bin/bash
# Two venvs on /local (the NFS home is nearly full):
#   vlm4: InternVL3.5 with its own remote code (needs transformers >= 4.52.1; 4.57.x is the last 4.x line)
#   vlm5: Qwen3-VL and Qwen3.5 natively (transformers 5)
# No flash-attn build: the new models run with PyTorch SDPA attention (recorded as a deviation from VERA's FlashAttention).
set -euo pipefail
E=/local/datasets/sp2117/envs
BASE=${BASE:-$HOME/miniforge3/bin/python3.12}
export PIP_CACHE_DIR=/local/datasets/sp2117/pip_cache
mkdir -p $E
HERE="$(cd "$(dirname "$0")" && pwd)"
for name in vlm4 vlm5; do
  [ -x $E/$name/bin/python ] || $BASE -m venv $E/$name
  $E/$name/bin/python -m pip install -q --upgrade pip
  $E/$name/bin/python -m pip install -q -r "$HERE/requirements-$name.txt"
  $E/$name/bin/python -m pip freeze > $E/${name}_pip_freeze.txt
  $E/$name/bin/python -c "import torch, transformers; print('$name', torch.__version__, torch.version.cuda, transformers.__version__, torch.cuda.is_available())"
done
