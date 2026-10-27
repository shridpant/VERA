#!/usr/bin/env bash
# Fresh environment; never modifies vlm4/vlm5. Record the resolved lock only after smoke.
set -Eeuo pipefail
ENV=/local/datasets/sp2117/envs/firearm-posttrain
ROOT=/local/datasets/sp2117/firearm_visilab/posttraining1
mkdir -p "$ROOT/setup"
test ! -e "$ENV" || { echo "Refusing to overwrite existing environment $ENV" >&2; exit 2; }
/local/datasets/sp2117/envs/vlm5/bin/python -m venv "$ENV"
"$ENV/bin/python" -m pip install --upgrade pip
# Match the already functioning native Qwen loader; resolve PEFT/TRL in isolation.
"$ENV/bin/python" -m pip install 'torch==2.14.1' 'torchvision==0.29.1' 'transformers==5.21.0' peft trl accelerate datasets pillow numpy safetensors scipy
"$ENV/bin/python" -m pip check
"$ENV/bin/python" -m pip freeze --all > "$ROOT/setup/resolved-pre-smoke.txt"
"$ENV/bin/python" -m pip inspect > "$ROOT/setup/pip-inspect.json"
"$ENV/bin/python" - <<'PY'
import sys, torch, transformers, peft, trl
print(sys.version, torch.__version__, transformers.__version__, peft.__version__, trl.__version__)
print('CUDA', torch.version.cuda, torch.cuda.is_available())
PY
