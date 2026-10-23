#!/usr/bin/env bash
# Build (or finish building) the reusable vera-research environment on oz.
# Idempotent: an existing prefix is reused and pip only installs what is missing.
# Nothing is written to ~/.cache/pip (PIP_NO_CACHE_DIR=1): the home quota is tight.
#
# Usage (on oz): bash ~/vera_research/setup/scripts/setup_env.sh
set -Eeuo pipefail

ROOT="${VERA_PROJECT_ROOT:-$HOME/vera_research}"
ENV="$ROOT/setup/environment/env"
REQ="$ROOT/setup/environment/requirements.txt"
LOG="$ROOT/setup/logs/setup_env_$(date +%Y%m%dT%H%M%S).log"
FLASH_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "setup_env start $(date -Is) host=$(hostname -f)"
export PIP_NO_CACHE_DIR=1 PYTHONNOUSERSITE=1 PIP_DISABLE_PIP_VERSION_CHECK=1

if [[ ! -x "$ENV/bin/python" ]]; then
  "$HOME/miniforge3/bin/conda" create -y -p "$ENV" -c conda-forge --override-channels python=3.10 pip
fi
PY="$ENV/bin/python"
"$PY" --version

"$PY" -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
"$PY" -m pip install -r "$REQ"
"$PY" -m pip install --no-deps "$FLASH_WHL"

"$PY" -m pip freeze > "$ROOT/setup/environment/pip_freeze.txt"
"$HOME/miniforge3/bin/conda" list -p "$ENV" --explicit > "$ROOT/setup/environment/conda_explicit.txt"
"$PY" - <<'PY'
import torch, torchvision, transformers, flash_attn, timm, cv2, numpy, sklearn, scipy
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
print("torchvision", torchvision.__version__, "transformers", transformers.__version__,
      "flash_attn", flash_attn.__version__, "timm", timm.__version__, "cv2", cv2.__version__,
      "numpy", numpy.__version__, "sklearn", sklearn.__version__, "scipy", scipy.__version__)
PY
du -sh "$ENV"
echo "setup_env done $(date -Is)"
