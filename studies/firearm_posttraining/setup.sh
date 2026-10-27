#!/usr/bin/env bash
# Recreate the tested Linux/Python 3.12 environment; never modifies vlm4/vlm5.
set -Eeuo pipefail
ENV=/local/datasets/sp2117/envs/firearm-posttrain
ROOT=/local/datasets/sp2117/firearm_visilab/posttraining1
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
test -f "$HERE/requirements.lock"
mkdir -p "$ROOT/setup"
test ! -e "$ENV" || { echo "Refusing to overwrite existing environment $ENV" >&2; exit 2; }
/local/datasets/sp2117/envs/vlm5/bin/python -m venv "$ENV"
if [[ "${1:-}" == --offline ]]; then
  "$ENV/bin/python" -m pip install --no-index --find-links "$ROOT/setup/wheels" --require-hashes -r "$HERE/requirements.lock"
else
  "$ENV/bin/python" -m pip install --require-hashes -r "$HERE/requirements.lock"
fi
"$ENV/bin/python" -m pip check
"$ENV/bin/python" -m pip freeze --all > "$ROOT/setup/resolved-pre-smoke.txt"
"$ENV/bin/python" -m pip inspect > "$ROOT/setup/pip-inspect.json"
"$ENV/bin/python" - <<'PY'
import sys, torch, transformers, peft, trl
print(sys.version, torch.__version__, transformers.__version__, peft.__version__, trl.__version__)
print('CUDA', torch.version.cuda, torch.cuda.is_available())
PY
