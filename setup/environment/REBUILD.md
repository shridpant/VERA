# Rebuilding the vera-research environment

The environment lives at `~/vera_research/setup/environment/env`, a conda prefix with Python 3.10 from conda-forge and
pip wheels. It was built on oz on 2026-10-22 by `setup/scripts/setup_env.sh`, which is idempotent: rerunning it
installs only what is missing.

```bash
bash ~/vera_research/setup/scripts/setup_env.sh
```

Pinned stack:

| Component | Version | Source |
|---|---|---|
| Python | 3.10 (conda-forge) | `conda create -p setup/environment/env python=3.10 pip` |
| torch / torchvision / torchaudio | 2.1.2+cu121 / 0.16.2+cu121 / 2.1.2+cu121 | https://download.pytorch.org/whl/cu121 |
| flash-attn | 2.5.8 | official wheel `flash_attn-2.5.8+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` (installed `--no-deps`) |
| transformers | 4.37.2 | the version InternVL2 targets (`generation_config.json` says 4.37.2) |
| other pins | see `requirements.txt` | PyPI |

Exact installed versions are in `pip_freeze.txt`, and the conda base packages in `conda_explicit.txt`.

Host facts at build time:

- NVIDIA driver 615.71.10 (CUDA UMD 13.4). There is no system `nvcc`, so nothing is compiled.
- RTX 6000 Ada is compute capability 8.9. The cu121 build covers it, and so does the flash-attn 2.5.8 wheel (Ampere/Ada/Hopper).
- InternVL2-8B in BF16 does **not** fit on the 15 GB A16 GPUs. `launch.sh` therefore only picks GPUs with ≥ 40 GB.

Notes:

- `PIP_NO_CACHE_DIR=1`: nothing goes to `~/.cache/pip`, because the home quota is tight.
- Jobs set `HF_HOME=~/vera_research/shared/models/hf_home`, `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. The model is
  loaded from the checksum-verified local copy (`shared/models/.../MODEL_PROVENANCE.json`). Its remote code is copied
  into `hf_home/modules` on first load.
- **Not installed:** `pytorch_lightning`, `peft`, `decord` and `pandas`. The VERA inference path does not use them; the
  local patch removes their imports.
- **Not installed:** the LAVAD/ImageBind feature-extraction dependencies (`requirements-imagebind-optional.txt`).
  experiment01 uses the authors' released ImageBind features. The first attempt to pin them (av 11.0.0) had no cp310
  wheel, and its source build failed.
- Model weights: `python3 -I setup/scripts/fetch_hf_model.py --repo OpenGVLab/InternVL2-8B --revision
  6fb9ad6924f69424e57fab2ab061d707688f0296 --out-root ~/vera_research/shared/models` downloads and verifies every
  file against the hub's SHA-256 or git blob hash. Files that already verify are skipped.
