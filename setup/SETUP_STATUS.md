# Setup status: READY (with two open items)

Updated 2026-10-22 after experiment01 run_001 and run_002 (both complete, validator 15/15).

| Component | Status | Detail |
|---|---|---|
| Connection | ready | `ssh sp2117@oz.gccis.rit.edu`. Oz is the GPU host itself and has no scheduler. See `connection.md`. |
| Old-experiment cleanup | **partial** | pip cache purged (21.17 GB, approved). The obsolete experiment folders (≈7.6 GiB of quota, plus 16.7 GB on local NVMe) are **not** removed: the agent's deletion was blocked by the permission classifier. Run `bash setup/scripts/cleanup_obsolete_experiments.sh --yes` to finish. See `cleanup_manifest.md`. |
| Source | ready | `source/VERA` @ `008ed401` = `15b8bcb8` + `patches/0001-…patch` (sha256 `de1d9888…`), branch `vera-research-local` |
| Environment | ready | `environment/env`: Python 3.10, torch 2.1.2+cu121, transformers 4.37.2, flash-attn 2.5.8. Rebuild with `scripts/setup_env.sh`; see `environment/REBUILD.md`. |
| Model | ready | `shared/models/OpenGVLab/InternVL2-8B/6fb9ad69…` (16.15 GB). All 23 files are verified against the hub's checksums (`MODEL_PROVENANCE.json`). |
| GPU check | passed | BF16 + FlashAttention 2 load on an RTX 6000 Ada. Peak 17.96 GiB allocated (19.5 GiB on the device). The A16 GPUs (15 GB) are too small, so `launch.sh` requires ≥ 40 GB. |
| Frame extraction | ready | VERA's own `Preprocessing/extract_frames.py`, run into the shared cache with an identity file |
| Step 1 runner | ready, validated | `scripts/run_experiment.py --stages vlm` |
| Steps 2–3 runner | ready, validated | `--stages refine`. Byte-identical to the unmodified `compute_auc.py` (validator C13; tests on synthetic videos). |
| ImageBind / LAVAD feature extraction | **not set up** | Not needed for UCF-Crime: the authors' per-video features can be range-fetched from their zip with `scripts/fetch_zip_member.py`. Any other video needs this first (`environment/requirements-imagebind-optional.txt`, untested). |
| Tests | passing | `scripts/test_vera_pipeline.py` 30/30 (log in `logs/`) |

## Reusable commands (on oz, from `~/vera_research`)

```bash
# acceptance tests (CPU)
setup/environment/env/bin/python setup/scripts/test_vera_pipeline.py
# a run (GPU; picks one idle >=40 GB GPU by UUID)
RUN=experiments/experimentNN/runs/run_NNN
bash setup/scripts/launch.sh --name expNN --log-dir $RUN/logs -- \
  setup/environment/env/bin/python setup/scripts/run_experiment.py --config experiments/experimentNN/config.yaml --run-dir $RUN --new-run
# resume or redo only some stages on an existing run (vlm skips windows already recorded)
setup/environment/env/bin/python setup/scripts/run_experiment.py --config <same config> --run-dir $RUN --stages refine
# validate, then summarize
setup/environment/env/bin/python setup/scripts/validate_run.py --run-dir $RUN [--require-report]
setup/environment/env/bin/python setup/scripts/summarize_run.py --run-dir $RUN
# fetch one more UCF-Crime clip, one zip member only (list first with --list)
python3 -I setup/scripts/fetch_zip_member.py --url '<official Dropbox zip URL with dl=1>' --member <path in zip> --out shared/videos/ucf_crime/<name>.mp4
```

## Known upstream behaviours (preserved, not fixed)

- **30 fps is hard-coded** for window length. RoadAccidents127 is exactly 30.0 fps; other clips must be checked.
- **Short videos break the refinement.** Fewer than 15 segments (< 225 frames): `np.convolve(mode='same')` returns 15
  values, so the smoothing is misaligned. Fewer than 7 segments: `top_n = 0` makes every segment a neighbour.
  `test_vera_pipeline.py` documents both.
- **Scores are rounded.** Each smoothed segment score is rounded to 0.1 before expansion to frames, and the final
  frame scores carry a position weighting centred on the middle of the video.

## Storage (home quota 150 GB)

- **Before:** ~140 GB used (department dump at 01:50; `du -sh ~` at 14:40).
- **Changes:** −21.17 GB pip cache; +~20 GB `~/vera_research` (model 16 GB, env 3.6 GB, frames 75 MB).
- **After (measured 16:13 EDT):** `du -s ~` = 148,675,424,768 bytes = 138.5 GiB, the same units as the earlier "140G".
  That leaves about 11.5 GiB of headroom.
- Removing the old experiment folders would free about 7.6 GiB more.
