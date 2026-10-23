# Configurable VERA inference

A configurable inference layer built from vanilla `origin/main` at
`15b8bcb8574a977c229c577f50bfe6f06d07106e` on branch `feature/configurable-runner`.
The original Python scripts and prompts are unchanged. This is a separate checkout;
previous `experiments/vanilla` runs in the other checkout are not modified.

## What vanilla expects

1. Obtain videos and install the chosen VLM's dependencies.
2. Run `Preprocessing/extract_frames.py --videos_dir ... --frames_dir ... --annotations_file ...`.
   It decodes with OpenCV and writes zero-based six-digit JPEG names. Its text manifest
   is not the JSON `Data/UCF_Eval.json` expected by inference.
3. Edit hard-coded paths/model/settings in `generate_initial_score.py`, then execute it.
   It runs InternVL2-8B, CUDA BF16, FlashAttention, one anchor every 16 frames, a
   nominal 30-FPS/10-second window, eight sampled images, and one 448-square tile per frame.
   Its root parser returns 0 when the text following the last `Output` contains a `0`,
   otherwise 1. It has import-time model loading and `.item()` on a Python-list result.
4. Obtain precomputed ImageBind segment features, set the feature, score and annotation
   paths in `compute_auc.py`, then run it for scene/temporal refinement and frame ROC-AUC.
   The README's initial-score output path and the refinement input path differ and
   must be connected manually. Feature extraction is not implemented by that script.

The new runner handles these paths explicitly, avoids import-time execution of those
scripts, captures raw model text and keeps both original-parser and strict-parser labels.
Refinement uses strict labels and refuses missing labels. It never fills failures with
another video or guessed labels.

## Install

Use Python 3.11 in an isolated environment:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-runner.txt
cp configs/datasets.example.json configs/datasets.local.json
```

Use the appropriate PyTorch build for your CUDA system if applicable. FlashAttention
is optional and separately installed for CUDA. `trust_remote_code=True` is needed by
InternVL2; use a trusted, pinned model revision. Model weights are not bundled.

On this Mac, an existing isolated environment can run the new checkout without a
second install:

```bash
../VERA/experiments/vanilla/.env/bin/python run_vera.py --config configs/local_smoke.json --check
```

`configs/local_smoke.json` and `datasets.local.json` have already been created for this
machine and are ignored by Git because they contain local paths. The former selects
up to three manifest-matched test videos and the first window per video, explicitly requests MPS and enables CPU
fallback for unsupported MPS operations. It has not completed real inference. The
older run 7 in the other checkout may still be running; do not start another large
model concurrently. Agent execution cannot detect MPS although regular Terminal can.

## One experiment = one JSON

The main CLI accepts:

```bash
.venv/bin/python run_vera.py --config configs/baseline.json --dataset ucf_crime --split test
```

`--dataset` and `--split` optionally override the JSON. All other settings live in
that single config. Defaults are in `vera_runner/config.py`, unknown settings fail,
and every run saves a fully resolved `command.json`. Pass a saved `command.json`
back to `--config` to create a new numbered run with those settings. Relative paths
are anchored to this checkout, not the shell's current directory. Paths/data/model
files and the captured source/environment must remain available for reproduction.

| Config | Behavior |
|---|---|
| `configs/baseline.json` | All selected windows, no intervention; preprocessing and Step 1 inference |
| `configs/blur_probe.json` | Baseline and Gaussian-blurred slot 0 for every window; label comparisons |
| `configs/grey_probe.json` | Baseline and grey-replaced slot 0 for every window; label comparisons |
| `configs/end_to_end.json` | Preprocess → infer → refine → frame ROC-AUC/AP; requires features and labels |
| `configs/local_smoke.json` | This Mac's untracked, partial, MPS baseline smoke configuration |

Copy any config before changing an experiment. Set `selection.video_ids`, `max_videos`
or `max_windows_per_video` for bounded runs. `split: train` only selects the training
partition as an inference input; this runner does not optimize weights or prompts.

## Dataset registration

Register a name once in `configs/datasets.local.json`, outside the experiment file:

```json
{
  "my_dataset": {
    "annotation_format": "vera_frames",
    "splits": {
      "train": {"videos": "/data/my_dataset/train"},
      "test": {
        "videos": "/data/my_dataset/test",
        "annotations": "/data/my_dataset/test.json",
        "features": "/data/my_dataset/segment_features"
      }
    }
  }
}
```

Then select `--dataset my_dataset`. Videos are found recursively; filename stems must
be unique in a split. No annotation file is needed for unlabeled inference. With a
manifest, only IDs present in both the root and manifest are selected; explicit missing
IDs are errors. Every selected input is captured in `output/dataset_manifest.json`.
UCF-Crime and XD-Violence examples are included. Their folder names are configurable,
not assumed; frame counts are validated against the manifest.

Annotation JSON is a list with `video`, optional `length`, and for evaluation
`temporal_label`: zero-based `[start,end)` FRAME intervals, flattened as
`[start1,end1,start2,end2]`. Normal videos use `[-1,-1]`. Endpoints are clipped at video
length as upstream does. Datasets using seconds, inclusive endpoints, or other schemas
must be explicitly converted before evaluation. A dataset name cannot infer these units.
Do not reuse UCF-specific questions for a new dataset without making that choice explicit:
`prompt.guiding_questions` can be a list of strings instead of `upstream_ucf`.

## Preprocessing and intervention semantics

OpenCV decode → default JPEG encode → PIL RGB load → PIL bicubic resize to 448×448
(upstream grid_size=1) → optional intervention → torchvision bicubic resize, tensor
conversion, ImageNet normalization → configured dtype/device. No crop or padding.
Nominal 30 FPS is the upstream window constant; actual FPS is recorded separately.
Window length, stride and frame count are configurable; changing them changes the
experiment and requires matching features for refinement.

```json
"intervention": {
  "mode": "blur",
  "slots": [0, 2],
  "blur_radius_fraction": 0.04,
  "grey_rgb": [127, 127, 127],
  "probe": true
}
```

- `baseline`: unmodified resized frames; slots must be empty.
- `blur`: GaussianBlur(radius = fraction × image width). Default 0.04 × 448 = 17.92.
- `grey`: replace each selected frame with an RGB (127,127,127) image by default.
- `slots`: zero-based positions WITHIN each sampled window, not original video frame IDs.
  Non-baseline modes require explicit slots; the other frames remain unchanged.
- `probe: true`: run baseline plus intervention from the SAME original window, record
  raw outputs, strict labels and flip/no-flip/unknown. These are full VERA generations,
  not the separate label-only numerical scorer used in experiment03B.
- `probe: false`: run only the selected condition. Interventions are never cumulative.

Blur/grey are experimental additions, not original VERA inference. The 0.04 blur
strength is configurable, not asserted to be a validated universal default. Saved PNGs
under `inputs/<anchor>/<condition>/` show the exact resized/intervened RGB frames before
normalization. Raw pixel checksums are recorded. Audio is not supplied; upstream UCF
question 5 still asks about sound unless you explicitly replace the questions.

## Backbones

`model.backend: internvl2` implements the InternVL2 batch-chat input convention and
supports CPU, MPS or CUDA; `auto` selects CUDA then MPS then CPU. BF16, FP16 and FP32
are explicit; changing precision/backend is not numerical equivalence to the paper.
FlashAttention defaults off for portability. MPS CPU fallback is explicit and recorded.
No memory limits are disabled. Large unquantized models may not fit this Mac.

To add a different model family, implement an adapter with `predict(images) ->
{raw_response, input_token_ids}` and a `device` field in `vera_runner/backends.py`,
register it in BACKENDS and config validation, then add tests. Different backbones
need different processors; a hub-name change alone does not implement Qwen/LLaVA.
The existing vanilla Qwen script is not a usable generic adapter (undefined gpu_index,
hard-coded CUDA and different questions). Only InternVL2 is implemented here.
The `mock` backend is exclusively for plumbing tests: outputs are visibly synthetic,
and `inference_completed` remains false.

## Refinement and evaluation

`configs/end_to_end.json` executes all released inference stages, provided features
are supplied. This runner does NOT silently download or recompute ImageBind features.
Each video's `.npy` file must have one row per ordered anchor from this exact sampling;
checksums and shapes are recorded. Matching row counts alone cannot prove extraction
semantics: you must ensure backbone, preprocessing, segment order and feature settings
match. Intervened runs using original features leave refinement similarities unchanged;
that must be considered when interpreting those results.

Refinement parameterizes upstream neighbor fraction, softmax similarity multiplier,
smoothing width/sigma, rounding and temporal position weighting. The default numerical
result is tested against the unmodified `compute_auc.py`. Too-short windows producing
zero neighbors or a longer convolution output are rejected, rather than silently
retaining upstream edge-case artifacts. Full window coverage is mandatory; probe runs
cannot be refined. Ground truth is used only in evaluation.

Metrics describe the selected set: frame ROC-AUC and generic frame average precision.
They are not a full-dataset result unless coverage matches the benchmark; generic AP
is NOT the XD-specific published AP pipeline. There is no invented whole-video
classification obtained by aggregating windows.

## Output, reliability and tests

```
runs/vanilla/
  run 1/
    command.json
    output/
      status.json
      diagnostic.log
      inference.log
      dataset_manifest.json
      source.json
      source_snapshot/
      environment.txt
      model_identity.json
      prompt.txt
      <video>/
        frames/
        inputs/
        video.json
        coverage.json
        raw_responses.jsonl
        segment_scores.json
        probe_comparison.json       # probes only
        frame_scores.npy            # refinement only
      summary.json
      metrics.json                  # evaluation only
```

Runs never overwrite earlier runs. `--check` creates `checks/run N` and checks dataset
resolution/imports/device/local model presence, not a promise that inference fits memory.
A process lock per output root prevents duplicate runs there, but different output roots
can run concurrently. A five-second heartbeat records stage, video and anchor; a fresh
heartbeat proves the process is alive, not that the GPU call is advancing. The runtime
wall limit defaults to one hour and exits with a timed_out status. Successful and caught
failed runs collect diagnostics; an OS kill may still leave a stale heartbeat. No resume
is implemented: a new invocation deliberately starts a new run.

Local models have file hashes recorded; an existing MODEL_PROVENANCE.json additionally
verifies all its files against hashes. Hub-model downloads use the configured revision;
for the strongest provenance, stage a verified local model first.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Tests cover actual upstream prompt equivalence, sampling arithmetic, exact interventions,
strict parsing, unknown-setting rejection, refinement equality to unmodified upstream,
and a synthetic video end-to-end probe + refine + evaluate workflow. Passing these tests
does not establish that InternVL2 inference works on MPS or reproduces CUDA predictions.

The local sample contains three originals, but only RoadAccidents011_x264 matches the released UCF test manifest. Local test-split configs therefore select that one video; the other sample clips are not silently reassigned to the test set. To infer on all three as an unlabeled custom collection, register their folder under a separate name without a manifest.

On this machine, the untracked registry also includes `salvaged_sample` (all three provided clips, no benchmark annotation). Use `--dataset salvaged_sample` for that explicitly unlabeled collection.
