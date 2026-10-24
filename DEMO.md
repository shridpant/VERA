# VERA Lab

VERA Lab is a separate, localhost-only page over the existing configurable
runner. It runs **one window** or a **complete video through a selected stage**,
then shows the eight images the model saw, the text it generated, and every file
each stage saved. It does not train or modify InternVL2. It uses the runner's
sampler, prompt builder, parser, and refinement code. The frame-game and score
pages are labelled teaching examples, not model data.

Original VERA's `generate_initial_score.py` discards the VLM response after
parsing a 0/1 label. It saves per-window `start`, `end`, and `score` in one JSON
file per video. The configurable runner keeps the raw text and its inputs, then
saves every numerical refinement stage. VERA does not cite supporting sampled
frame IDs for each generated sentence. This lab shows the eight input images; it
does not add or validate citations.

## Start the lab

The page checks Python dependencies before launching a job and names any that
are missing. Run it with an interpreter that has `requirements-runner.txt`.

### On a Mac (MPS)

```bash
uv venv --seed --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-runner.txt
.venv/bin/python tools/demo_server.py --port 8765 \
  --video-dir ~/Documents/research/vera_research/shared/videos/ucf_crime \
  --model-path ~/Documents/research/vera_research/shared/models/OpenGVLab/InternVL2-8B/6fb9ad6924f69424e57fab2ab061d707688f0296
```

Open `http://127.0.0.1:8765/`. The checkpoint folder is the verified Oz copy
(16 GB, copied with `rsync`); its `MODEL_PROVENANCE.json` is checked on load.
Leave device and precision on **auto**: on Apple silicon that is MPS with FP16
and CPU fallback for operators MPS lacks. On an M3 Max (128 GB) the first window took 35 s (15 s to load the model,
19 s to generate) and the next window 11 s.

### On Oz (RTX 6000 Ada)

Use an RTX 6000 Ada, not one of the 15 GB A16 cards. Oz CUDA ordinals do not
match the indices printed by `nvidia-smi`; select the GPU by UUID:

```bash
cd ~/vera_research/setup/source/VERA-lab
CUDA_VISIBLE_DEVICES=GPU-6a490dc2-d384-9010-bc0b-b0b76076bae3 ~/vera_research/setup/environment/env/bin/python tools/demo_server.py \
  --port 8876 \
  --video-dir ~/vera_research/shared/videos/ucf_crime \
  --model-path ~/vera_research/shared/models/OpenGVLab/InternVL2-8B/6fb9ad6924f69424e57fab2ab061d707688f0296 \
  --feature-dir ~/vera_research/shared/features/vera_official_ucf_imagebind \
  --annotation-file Data/UCF_Eval.json
```

From your Mac, tunnel to the Oz-only service and open `http://127.0.0.1:8765/`:

```bash
ssh -N -L 8765:127.0.0.1:8876 sp2117@oz.gccis.rit.edu
```

Stop a local lab on port 8765 first. The Oz server binds only to loopback, so it
is not exposed to other computers.

## The run page

The page opens on the pipeline, drawn in two lanes: what InternVL2 sees for one
window, and what happens across the whole video once every window has a label.
Stages that the selected run includes are highlighted. Each stage that has a
setting carries a ⚙ tag; click the stage to jump to that setting.

| Knob | Stage it changes | Default |
|---|---|---|
| Video: drop, ⚄ die, or path | 1 · input | none; the die picks a random library video and start time |
| Run through stage | which stages run | one sampled window |
| Window centre | 2–3 · which anchor's eight images | 5 s (nearest anchor) |
| Guiding questions | 4 · prompt | VERA's released UCF-Crime questions |
| Checkpoint, device, precision, download | 5 · InternVL2 | verified local copy, auto, auto, no download |
| (fixed) FlashAttention, generation | 5 · InternVL2 | on for CUDA when installed; greedy, 1024 new tokens |
| Feature file | 7–8 · scene and temporal refinement | library match, if any |
| Annotations | 9 · metrics | the server's `--annotation-file` |

Defaults are VERA's original settings. **↺ Reset to VERA defaults** restores every
knob and clears the video. With these settings a full RoadAccidents127 run on Oz
reproduced experiment01's 162 responses word for word.

The die only fills in a video; it never starts inference. For refinement targets
it picks only videos that have a matching feature file. A dropped video is
previewed in the browser and uploaded in bounded chunks. If you change the video
while a die roll or upload is still in flight, the late reply is discarded.

Checkpoints: local copies and `OpenGVLab/InternVL2-8B` use the runner's pinned
revision. Another InternVL2 Hub ID (for example `OpenGVLab/InternVL2-2B`) uses its
default branch, and `model_identity.json` records the commit actually loaded.
InternVL2 needs `trust_remote_code`, so such an ID runs that repository's current
model code; prefer a verified local copy. Window and full-video runs share the
runner's Hugging Face cache, `.cache/huggingface` in this checkout unless `HF_HOME`
is set. With downloads off, a Hub ID still works if that revision is already there.

## Warm model between window runs

Loading eight billion weights dominates a one-window run. Window runs therefore go
to one persistent process (`tools/window_probe.py --serve`) that keeps the model
loaded:

- The next window with the same checkpoint, device, precision and download
  setting reuses it. Questions and the window may change freely.
- Changing any of those model settings starts a fresh process.
- A full-video run stops the warm process first, because the runner loads its
  own copy of the model.
- After 15 minutes without a request the process exits and frees GPU memory.
  The server replaces it a minute before that, so a late click never races the exit.

The page shows whether a model is loaded; each result reports load and
inference seconds separately. Two consecutive windows, measured on 2026-10-24: Oz RTX 6000 Ada 31.0 s then 4.6 s;
Mac M3 Max 35.3 s then 11.1 s.

## Where it runs and what it uses

Under the run status, a usage panel refreshes every few seconds (`GET /api/usage`).
It names the machine the lab server runs on. Through the SSH tunnel that is `oz`,
even though the page is open on your Mac. It also names the process holding the
model (the full-video runner or the window model) and shows:

- on CUDA, how busy each visible GPU is, its memory, and how much the model
  process holds (`nvidia-smi`, limited to `CUDA_VISIBLE_DEVICES`);
- on a Mac, Apple GPU utilization and GPU-allocated unified memory (`ioreg`), and
  the model process's full memory footprint, GPU buffers included (`top`);
- the model process's CPU use.

Nothing here needs elevated rights or changes the run.

## Choose the final stage

| UI choice | Existing code | Files you can inspect |
|---|---|---|
| One sampled window | `tools/window_probe.py` with `vera_runner` | Eight PNGs, exact source indices/timestamps, prompt, raw text, strict 0/1 label |
| Decode video | `run_vera.py` `preprocess` | Video metadata, all decoded JPEG frames, window coverage |
| All windows | `run_vera.py` `preprocess,infer` | Inputs for every window, raw responses JSONL, parsed segment-score JSON |
| Refine | `run_vera.py` `preprocess,infer,refine` | ImageBind feature identity, scene score, temporal score, final frame score arrays |
| Evaluate | `run_vera.py` `preprocess,infer,refine,evaluate` | Above plus metrics against supplied frame intervals |

Stage choices are ordered prefixes. A later stage always saves earlier stages.
For refinement, supply the official aligned ImageBind `.npy` with exactly one
row per ordered window. Its filename must match the video stem. For evaluation,
also supply the UCF-style annotation JSON.

After a full-video run, a strip shows every window's parsed label (red = 1,
grey = 0, striped = unparsed). Select a window to see its eight model inputs and
the response it produced.

## Scratch files and stage names

Every new run clears only `demo_runs/current/`, then writes to that same
location. Uploaded input is moved into that directory; unused uploads are
cleared. Save any artifact you need **before the next run**. Reloading the
page or restarting the lab server restores the last completed scratch run.
The app writes `stage_manifest.json` with ordered names such as
`01_sampled_input`, `03_window_predictions`, and `06_frame_timeline`. The stage
selector displays available files and lets you preview or download them.
Directories open 50 entries at a time, so you can inspect individual sampled and
decoded frames. Score arrays are plotted against their index; the downloadable
`.npy` stays complete.

The full runner keeps its native files under
`demo_runs/current/pipeline/run 1/output/<video-id>/`. The lab indexes those
files rather than copying large decoded-frame sets. The exact run config,
source snapshot, checksum information, logs, and status remain in the same
scratch tree. The warm process logs to `demo_runs/model_worker.log`. Original
videos and source features are never deleted. Model runs accept requests only
from the same-origin lab page with its current session token. The server also
locks the scratch directory across processes.

## Validation and acceptance

```bash
.venv/bin/python tests/run_tests.py --suite lab
.venv/bin/python tests/run_tests.py --suite all
for file in tools/web/*.js; do node --check "$file"; done
```

The `lab` suite runs the HTTP API with mocked launches and the real warm process
with the mock backend. It covers stage selection, dependency preflight, model
settings, die selection, scratch replacement, uploads, the per-window strip, and
artifact confinement. Real-model results are recorded in
`tests/reports/ONE_WINDOW_ACCEPTANCE.md`. On 2026-10-24 the real-model test passed on Oz
(CUDA, BF16) and on the Mac (MPS, FP16). A full 162-window RoadAccidents127 run on
Oz reproduced experiment01's frame ROC-AUC for that video (0.99494).

## Limitations

A one-window run is **not** the final refined timeline. The VLM does not receive
audio or every source frame. For videos that are not 30 FPS, the configured
300-frame window is not exactly ten real seconds.
