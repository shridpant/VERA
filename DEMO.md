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

The page opens on the pipeline, drawn in two lanes: what InternVL2 sees and says
for one window, and what happens across the whole video once every window has a
label. Stages the selected run includes are highlighted. During a run, the stage
working now is outlined and prints its latest real value (frames decoded, window
number, tokens generated so far, label, box status). Click a stage to jump to the
setting that changes it.

**Simple** (the default) asks only for the video and *One window* or *Whole video*,
and uses VERA's original settings with boxes on. *Whole video* runs as far as the
lab's inputs allow: refinement and metrics when the server has the video's feature
file and annotation, otherwise all windows with labels and boxes. **Advanced**
shows every knob; Simple never sends them.

| Knob (Advanced) | Stage it changes | Default |
|---|---|---|
| Video: drop, ⚄ die, or path | 1 · input | none; the die picks a random library video and start time |
| Run through stage | which stages run | one sampled window |
| Window centre | 2–3 · which anchor's eight images | 5 s (nearest anchor) |
| Guiding questions | 4 · prompt | VERA's released UCF-Crime questions |
| Checkpoint, device, precision, download | 5 · InternVL2 | verified local copy, auto, auto, no download |
| (fixed) FlashAttention, generation | 5 · InternVL2 | on for CUDA when installed; greedy, 1024 new tokens |
| Boxes, what to box | 7 · boxes | on for windows labelled 1; the model names the anomaly |
| Feature file | 8–9 · scene and temporal refinement | library match, if any |
| Annotations | 10 · metrics | the server's `--annotation-file` |

**Exact prompt** shows the text sent with every window, rendered by the runner's
own `prompt()` as you edit the questions. After a run it also shows that run's full
language-model input: InternVL2's chat template, its built-in (Chinese) system
message, VERA's prompt, and the image-token run counted (`<IMG_CONTEXT>×2048` for
eight images). VERA's call puts all eight images where the first `<image>`
placeholder is; the other `Frame N: <image>` lines stay as text, exactly as upstream.

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

## Watching a run

Nothing on this view is animated for effect. The model process appends each step
to `timeline.jsonl` as it happens: decoding, the window and its eight images, the
text so far (every quarter second), the parsed label, the naming call and each box,
refinement and metrics. The page reads that file about once a second.

- **Follow live** shows each window as soon as its images are sampled, then its
  text while it is generated, its label and its boxes.
- The strip above shows every window in time order: red is labelled 1, grey 0,
  striped unparsed, faded not reached yet, and a dot marks boxes. Click any window.
- **Play** steps through the windows in order at 1, 2 or 5 per second, a timelapse
  of what the model saw and said. After refinement, the final frame-score curve
  appears with a cursor at the window on screen.
- **Event log** lists every recorded event with its values.

## Boxes

For each window labelled 1 (with boxes on), the lab makes extra calls after VERA's
answer; labels and refinement are unchanged:

1. **Name it.** In the same conversation as the window (same eight images): “In at
   most eight words, name the main anomalous object or event in these images. Reply
   with the phrase only.” In Advanced you can type the phrase instead.
2. **Box it on each image.** One image per call, with InternVL2's model-card
   grounding prompt, 100 new tokens, greedy: “Please provide the bounding box
   coordinates of the region this sentence describes: `<ref>phrase</ref>`”.

The reply is read from the decode that keeps InternVL2's special tokens, for
example `<ref> red truck </ref> <box> [[342, 126, 1000, 712]] </box>`, on a 0–1000
grid of the 448×448 image the model saw, so the boxes are drawn on exactly those
images. Each image's result is one of:

- **box**: drawn in red, with its share of the image;
- **whole image**: the first box covers at least 95 % of the image. It is drawn
  dashed and labelled “not a location”;
- **no box**, or **unreadable box**: nothing is drawn.

The tagged text of every call is shown under the images and saved in
`boxes.jsonl` (full video) or `probe/boxes.json` (one window). These are the
model's own answers to “where is this?”. UCF-Crime has no box annotations, so
they are not validated localization. Grounding pilot WI-01 showed many whole-image
replies on later crash frames. On RoadAccidents127 at 74 s the Mac boxed the truck
on images 1–4 and returned whole-image replies for 5, 6 and 8.

**Your own boxes:** tick *Pick frames to box*, click images in any windows,
including ones labelled 0, type what to box, and press *Box N frames*. Only the
picked frames are sent, through the warm model, which loads first if needed. They
are drawn in orange and saved in `manual_boxes.jsonl`, with their events in
`manual_timeline.jsonl`. Nothing is boxed for windows labelled 0 unless you ask
this way.

Cost: on Oz each grounding call takes about 0.6 s and the naming call about 2.6 s
(WI-01), so a window labelled 1 adds about 8 s. On the Mac (MPS) it is about
2 s per image.

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


## Scratch files and stage names

Every new run clears only `demo_runs/current/`, then writes to that same
location. Uploaded input is moved into that directory; unused uploads are
cleared. Save any artifact you need **before the next run**. Reloading the
page or restarting the lab server restores the last completed scratch run.
The app writes `stage_manifest.json` with ordered names such as
`01_sampled_input`, `03_window_predictions`, `04_anomaly_boxes`, and
`07_frame_timeline`; `timeline.jsonl` and `model_input.txt` are listed too. The stage
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
with the mock backend. It covers stage selection (including *Whole video*),
dependency preflight, model settings, die selection, scratch replacement, uploads,
the timeline and prompt endpoints, manual boxes, and artifact confinement. Runner
tests cover the box parser on WI-01's real outputs, and a mock run in which only
windows labelled 1 are boxed and every step reaches the timeline. Real-model results are recorded in
`tests/reports/ONE_WINDOW_ACCEPTANCE.md`. On 2026-10-24 the real-model test passed on Oz
(CUDA, BF16) and on the Mac (MPS, FP16). A full 162-window RoadAccidents127 run on
Oz reproduced experiment01's frame ROC-AUC for that video (0.99494).

## Limitations

A one-window run is **not** the final refined timeline. The VLM does not receive
audio or every source frame. For videos that are not 30 FPS, the configured
300-frame window is not exactly ten real seconds.
