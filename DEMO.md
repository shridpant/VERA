# VERA Lab

VERA Lab is a separate, localhost-only viewer over the existing configurable
runner. It can run **one window** or a **complete video through a selected stage**.
It does not train or modify the InternVL2 model. It uses the same eight-frame
sampler, prompt builder, parser, and full-video refinement code as the runner.
The frame-game and score pages are labelled teaching examples, not model data.

Original VERA's `generate_initial_score.py` discards the VLM response after
parsing a 0/1 label. It saves per-window `start`, `end`, and `score` in one JSON
file per video. The configurable runner retains the raw text and its inputs,
then saves every numerical refinement stage. VERA does not cite supporting
sampled frame IDs for each generated sentence. This lab shows the eight input
images; it does not add or validate citations.

## Run where the dependencies and checkpoint are installed

This Mac's Python 3.14 environment lacks `cv2`, `transformers`, and several
other model packages, and its local model mirror has no weight shards.
Installing only `cv2` would expose the next missing dependency. The UI now
checks dependencies **before** launching a job and shows the missing names.
The previously set up Oz Python environment has the pinned packages and a
verified InternVL2-8B checkpoint.

On Oz, start the server from this checkout with the existing Python environment.
Use an RTX 6000 Ada, not one of the smaller A16 cards. Oz CUDA ordinals
do not match the indices printed by `nvidia-smi`; select the verified GPU UUID
rather than numeric index. Adapt the paths if your Oz folder differs:

```bash
cd ~/vera_research/setup/source/VERA-lab
CUDA_VISIBLE_DEVICES=GPU-6a490dc2-d384-9010-bc0b-b0b76076bae3 ~/vera_research/setup/environment/env/bin/python tools/demo_server.py \
  --port 8876 \
  --video-dir ~/vera_research/shared/videos/ucf_crime \
  --model-path ~/vera_research/shared/models/OpenGVLab/InternVL2-8B/6fb9ad6924f69424e57fab2ab061d707688f0296 \
  --feature-dir ~/vera_research/shared/features/vera_official_ucf_imagebind \
  --annotation-file Data/UCF_Eval.json
```

From your Mac, tunnel to the Oz-only service and open
`http://127.0.0.1:8765/`:

```bash
ssh -N -L 8765:127.0.0.1:8876 sp2117@oz.gccis.rit.edu
```

If port 8765 is already serving the old Mac page, stop that local Python
server first. The Oz server binds only to loopback. It does not expose Oz to
other computers.

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
also supply the UCF-style annotation JSON. The server's configured sample
library fills these paths when the matching files exist.

Select **I'm feeling lucky** to choose an input without starting inference.
A dropped video is previewed in the browser and uploaded in bounded chunks.
At most one model run runs at a time. The run uses the current Python
interpreter and a pinned model revision. CUDA + BF16 is appropriate on Oz's
RTX 6000 Ada; MPS + FP16 is optional on a provisioned Mac.

## Scratch files and stage names

Every new run clears only `demo_runs/current/`, then writes to that same
location. Uploaded input is moved into that directory; unused uploads are
cleared. Save any artifact you need **before the next run**. Reloading the
page or restarting the lab server restores the last completed scratch run.
The app writes
`stage_manifest.json` with ordered names such as `01_sampled_input`,
`03_window_predictions`, and `06_frame_timeline`. The stage selector displays
available files and lets you preview or download them. Directories open in a
50-entry-at-a-time browser so you can inspect individual sampled and decoded
frames. For score arrays it
shows a downsampled plot; the downloadable `.npy` remains complete.

The full runner keeps its native files under
`demo_runs/current/pipeline/run 1/output/<video-id>/`. The lab indexes those
files rather than copying large decoded-frame sets. The exact run config,
source snapshot, checksum information, logs, and status remain in the same
scratch tree. Original videos and source features are never deleted.
Model runs accept requests only from the same-origin lab page with its current
session token. The server also locks the scratch directory across processes.

## Validation and limitations

```bash
python -m unittest discover -s tests/demo -v
python -m py_compile tools/demo_server.py tools/window_probe.py tools/stage_view.py
for file in tools/web/*.js; do node --check "$file"; done
```

The HTTP tests mock model launches. They verify stage selection, dependency
preflight, scratch replacement, uploads, and artifact confinement. The full
runner also has separate tests. A one-window model call and full-video run
should be accepted on Oz before using outputs in the thesis. A one-window run
is **not** the final refined timeline. The VLM does not receive audio or every
source frame. For videos that are not 30 FPS, the configured 300-frame window
is not exactly ten real seconds.
