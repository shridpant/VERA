# experiment01 / run_001: one-video VERA end-to-end validation

**Status: COMPLETE.** All three VERA inference steps ran on one video. Validation passes all 15 checks, and the outputs
can be inspected. This is a setup validation on a single clip. It does not reproduce the benchmark AUC/AP and does not
show that the explanations help people.

- Run: `~/vera_research/experiments/experiment01/runs/run_001` on oz.gccis.rit.edu.
- Job: `exp01_run001_20261022T155048_12135`, 15:50:49 to 15:57:01 EDT, 2026-10-22, exit 0.
- GPU: NVIDIA RTX 6000 Ada Generation (GPU 4, UUID `GPU-6a490dc2-d384-9010-bc0b-b0b76076bae3`, 46 GB). The handoff's
  "2 × A100 24 GB" does not describe this host, and one GPU was enough.

## What ran

| | |
|---|---|
| Video | UCF-Crime `RoadAccidents127_x264.mp4` (primary; appendix B.5 example). Taken from the official UCF-Crime Dropbox zip `Anomaly-Videos-Part-3.zip` by reading only that member over HTTP ranges (20.4 MB fetched, CRC-32 `b1af440a` verified). |
| Video identity | sha256 `618a6a63273de405bb70b4af61aa1d7d0ffb279dd41e531b1575332be48e06ff`; H.264, 320×240, **30.0 fps**, **2580 decoded frames** (container also says 2580; VERA manifest says 2580), 86.0 s |
| Model | `OpenGVLab/InternVL2-8B` @ `6fb9ad6924f69424e57fab2ab061d707688f0296`; **BF16**; frozen, `eval()`. FlashAttention 2 was active in both the vision encoder and the LLM (`InternLM2FlashAttention2`). |
| Source | VERA `15b8bcb8` + local patch → `008ed401` (branch `vera-research-local`). The executed code is the root `generate_initial_score.py`; the misspelled `Inference/generate_intial_score.py` was not used. |
| Prompt | `VERA_learner_instruct.txt` with the 5 released UCF-Crime guiding questions inserted verbatim. It is the same for all 162 windows (one model-input hash). Rendered chat input, including InternVL's built-in system message: `results/model_input_rendered.txt`. **No audio was supplied.** |
| Sampling | anchors `range(0, 2580, 16)` → **162 windows**; window `[a−150, a+150)` clipped to the video; 8 uniformly spaced frames; 448×448; `num_beams=1, do_sample=False, max_new_tokens=1024` (InternVL adds `eos_token_id=92542`) |
| Step 2 features | the authors' released ImageBind features for this video (`ucf_crime_vision_features.zip`, single member by HTTP range), shape (162, 1024) float64, sha256 `f0c2470c…`. They were not recomputed here. |
| Steps 2–3 | `compute_auc.py` math (top-15% cosine neighbours = 24, softmax(10·cos); Gaussian smoothing size 15 σ 10; round to 0.1; expand ×16; position weighting σ = 1290). The output is **identical to the unmodified upstream script** (check C13). |

## Findings

**Step 1: raw window decisions** (`results/raw_responses.jsonl`, `results/segment_scores.json`)

- All 162 outputs had a single well-formed `Output:` field, so 0 are unparsed. **13 windows were labelled 1** and 149 labelled 0.
- The strict parser, the original root-script parser and the misspelled-variant parser agree on all 162 windows.
- Windows labelled 1, by anchor frame (time):
  - **496** (16.5 s) and **704** (23.5 s): explanations about a person crossing the road "in a way that is not
    consistent with their usual behavior";
  - **2128** (70.9 s), **2176** (72.5 s), **2208–2288** (73.6–76.3 s; 6 windows), **2320** (77.3 s) and **2352–2368**
    (78.4–78.9 s): explanations about "a red truck colliding with a train" or "a red truck and a train crossing the same road at the same time".
- Every other window is labelled 0. Typical text: "The video shows a red truck and a white van driving on a road with
  other vehicles and traffic lights…".

**Benchmark annotation** (`results/benchmark_annotation.json`; used only for the diagnostic below, never in Steps 1–3)

- UCF-Crime marks frames **2160–2300** (72.0–76.7 s). Like `compute_auc.py`, this report reads it as zero-based
  `[2160, 2300)`.
- The convention of the original annotation file (1-based or inclusive end) was not checked. It moves at most one frame
  at each end.

**Steps 2–3: refined scores** (`results/scene_refined_scores.*`, `results/temporal_refined_scores.*`, `results/frame_scores.*`)

| Stage | Range | Peak |
|---|---|---|
| Scene-refined segment score | 0.000–0.550 | |
| Temporally smoothed segment score | 0.000–0.504 | anchor 2240 (74.7 s) |
| **Final frame score** | 0.000–**0.388** | frame 2208 (73.6 s) |

- Frames scoring at least half of the maximum form one contiguous span, 2128–2479 (70.9–82.6 s).
- **Every frame before 2032 (67.7 s) scores exactly 0.** Nonzero scores span frames 2032–2579 (67.7–86.0 s).
- The two isolated early windows labelled 1 smooth to small values: anchor 496 to 0.034, anchor 704 to 0.038. Both
  round to 0.0, so their frames score 0.

**Per-video diagnostic** (`results/diagnostics.json`; this video only, not comparable to the paper's benchmark numbers)

- Both classes are present (140 annotated frames, 2440 others).
- Frame-level ROC AUC on this video: 0.9949.
- Mean final score is 0.356 inside the annotation and 0.033 outside it.

**Reference comparison** (labelled REFERENCE ONLY; the authors' released initial labels for this video, not generated here)

- The released labels agree with this run's fresh labels on **156/162 windows (96.3%)**:
  - this run says 1 where the released file says 0 at 496, 704 and 2224;
  - the released file says 1 where this run says 0 at 672, 1280 and 2192.
- The window bounds in the released file match this run's exactly.
- After Steps 2–3, the two label sets give **byte-identical frame scores**. The intermediates do differ (scene-refined
  by up to 0.057 on 82 segments, smoothed by up to 0.031 on 132), but `compute_auc.py` rounds each smoothed segment
  score to one decimal before expanding to frames.

## Example windows (verbatim, chosen by rule)

Rule, fixed in `setup/scripts/summarize_run.py` before reading any text: windows at positions round(q·161) for
q ∈ {0, .25, .5, .75, 1}, plus the highest temporally refined window and the first window with raw label 1. All 162
windows are kept in `results/raw_responses.jsonl`. The **scores** below are refined scores; each **explanation** was
written once, for that window's 8 frames, and refinement never rewrites it.

| Rule | Anchor (time) | Frames | Raw | Scene | Temporal |
|---|---|---|---|---|---|
| q = 0 | 0 (0.0 s) | 0–150 | 0 | 0.000 | 0.000 |
| first raw 1 | 496 (16.5 s) | 346–646 | 1 | 0.064 | 0.034 |
| q = .25 | 640 (21.3 s) | 490–790 | 0 | 0.039 | 0.031 |
| q = .5 | 1280 (42.7 s) | 1130–1430 | 0 | 0.000 | 0.000 |
| q = .75 | 1936 (64.5 s) | 1786–2086 | 0 | 0.000 | 0.000 |
| highest temporal | 2240 (74.7 s) | 2090–2390 | 1 | 0.532 | 0.504 |
| q = 1 | 2576 (85.9 s) | 2426–2580 | 0 | 0.213 | 0.117 |

Raw text, unedited:

- **Anchor 2240** (highest refined score):
  > Answers to Prompt Questions: The video shows a red truck and a train crossing the same road at the same time, which is an unusual and potentially dangerous situation. There is visible damage to the train, indicating an anomaly.
  > Output: 1
- **Anchor 0:**
  > Answers to Prompt Questions: The video shows a red truck and a white van driving on a road with other vehicles and traffic lights. The vehicles are in their typical positions and engaging in activities consistent with their usual behavior. There is no visible damage or unusual movement, and no unusual sounds or noises are indicated. Therefore, there is no anomaly in the video.
  > Output: 0
- **Anchor 496** (first raw 1):
  > Answers to Prompt Questions: The video shows a person crossing the road in a way that is not consistent with their usual behavior. There are no vehicles or objects in unusual positions, and no visible damage or unusual movement. There are no unusual sounds or noises. Based on the analysis, there is an anomaly in the video.
  > Output: 1
- **Anchor 2576** (last window):
  > Answers to Prompt Questions: The video shows a train on the tracks, and there are no people or vehicles in unusual positions or engaging in activities that are not consistent with their usual behavior. There is no visible damage or unusual movement in the video. There are no unusual sounds or noises in the video. Based on the analysis, there is no anomaly in the video.
  > Output: 0

## Flags for later human inspection (not judged here)

- **Sound claims without audio.** Guiding question 5 asks about sounds, but only frames were given. 140 of 162
  responses mention sound, and all of them assert an absence ("no unusual sounds or noises"). No response claims a sound.
- **Repetition.** Only 67 of the 162 explanations are distinct; 95 windows repeat an earlier text verbatim.
- **Explanation and score can disagree after refinement.**
  - Anchor 496 explains an anomaly. Its smoothed score is 0.034 and its frames score 0.
  - The last window explains "no anomaly", but its scene-refined score is 0.213.
  - Anchors 2224, 2256 and 2272 are labelled 1 while saying "There is no visible damage or unusual movement". They cite
    the simultaneous truck/train crossing instead.
- **Claims not checked against the video:** for example "visible damage to the train" and "debris on the ground".
  Neither this run nor the agent viewed the frames. Grounding and faithfulness are for the planned human review.
- **Offline, retrospective context.** Each window includes up to 5 s of *future* frames. Scene refinement draws on
  segments from the whole video, and the position weighting depends on the video's total length. This is not a
  causal live-alert setting.
- **Rounding.** Refined scores are rounded to 0.1 before frame expansion. This hid the 6-window disagreement with the
  released labels completely. Refined scores are not calibrated probabilities.

## Runtime and resources (measured)

- **Model load:** 4.2 s; the safetensors files are on NFS, and the page cache was warm after the smoke tests.
- **Generation:**
  - 346.2 s in total for 162 windows: mean 2.14 s, median 2.01 s, max 3.4 s.
  - 76.5 new tokens on average, max 129. No window hit the 1024-token limit.
- **Wall time:** VLM stage 363.7 s; whole job 372 s. Frames came from the shared cache; the first extraction took 46.3 s
  during the smoke test.
- **Memory:**
  - peak GPU memory allocated by torch: 17.96 GiB;
  - peak used by the device (nvidia-smi, 1 Hz): 19,552 MiB;
  - peak host RSS: 6.15 GiB.

## Validation and reproducibility

- **`validate_run.py` passes 15/15 checks** (`results/validation.json`). They include:
  - one video;
  - exactly 162 = `len(range(0, 2580, 16))` windows;
  - 8 in-range frames per window, from the upstream formula;
  - the exact prompt, with no video name, category or annotation frame in it;
  - strict labels on all windows, and the parser on 13 valid and malformed cases;
  - feature rows matching anchors one-to-one;
  - finite, in-range arrays, with 2580 frame scores;
  - the same video checksum everywhere;
  - frame scores identical to the **unmodified** `compute_auc.py` run in a sandbox.
- **Correction disclosed.** On its first pass C07 failed for every window. InternVL's `batch_chat` adds
  `eos_token_id = 92542` (`<|im_end|>`) to the generation-config dict in place, as it does upstream. The records were
  right; the check was wrong. It now requires the configured settings exactly, plus that single injected key. Both
  validator logs are in `logs/`.
- **GPU-free acceptance tests:** `setup/scripts/test_vera_pipeline.py` passes 30/30. Besides the parser, anchor and
  window checks, it confirms `vera_refine` == unmodified `compute_auc.py` on three synthetic videos and covers the short-video edge cases.
- **Resume:** a refine-only rerun on this run directory (`--stages refine`, no VLM calls) reproduced all 11 result
  files byte for byte (`logs/resume_refine_only_*.log`).
- **Fresh rerun:** `run_002`, the same command in a new directory, regenerated all 162 windows. Raw responses
  (162/162), labels, model inputs and frame scores are **identical** to run_001.

## Deviations from the original scripts (all recorded in `run_manifest.json` / `config.yaml`)

1. **Window loop.** It runs in `setup/scripts/run_experiment.py`, with the same arithmetic as
   `generate_initial_score.py`'s main loop. The VERA classes, image transform, prompt assembly and `batch_chat` call
   are imported from the pinned source.
2. **Source patch `0001`.** It makes the script importable, drops unused imports, and uses `nn.Module` in place of
   `pl.LightningModule`. `forward()` uses `self.generation_config` and keeps raw responses. The loader raises instead of
   substituting a random video. The list-vs-`.item()` bug is fixed.
3. **Labels.** They come from a strict parser of the `Output:` field. Unparseable text would get no label; none occurred.
4. **Model source.** A checksum-verified local copy of revision `6fb9ad69` was used. The exact InternVL2-8B revision
   used in the paper is not stated.
5. **Step 2 features.** These are the authors' released features; their extraction settings are not published with
   the file. The LAVAD/ImageBind extractor is not set up.
6. **Package versions.** VERA does not specify them. This run used torch 2.1.2+cu121, transformers 4.37.2 and
   flash-attn 2.5.8.
7. **Hard-coded 30 fps.** It is kept, and matches this clip's actual 30.0 fps, so windows are exactly 10 s.

## Artifacts (relative to this run directory)

- `run_manifest.json` (host, GPU, commits, model revision, checksums, settings, commands, timings, deviations),
  `environment.txt`, `resolved_config.yaml`
- `results/raw_responses.jsonl` (every window: indices, timestamps, exact prompt, raw text, labels, runtime),
  `results/model_input_rendered.txt`
- `results/segment_scores.json`, `results/segment_scores_vera_format.json`
- `results/scene_refined_scores.{json,npy}`, `results/temporal_refined_scores.{json,npy}`
- `results/frame_scores.{npy,csv}`, `results/frame_scores_before_position_weighting.npy`
- `results/benchmark_annotation.json`, `results/diagnostics.json`,
  `results/reference_official_scores_frame_scores.npy` (reference only)
- `results/frames_identity.json`, `results/summary.json`, `results/validation.json`, `logs/`

## Reproduce (on oz, from `~/vera_research`)

```bash
RUN=experiments/experiment01/runs/run_003   # always a new directory
bash setup/scripts/launch.sh --name exp01 --log-dir $RUN/logs -- \
  setup/environment/env/bin/python setup/scripts/run_experiment.py --config experiments/experiment01/config.yaml --run-dir $RUN --new-run
setup/environment/env/bin/python setup/scripts/validate_run.py --run-dir $RUN
```

Refinement only, on an existing run:
`setup/environment/env/bin/python setup/scripts/run_experiment.py --config experiments/experiment01/config.yaml --run-dir experiments/experiment01/runs/run_001 --stages refine`.
