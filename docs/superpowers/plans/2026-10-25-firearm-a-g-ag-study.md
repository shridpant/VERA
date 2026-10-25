# Firearm A/G/AG Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run unchanged VERA (A), firearm grounding only (G) and VERA plus grounding in one response (AG) on all 398 VISILAB firearm videos on one RTX 6000 Ada, then produce T1–T11, P1–P9 and a compiled report.

**Architecture:** A new study driver (`run_firearm_study.py`) sits beside `run_vera.py` and reuses the runner's modules unchanged: `data.windows/preprocess`, `backends.prompt/strict_label`, `refine.refine`. A keeps the runner's exact `batch_chat` call. G and AG pass a different question through the same call, with the same eight images and the same packing. A unit ledger makes the run resumable. ImageBind features come from a vendored copy of LAVAD's loader, validated against VERA's released RoadAccidents127 features.

**Tech Stack:** Python 3.10 VERA env on Oz (torch 2.1.2+cu121, transformers 4.37.2, flash-attn 2.5.8), Python 3.11 `.venv` on the Mac for mock tests, unittest, scipy (`linear_sum_assignment`), PyYAML, matplotlib (analysis venv on Oz), pdflatex (Oz `/usr/bin/pdflatex`).

**Spec:** `studies/firearm/HANDOFF.md` (sha256 87af03d2…a42, copied from the user's handoff).

## Global Constraints

- Base: `feature/configurable-runner` at e13acce (it now contains the Lab code; the handoff's eb342d94 is an ancestor). Branch `study/firearm-a-g-ag`; never push or merge without instruction.
- InternVL2-8B revision 6fb9ad6924f69424e57fab2ab061d707688f0296, verified against MODEL_PROVENANCE.json; CUDA BF16; flash_attention true (upstream); greedy; num_beams 1; max_new_tokens 1024 in A, G and AG.
- Sampling: stride 16, nominal fps 30, window_seconds 10, 8 frames, 448×448 bicubic resize, ImageNet norm. No crops, blur, downsampling or causal windows. Actual fps recorded.
- A prompt is byte-for-byte `backends.prompt(cfg)`. G/AG prompts are saved verbatim. Prompts never contain category, annotation or A output.
- Packing (checked in the pinned `batch_chat`): `num_patches_list=[8]` puts all 8 images (8×256 IMG_CONTEXT) at the first `<image>`; `Frame2..Frame8: <image>` remain literal text. G and AG use exactly this packing.
- One GPU process at a time, Ada pinned by UUID through `setup/scripts/launch.sh`.
- Strict anomaly label = the final `Output:` field, isolated; invalid is missing, never 0. Legacy and runner-strict labels are also saved.
- No anomaly accuracy/AUC/AP. Categories are not anomaly truth.
- Bulky data on Oz `/local/datasets/sp2117/firearm_visilab/`; the NFS home is near quota.
- Commits: one per task, each after its tests pass; the commit message states the gate. Results are committed only after the run is verified.

## Review Focus

1. AG answers whose box lines contain digits after `Output` (legacy parser corruption) → strict label None, legacy label recorded, disagreement counted. Test in Task 4.
2. A crash or timeout mid-window, then resume → no duplicate complete units, failed attempt kept, output equals an uninterrupted run. Test in Task 6.
3. Phone video (1920×1080) vs CCTV (640×480) boxes → the same normalized 0–1000 geometry after the non-aspect-preserving 448 resize. Test in Task 2.
4. Truncated generation (1024 tokens, no EOS) → status `truncated`, never an empty box list or a label. Test in Task 4.
5. A video shorter than 15 windows, or with an invalid label → refinement exclusion with reason; initial labels kept. Test in Task 8.

---

### Task 1: Inventory, staging and registry

**Files:**
- Create: `vera_runner/firearm_data.py`, `run_firearm_study.py` (subcommand `prepare`), `tests/firearm/test_inventory.py`

**Interfaces:**
- Produces: `parse_folder(name) -> dict` with keys `action, camera, location, subject, brightness, take`; `staged_id(category, folder) -> str` = `f'{category}__{folder}'`; `expected_windows(frames, stride=16) -> int` = ceil(frames/stride); `build_inventory(original_root, staged_root) -> list[dict]` (id, category, source_path, sha256, size, width, height, fps_meta, frames_meta, frames_decoded, duration_s, camera, label_file, parsed fields); `write_registry(path, staged_root)`.

- [ ] Tests: `PAH1_C1_P1_V1_HB_3` parses to PAH1/C1/P1/V1/HB/3; `N1_C2_P5_V4_LB_2` likewise; an unparseable name raises; staged ids are unique for duplicate folder names in different categories; `expected_windows(225)==15`, `(224)==14`, `(1)==1`; staging refuses to overwrite an existing symlink with a different target.
- [ ] Implement. Staging uses symlinks to the read-only originals. The decoded count comes from a full OpenCV decode, as `data.preprocess` does.
- [ ] Run `prepare` on Oz: `inventory.json/.csv`, `registry.json` (dataset `firearm_visilab`, split `test`, no manifest), `PROVENANCE.md` (both URLs, md5 1bcf357d…, sha256 b52fa773…, the CC BY 4.0 vs CC BY-NC 3.0 discrepancy, Mendeley resume note). Gate: 398 rows; categories 141/139/118; decoded == metadata count, or every difference listed.
- [ ] Commit.

### Task 2: Annotation audit and coordinate conversion

**Files:**
- Create: `vera_runner/firearm_annotations.py`, `tests/firearm/test_annotations.py`, `studies/firearm/AUDIT.md`

**Interfaces:**
- Produces: `load_coco(path) -> dict`; `validate_coco(coco, frames, width, height) -> list[str]` (problems); `frame_index(image_id, origin) -> int`; `to_input_norm(bbox_xywh, width, height) -> [x1,y1,x2,y2]` on 0–1000 (equal to 448-input pixels × 1000/448); `frame_status(video) -> dict[int, str]` with values `positive`, `listed_no_box`, `negative_category`, `unknown`; `render_overlay(frame_path, boxes, out_path)`.

- [ ] Tests: a synthetic 640×480 and a 1920×1080 frame with a red rectangle at known xywh, resized by the runner's exact PIL call to 448, then red-pixel bounds vs `to_input_norm` (≤1 input pixel); IoU invariance under the anisotropic resize; origin 1 maps image_id 1 to frame 0; boxes out of bounds and unknown category ids are reported.
- [ ] Audit on Oz (evidence in `AUDIT.md`, montages in the data dir): for 12 positive videos (both cameras, both categories), overlay image_id k on frames k-1 and k at the fastest-moving boxes and at the first/last boxed frame; view them. Sample `listed_no_box` frames (gun visible or not?) and No_Gun frames (no firearm?). Inspect `label_antiguo.json`. Record the decided origin, which statuses are complete (precision allowed) and which are recall-only or unknown. Commit the decision as data (`annotation_policy.json`), not as code.
- [ ] Commit.

### Task 3: Condition prompts

**Files:**
- Create: `vera_runner/conditions.py`, `studies/firearm/prompts/G.txt`, `studies/firearm/prompts/AG_insert.txt`, `tests/firearm/test_conditions.py`

**Interfaces:**
- Produces: `CONDITIONS = ('A','G','AG')`; `build_prompt(cfg, condition, study) -> str`; `prompt_hashes(cfg, study) -> dict[str,str]`.
- G = `''.join(f'Frame{i+1}: <image>\n')` (the runner's prefix) + G.txt. AG = A with AG_insert.txt inserted once, immediately before the marker "Based on the analysis above, please conclude…". The A text's other bytes are unchanged and its final Output format is kept.

- [ ] Tests: `build_prompt(...,'A') == backends.prompt(cfg)`; AG minus the insert == A; the insert lies before the marker and after question 5; G and AG contain 8 `<image>` and the same box-format lines; no prompt contains `Handgun`, `Machine_Gun`, `No_Gun`, `label`, `annotation`, `category`.
- [ ] Write G.txt/AG_insert.txt: one class `firearm` (handguns and long guns), every frame regardless of anomaly, one line per frame `FrameK: [[x1, y1, x2, y2], ...]` with integer 0–1000 xyxy; `[]` = none seen; `unknown` = cannot assess; AG gives the lines before the final Output.
- [ ] Commit (prompts are draft until Task 13 locks them).

### Task 4: Parsers

**Files:**
- Create: `vera_runner/slots.py`, `tests/firearm/test_slots.py`

**Interfaces:**
- Produces: `final_output_label(text) -> int|None` (last `Output\s*:` field; the remainder, stripped of fences/brackets/quotes, must be exactly 0 or 1); `split_answer(text) -> (before_output, output_field)`; `parse_slots(text, n_slots=8, scale=1000, whole_image_area=0.95, truncated=False) -> {'response_status', 'slots':[{'slot','status','boxes','whole_image','raw'}]}`. Slot status: `boxes|empty|unknown|malformed|missing|duplicate|truncated`. Response status: `ok|truncated|no_slots`.

- [ ] Tests: plain and `<box>`-wrapped lists; `[]`; `unknown`; `none`→malformed; slots out of order; missing Frame8; duplicate Frame3 (both kept, status duplicate); values >1000, x1≥x2, 3 numbers → malformed; floats kept; all values ≤1 flagged `scale_suspect`; whole-image box (area ≥ 0.95) flagged but kept; truncated text → every slot `truncated`; AG with box digits after "Output:" → `final_output_label` None while the legacy rule gives 0; AG "Output:\n1" after the boxes → 1; two Output fields → the last one is used, and the runner's `strict_label` gives None (disagreement recorded).
- [ ] Commit.

### Task 5: Backend call with an explicit question and telemetry

**Files:**
- Modify: `vera_runner/backends.py` (`InternVL2.predict` gains `question=None`; adds `last_call`); `Mock` gains `question=None` and synthetic slot boxes for G/AG
- Test: `tests/firearm/test_backend_question.py`

**Interfaces:**
- Produces: `predict(images, on_text=None, question=None) -> {'raw_response','input_token_ids'}` (A: `question=None` → `self.question`, identical call); `backend.last_call = {'output_ids','tagged_text','eos','generated_tokens','peak_gpu_bytes','seconds'}`.

- [ ] Tests (fake model, as in `tests/backends`): the `batch_chat` kwargs for A are unchanged from the existing equality test; G passes its question with `num_patches_list=[8]` and the same pixel tensor; `eos` is true iff 92542 is among the generated ids; a fake 1024-id output without EOS gives `eos False`.
- [ ] Run the full existing suite (52) to prove A is untouched. Commit.

### Task 6: Ledger, resume and the study loop

**Files:**
- Create: `vera_runner/ledger.py`, `vera_runner/study.py`; `run_firearm_study.py` subcommands `run`, `validate`, `reconcile`
- Test: `tests/firearm/test_study_run.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: run dir layout `spec.json` (immutable: resolved config, prompt texts and hashes, inventory sha, code commit, condition order) · `sessions/NNN/` (status, heartbeat, env, GPU UUID) · `ledger.jsonl` (start/end events) · `videos/<id>/{video.json, windows.json, inputs/<anchor>/NN.png, records/<cond>.jsonl}` · `input_ids/<sha>.json`. A unit is (video, anchor, condition). It is complete when a `complete` record matches the config, model, prompt and input (8 rgb sha256) hashes. Otherwise a new attempt runs, up to `max_attempts`. A record keeps the runner's A fields plus `final_output_label`, `strict_label`, `upstream_label`, `slots` (G/AG), `eos`, `truncated`, `generated_tokens`, `peak_gpu_bytes`, `attempt`, `session`.
- Condition order per window is frozen: A, G, AG. Each call is a fresh `batch_chat` with no history. A per-call watchdog writes a `timeout` end event and exits 124. `run` refuses a run dir whose spec differs.

- [ ] Tests (mock backend, synthetic videos): a full run gives 3 complete units per window; the 8 rgb hashes are identical across A/G/AG; mock questions never contain an earlier response; an injected exception at call 5, then `run` again → call 5 has a failed attempt plus a later complete one, no unit has two completes, and records equal a clean run; a changed prompt file → refusal; a corrupted input PNG → rerun of that window; `reconcile` lists every expected unit with a disposition.
- [ ] Commit.

### Task 7: ImageBind features (LAVAD procedure)

**Files:**
- Create: `third_party/lavad_imagebind/` (vendored from lucazanella/lavad@1ad46c66 `libs/ImageBind/imagebind/{models/*,data.py}` with LICENSE and ATTRIBUTION.md), `vera_runner/imagebind_features.py`, `run_firearm_study.py features`, `tests/firearm/test_feature_clips.py`, `studies/firearm/requirements-imagebind.txt`

**Interfaces:**
- Produces: `clip_frame_indices(anchor, n_frames, fps=30, clip_seconds=10, num_samples=10) -> list[int]` (LAVAD: `range(max(c-150,0), min(c+150,n))` then `np.linspace(0,t-1,10,dtype=int)`); `extract(video_frames_dir, n_frames) -> np.ndarray[ceil(n/16), 1024]`; per-video `.npy` + `features_manifest.json` (weights sha256, code commit, dtype).

- [ ] Tests: clip indices equal LAVAD's `uniform_temporal_subsample` on synthetic lengths (1, 15, 225, 450); rows == number of anchors.
- [ ] Env on Oz: `/local/.../envs/imagebind` venv with `--system-site-packages` over the VERA env, plus pytorchvideo@28fe037d, fvcore, iopath, ftfy and a wheel-backed av; `pip freeze` saved. Weights `imagebind_huge.pth` → `/local/.../models/imagebind/` with sha256.
- [ ] Validation gate: RoadAccidents127 (archived UCF video) through our extractor vs the released `RoadAccidents127_x264.npy`: same shape (162×1024), median row cosine ≥ 0.99, and refinement on experiment01 labels gives identical frame scores. Report the numbers; if the gate fails, stop and diagnose before feature extraction.
- [ ] Commit.

### Task 8: Refinement of A and AG

**Files:**
- Create: `run_firearm_study.py refine`, `tests/firearm/test_refine_study.py`

**Interfaces:**
- Consumes: `refine.refine(records, features, length, stride, cfg)` unchanged; records built from strict `final_output_label`.
- Produces: `refined/<cond>/<id>/{scene,temporal,frame_scores}.npy`, `refined/exclusions.csv` (video, condition, reason ∈ {too_few_windows, invalid_labels, missing_features, row_mismatch}), `refined/feature_hashes.json` (the same sha for A and AG).

- [ ] Tests: a video with 14 windows → excluded `too_few_windows`, labels kept; one invalid label → `invalid_labels`; A and AG read the same feature file sha; numerical equality with `compute_auc.py` comes from the existing refinement test.
- [ ] Commit.

### Task 9: Localization matching and metrics

**Files:**
- Create: `vera_runner/localization.py`, `tests/firearm/test_localization.py`

**Interfaces:**
- Produces: `iou(a,b)`; `match(preds, gts, threshold) -> list[(pi, gi, iou)]` via `scipy.optimize.linear_sum_assignment` on cost `-(1+IoU)` for IoU ≥ threshold, else 0 (maximize the number of matches, then total IoU; candidates are pre-sorted by (x1,y1,x2,y2) for deterministic ties); `score_instance(slot, gts, status, threshold) -> {'tp','fp','fn','matched_ious','fp_evaluable'}`. FP counts only where the frame status is complete. A slot that is malformed, unknown, missing or truncated counts every GT as FN end to end and is excluded from precision.

- [ ] Tests: hand-computed IoUs; two predictions on one GT → 1 TP + 1 FP; a tie resolved identically under input permutation; a whole-image box is kept in the metric; FP on a `listed_no_box` frame counted only if that status is complete; No_Gun predictions are FP.
- [ ] Commit.

### Task 10: Analyses, tables and plots

**Files:**
- Create: `studies/firearm/analysis/{load.py,bootstrap.py,tables.py,plots.py}`, `run_firearm_study.py analyze|plots`, `tests/firearm/test_analysis.py`

**Interfaces:**
- Consumes: the run dir, inventory, annotation policy, refined outputs.
- Produces: `analysis/tables/T1..T11.{csv,tex}`, `analysis/figures/P1..P9.{pdf,png}`, `analysis/per_video.csv` (T9), `analysis/qualitative.json` (selection rule and IDs), `analysis/numbers.json` (every number the report cites).
- Bootstrap: resample source videos within category, B=2000, seed 20261025, percentile 95 % CI. Pooled-window and equal-video summaries are both reported. Sensitivity clustering by recording group only if the audit identifies groups. Size bins = GT-area tertiles computed from GT alone.
- Percent anomaly uses valid windows; the invalid rate uses all scheduled windows; T4 pairs only windows valid in both A and AG.

- [ ] Tests on a mock-run fixture: T2 counts add up to the scheduled units; T4 support is the intersection; T5 cells sum to the paired count, with unparsed pairs separate; T8 shows N/A with a reason where completeness is missing; the bootstrap is reproducible under its seed; every table/figure file exists.
- [ ] Commit.

### Task 11: Annotated YAML, conversion and validation

**Files:**
- Create: `studies/firearm/experiment_A_G_AG.annotated.yaml`, `run_firearm_study.py convert|validate`, `tests/firearm/test_convert.py`, `studies/firearm/README.md` (short, TOC: purpose, commands, layout, knobs)

**Interfaces:**
- The YAML has `runner:` (a config that `vera_runner.config.load` accepts unchanged), plus `conditions`, `parsing`, `matching`, `execution`, `analysis`. Every knob is commented and marked [scientific] or [operational]. `convert` writes `experiment_A_G_AG.json`; `validate` loads it, checks prompts/registry/model provenance/GPU and writes nothing outside `checks/`.

- [ ] Tests: YAML → JSON round-trip equals the hand-written expected dict; an unknown key fails; the `runner` block passes `config.load`; a changed scientific knob changes the spec hash.
- [ ] Commit.

### Task 12: Report

**Files:**
- Create: `studies/firearm/report/{main.tex,references.bib,build.sh}`

- [ ] `build.sh` copies tables/figures from `analysis/`, runs pdflatex+bibtex twice, and zips the source. `main.tex` uses `\input` for every number and table. Title: "Firearm Localization and Its Effect on VERA's Anomaly Judgments". Sections: methods (A/G/AG, prompts verbatim, packing, native timing, full inclusion, audit, strict parsing, paired metrics, exclusions, offline refinement, compute), results, limitations (bundled prompt change, packing, no anomaly truth, short-clip refinement), future work.
- [ ] Gate: it compiles on Oz from a mock-run analysis. Commit.

### Task 13: Real smoke tests, pilot and prompt lock (Oz)

- [ ] `validate`, then a smoke run on 6 windows (2 per category, both cameras; seeded choice recorded): prompt identity, frame hashes, slot parsing, coordinates drawn on the inputs (viewed), token IDs, EOS, timing, memory.
- [ ] Slot-interpretability check: on 2 Handgun windows, keep slot k ∈ {0,3,7} and grey the others with the runner's grey intervention; does G report boxes only at Frame k+1? Record it in `PROMPT_LOG.md` (diagnostic; never analysed as data).
- [ ] A preservation: `run_vera.py` baseline on the same 2 windows → raw text identical to study A.
- [ ] Revise the prompts only for format failures, logging every version and example. Then lock: the prompt sha256 goes into the YAML. Label the study exploratory if the pilot changed the prompts.
- [ ] Profile: s/window per condition, GB per window → projected GPU-hours and disk. Commit.

### Task 14: Independent review before the full run

- [ ] A fresh reviewer agent (most capable model) gets the handoff, the diff and the smoke artifacts. It checks scope, frame/coordinate validity, leakage, denominator consistency, identical A/G/AG inputs and parser robustness. Each finding is fixed or answered in `studies/firearm/REVIEW.md`. AI review is not a human signature.

### Task 15: Full run, features, refinement, reconciliation, analysis, report

- [ ] Launch `run` via `launch.sh --gpu <Ada UUID>` inside a relaunch loop (resume until complete, max 20 relaunches). Monitor with a background wait, not polling.
- [ ] Then `features` (same GPU, after inference), `refine`, `reconcile` (every expected unit has a disposition), `analyze`, `plots`, `report/build.sh`.
- [ ] Copy the compact results (tables, figures, PDF, source zip, per-video CSV, configs, ledger summary) to the Mac. Commit the results after verification. Update memory.
