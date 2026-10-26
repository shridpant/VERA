# Firearm VERA package: detector module, multi-category loop, boxes and importance

Plan written 2026-10-26 from the user's request; the user skipped the design questions, so every
choice not quoted from the user is marked **(my call)**. Builds on Training 1
([variants plan](2026-10-26-firearm-training1-variants-plan.md)), the A/G/AG study
(`study/firearm-a-g-ag`) and the grounding study (`experiment/firearm-grounding`).

## Contents

- [Why](#why)
- [What the package outputs](#what-the-package-outputs)
- [Goals, non-goals, anti-goals](#goals-non-goals-anti-goals)
- [Models](#models)
- [Module 1: the box detector](#module-1-the-box-detector)
- [Module 2: the multi-category training loop](#module-2-the-multi-category-training-loop)
- [Module 3: amalgamation (boxes and importance score)](#module-3-amalgamation-boxes-and-importance-score)
- [Phases and GPU schedule](#phases-and-gpu-schedule)
- [Decision gates](#decision-gates)
- [Final evaluation](#final-evaluation)
- [Downloads and environments](#downloads-and-environments)
- [Risks](#risks)
- [Deliverables](#deliverables)
- [Calls I made](#calls-i-made)

## Why

Three findings from 25–26 October:

- **VERA's packing can't draw boxes.** VERA puts eight frames into one image input. Asked for boxes
  that way, InternVL2-8B returned `[]` for 99.6 % of frame slots, and every box it did write
  covered the whole image (A/G/AG).
- **The model looks at the gun but doesn't recognise it.**
  - Its attention lands on the annotated firearm (lift +1.54 over another video's boxes).
  - Every correct firearm answer changes when the gun is greyed out.
  - Yet only 15.6 % of handgun videos are named correctly; the rest are called No_Gun (grounding
    study).
- **Rewording the questions moves accuracy less than the training order does.** Faithful VERA and
  the fixed variant each spread about 15 points across three run seeds, and selection rests on 62
  validation videos (Training 1).

So the package separates *finding* the firearm (a detector on single frames, from the newest
foundation model) from *naming* it. Its training loop is built for several categories, and each
decision comes with boxes and an importance score that says how much each frame and box mattered.

## What the package outputs

For each video:

- **Labels:** the category (Handgun / Machine_Gun / No_Gun), the action group (6) and the action
  code (41, exploratory).
- **Explanation:** the model's answers to its learned guiding questions, as in VERA.
- **Evidence:** the 8 sampled frames, each with:
  - its *frame importance*;
  - the detector's boxes (label `handgun`, `long gun`, `firearm` or `person`, coordinates on a
    0–1000 grid);
  - each box's *box importance*.

The output is one JSON record per video. VERA Lab draws the boxes and importance on its existing
box timeline.

## Goals, non-goals, anti-goals

**Goals**

1. **Detector.** Firearm boxes from a separate detector module built on the latest foundation
   VLM, tested with thinking on and off.
2. **Multi-category loop.** VERA's training loop rebuilt for several categories. It stays generic:
   the classes come from a codebook, so the same loop runs on UCF-Crime's 14 classes later.
3. **Boxes and importance in one output.**
4. **Evidence.** Each part is checked by an ablation before the next part depends on it.

**Non-goals** (my call)

- No weight training or LoRA; every model stays frozen, as in VERA.
- No new human annotation.
- No anomaly AUC claims.

**Anti-goals**

- Reading the test split more than once per final configuration.
- Claims from a single run seed.
- Counting whole-image boxes (area ≥ 95 %) as detections.
- Silently changing decoding between arms.

## Models

"Use the latest version of this" read as the latest version of VERA's backbone family, with one
newer comparator (my call).

| Role | Model | Thinking switch | State |
|---|---|---|---|
| Primary (latest InternVL) | InternVL3.5-8B (Aug 2025) | system prompt (R1 protocol, `<think>` tags); off = no system prompt | weights on Oz `/local/datasets/sp2117/models/OpenGVLab/InternVL3_5-8B`, needs transformers ≥ 4.52.1 |
| Newest comparator | Qwen3.5-9B (Feb 2026) | `enable_thinking` in the chat template | to download; box output undocumented, needs a probe |
| Grounding reference | Qwen3-VL-8B Instruct and Thinking (Oct 2025) | separate checkpoints | to download; documented 2D grounding |
| Current backbone | InternVL2-8B (6fb9ad69) | none | on Oz |
| Scale check (optional) | InternVL3.5-38B | system prompt | to download (~77 GB); runs sharded on the A16 pool |

Decoding (my call):

- **Thinking off:** greedy, as VERA.
- **Thinking on:** each model card's recommended sampling (e.g. InternVL3.5: T = 0.6) with fixed
  seeds and at most 4,096 new tokens, truncation logged.
- **Sampling variance:** each thinking arm is repeated with two seeds on a 100-frame subset.

## Module 1: the box detector

- **Input:** one frame at a time, with the model's own dynamic tiling (up to 6 tiles at 448 px),
  never VERA's 8-in-one packing.
- **Prompt:** one fixed referring phrase, "the firearm". It names no category, so the box doesn't
  leak the label. A second call asks for "the person" for the explanation overlay.
- **Output:** boxes and the model's own label for each, parsed with the existing tag-format probe
  (`internvl_ref_box` / `qwen_box_start` / none).

**Ablation E1 (Phase 1)**

- **Frames:** the 8 VERA-sampled frames of non-test videos (240 train + 62 validation), so the
  test split stays untouched.
  - About 1,000 frames: every positive firearm frame with an audited box, capped at 4 per video,
    plus 2 frames per No_Gun video for false positives.
  - Thinking arms use a fixed 300-frame subset of these.
- **Metrics:**
  - recall and precision at IoU 0.50 and 0.75, by category and box size;
  - the No_Gun false-positive rate (any firearm box);
  - the empty and whole-image rates;
  - parse validity;
  - seconds and tokens per frame;
  - the label agreement of the detector's own handgun / long-gun label.
- **No mAP:** none of these models gives a native confidence, so no AP is computed, as in A/G/AG.
- **Key question:** does the detector find the handguns VERA's classifier misses? It is reported
  on the handgun frames the seed questions got wrong.

## Module 2: the multi-category training loop

"Loop v2" keeps VERA's core: one frozen model is both learner and optimizer, and only the guiding
questions change. Training 1's lessons are built in (my calls):

1. **Slots owned by confusions.** The question set has a slot for each frequent confusion pair
   (Handgun↔No_Gun, Handgun↔Machine_Gun, Machine_Gun↔No_Gun) and for the action group. Each step
   rewrites only the slot of the most frequent confusion on the check set; the other slots stay.
   This stops VERA chasing the last batch.
2. **The optimizer sees errors, not just a batch.**
   - It gets the check-set confusion matrix.
   - It sees the targeted pair's 2 example videos, 16 frames in all, as in VERA. This keeps the
     input inside the 8,192-token context that broke the fixed variant.
   - It also gets the slot's earlier versions with their scores.
3. **Accept-if-not-worse** on a fixed check set of 30 training videos per category (90). That is
   bigger than the fixed variant's 24, so one video moves the score by 1.1 points, not 4.
4. **No validation-based checkpoint selection.**
   - The final question set is the last accepted one.
   - Validation is reported, not used to choose, because selecting on 62 videos was the noisiest
     step in Training 1.
5. **Three run seeds per arm by default**, with mean and spread reported.
6. **Generic.** Classes, groups and codes come from `codebook.json`, and nothing firearm-specific
   is hard-coded.

**Arms (Phase 2)**

- **L1:** loop v2; learner and optimizer with thinking off.
- **L2:** L1 with the optimizer's thinking on. Its input is text plus 16 frames, so thinking adds
  about one long call per step.
- **L3:** L1 plus the detector's crops as a second view for the learner. The learner sees the 8
  frames and up to 2 zoomed crops of detected firearm boxes, chosen by box consistency across
  frames.

## Module 3: amalgamation (boxes and importance score)

**Boxes** enter the package three ways:

1. **Zoom view:** crops of detected firearm boxes as extra learner input (arm L3).
2. **Overlay:** box overlays shown with the explanation.
3. **Box-conditioned question:** one fixed question about the detected region, "Is the object in
   the highlighted box a handgun, a long gun, or neither?"

**Importance score.** It measures how much the decision depends on each piece of evidence, by
occlusion (my call). Attention is reported only as a cheap comparison: the grounding study showed
it does not track reliance (ρ = 0.11).

- **Frame importance:** grey out one sampled frame. The importance is the drop in the log
  probability of the predicted category (8 scoring passes, no generation).
- **Box importance:** grey out a detected box, and separately an equal-size control patch in the
  same frame. The importance is the box drop minus the control drop. This reuses
  `vera_runner/reliance.py` (`mask_image`, `control_box`) from the grounding branch.
- **Checks:**
  - *Deletion:* removing the top-importance box flips the decision more often than removing a
    random equal-size region.
  - *Pointing game:* the top-importance box overlaps an annotated firearm (IoU ≥ 0.5).

## Phases and GPU schedule

The user's budget: keep 3 of the 4 RTX 6000 Ada busy, and use several of the other GPUs in
combination, up to 7. I read "the others" as Oz's A16s (15 GiB each), pooled to hold one larger
model (my reading; correct me).

**Phase 0 — set-up (½ day; CPU, plus about 1 GPU-h for probes)**

- Environments, downloads and box-format probes (one call per model, thinking on and off).
- Frame manifests and fixed subsets.
- Smoke runs of every Phase 1 job.

**Phase 1 — ablations in parallel (about 1 day)**

| GPU | Run | Content |
|---|---|---|
| Ada 4 | E1a, detector, latest InternVL | InternVL3.5-8B thinking off (1,000 frames) and on (300 frames; 2 seeds on 100); InternVL2-8B single-frame baseline |
| Ada 5 | E2, recognition ("the other runs another") | VERA classifier with the seed questions on the 302 non-test videos. Backbones: InternVL2-8B and InternVL3.5-8B (thinking off and on). Inputs: VERA packing, single-frame tiles, and the oracle crops of annotated boxes. Separates "can't see" from "can't name" |
| Ada 6 | E1b, detector comparators | Qwen3.5-9B thinking off/on; Qwen3-VL-8B Instruct and Thinking |
| A16 × 6 (Oz 0–3, 8–9) | E1c, scale check (optional) | InternVL3.5-38B sharded, thinking off on 300 frames and on for 100 |
| Ada 7 | free | — |

**Phase 2 — loop v2 and the detector at scale (about 1 day)**

- **Adas 4–6:** arms L1, L2 and L3 × 3 run seeds = 9 runs, three rounds.
- **A16 pool:** the chosen detector over all 3,184 sampled frames of all 398 videos (cached for L3
  and the package), as three 2-GPU workers.

**Phase 3 — package, importance and final test (½ day)**

- Importance on the test split.
- One test run per final configuration.
- Report, VERA Lab integration and PR.

GPU estimate: 35–45 Ada GPU-hours plus up to about 20 A16-pool hours. Throughputs are from the
smoke runs; Phase 0 re-estimates them before Phase 1 starts.

## Decision gates

Declared before results (my thresholds; the user can change them):

- **G1, detector:**
  - **Bar:** an arm is usable when recall at IoU 0.50 is at least 0.50, the whole-image rate is at
    most 10 % and the No_Gun false-positive rate is at most 10 %.
  - **Choice:** among usable arms, the highest recall, then the lower false-positive rate, then the
    faster arm.
  - **If none is usable:** boxes are reported as not solved, L3 is dropped, and the package shows
    boxes as diagnostics only.
- **G2, classifier backbone:**
  - InternVL3.5-8B replaces InternVL2-8B when its seed-question category accuracy on the 302
    non-test videos is higher.
  - The paired recording-group bootstrap must exclude zero.
  - Thinking for the learner is chosen the same way.
- **G3, final arm:**
  - **Rule:** the Phase 2 arm with the highest mean check-plus-validation category accuracy
    across its 3 seeds.
  - **Test:** all three seeds of every arm run on test once, and the report shows all of them.

## Final evaluation

On the 96 test videos, once:

- **Classification:** category, group and code accuracy against the seed questions on InternVL2-8B
  (59.4 %) and Training 1's runs. Paired recording-group bootstrap, 2,000 draws, seed 20261026.
- **Handgun accuracy:** against 15.6 % (all videos) and Training 1's per-category results.
- **Boxes:** recall and precision on the test frames with audited boxes.
- **Importance:** the deletion test and the pointing game.
- **Cost:** seconds per video for the whole package.

Success, declared now (my call):

- G1 passes;
- handgun accuracy improves with an interval that excludes zero;
- the deletion test beats random regions with an interval that excludes zero.

The category accuracy gain over 59.4 % is reported either way.

## Downloads and environments

These need the user's go-ahead. Sizes are approximate and are checked at download.

- **Qwen3.5-9B:** about 19 GB.
- **Qwen3-VL-8B-Instruct and -Thinking:** about 17 GB each.
- **InternVL3.5-38B:** about 77 GB, optional.

All of it goes to `/local/datasets/sp2117/models/` (714 GB free); nothing goes to the NFS home.

- **Environments:** one venv per transformers line under `/local/datasets/sp2117/envs/`: `vlm-new`
  for InternVL3.5 and Qwen3-VL, and `qwen35` if Qwen3.5 needs transformers from main.
- **VERA's own environment is not touched.**

## Risks

- **Undocumented box output:** InternVL3.5's box format is not on its model card, and Qwen3.5's box
  output is undocumented. Phase 0 probes both; Qwen3-VL is the documented fallback.
- **Thinking-mode sampling is not deterministic:** fixed seeds and a repeat on 100 frames bound the
  variance.
- **A16 sharding is slow** (about 200 GB/s memory per GPU): the 38B check is optional and capped
  at 300 frames.
- **Thinking traces can be long:** the 4,096-token cap, with truncation counted as invalid.
- **Training 1's lesson:** check every new prompt's token count against the context before a long
  run.

## Deliverables

- **Code on branch `experiment/firearm-package`:**
  - `vera_runner/detector.py`, `vera_runner/importance.py`;
  - loop v2 in `vera_runner/verbalized_v2.py`;
  - `run_firearm_package.py`: `probe`, `detect`, `recognise`, `train`, `importance`, `package`,
    `analyze`.
- **Configs and jobs:** tests on fakes for every module.
- **Records:** a variants-style plan record (declared before results) and DEVIATIONS.md.
- **Report:** a LaTeX report, with ablation tables E1/E2, the loop v2 comparison and the package
  evaluation.
- **VERA Lab:** boxes and importance on the timeline.
- **PR** into `feature/configurable-runner` after the package succeeds or is reported.

## Calls I made

- "Latest version" = InternVL3.5-8B, plus the newest Qwen (Qwen3.5-9B) and Qwen3-VL as the
  grounding reference.
- "The others … up to 7" = Oz's A16 pool, used mainly for the optional 38B scale check.
- The detector prompt is category-free ("the firearm") and has no confidence or mAP.
- Loop v2 design: slots owned by confusions, a 90-video check set, no validation-based selection,
  3 seeds per arm.
- The importance score is occlusion-based, not attention-based.
- Gate thresholds of 0.50 recall and 10 % for the whole-image and false-positive rates.
- The test split is untouched until Phase 3.
