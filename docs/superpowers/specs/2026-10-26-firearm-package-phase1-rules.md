# Firearm package: Phase 1 decision rules

Declared 2026-10-26 17:30 ET, before any Phase 1 result. Implements the gates of the
[package plan](2026-10-26-firearm-package-design.md) in `studies/firearm_package/analysis_phase1.py`.
Nothing here reads the test split.

## Contents

- [E1 and gate G1 (detector)](#e1-and-gate-g1-detector)
- [E2 and gate G2 (recognition)](#e2-and-gate-g2-recognition)
- [Probes](#probes)

## E1 and gate G1 (detector)

- **Frames:** about 1,000 E1 frames from the 302 non-test videos (`run_firearm_package.py manifest`).
  - Firearm videos give up to 4 of their 8 sampled frames with an audited box.
  - No_Gun videos give 2 frames.
- **Subsets:** subset300 and subset100 are seeded, stratified by category, and nested.
- **Metrics per arm:**
  - **recall@0.50:** matched annotated boxes / annotated boxes, on firearm frames.
  - **precision@0.50:** matched boxes / all predicted boxes on firearm frames; whole-image boxes count as wrong.
  - **No_Gun false positives:** the share of No_Gun frames with any box.
  - **whole-image rate:** boxes covering ≥ 95 % of the frame / all boxes.
  - **Intervals:** recording-group bootstrap.
- **Usable arm:** recall@0.50 ≥ 0.50, whole-image rate ≤ 10 % and No_Gun false positives ≤ 10 %.
- **Ranking:** usable arms are ranked on subset300, the frames every main arm ran, by:
  1. recall@0.50;
  2. then the lower No_Gun false-positive rate;
  3. then the faster median call.

  The top arm is the package detector.
- **Excluded from ranking:**
  - `_r2` arms repeat a thinking arm with another seed, to measure sampling variance.
  - `internvl35_38b_on` covers subset100 only; it is reported, not ranked.
- **If no arm is usable:** boxes are reported as not solved, arm L3 of Phase 2 is dropped, and the
  package shows boxes only as diagnostics.

## E2 and gate G2 (recognition)

- **Setup:** the seed questions on the 302 non-test videos.
- **Intervals:** paired recording-group bootstrap on category accuracy; "beats" means the interval
  of the difference lies above zero.
- **Backbone:** InternVL3.5-8B replaces InternVL2-8B for Phase 2 only if `iv35_packed` beats
  `iv2_packed`. Both use VERA's own packed input, so only the backbone differs.
- **Learner input:**
  - on the chosen backbone, the better of `frames` and `tiles` replaces `packed` only if it beats
    `packed`;
  - otherwise Phase 2 keeps VERA's packed input.
- **Learner thinking:** on in Phase 2 only if `iv35_think_frames` beats `iv35_frames`. Thinking
  stays off when the backbone is InternVL2.
- **Oracle crops (`*_crops`):** they use the true boxes, so they are an upper bound. They show whether
  seeing the gun up close would let the model name it. They are not a candidate input.
- **Sanity check:** `iv2_packed` on the 62 validation videos should match Training 1's step-0
  validation answers. Agreement is reported; batch size 1 here against 2 there can change a few
  answers.

## Probes

- **What runs:** each model and thinking setting runs on 6 manifest frames (2 Handgun, 2 Machine_Gun,
  2 No_Gun) before Phase 1. InternVL also gets the model card's referring prompt ("the firearm") as a
  second candidate.
- **What the probes may change:**
  - the coordinate reading: a model that writes coordinates above 1,000 uses pixels, so its parser
    gets a scale;
  - InternVL's prompt: the detection prompt is replaced by the referring prompt only if InternVL
    returns no parseable box on any firearm probe frame with it.

  Any such change is recorded in `studies/firearm_package/DEVIATIONS.md` before Phase 1 starts.
