# Annotation audit (existing VISILAB boxes)

Annotation validation only; no new labels were collected. Evidence is on Oz in
`/local/datasets/sp2117/firearm_visilab/audit/` (`validation.json`, `coverage.json`,
`montages.json`, montage PNGs), made by `run_firearm_study.py audit` with seed 20261025.
The decisions are recorded as data in [annotation_policy.json](annotation_policy.json).

## Contents

- [Inventory](#inventory)
- [Label files](#label-files)
- [Frame mapping](#frame-mapping)
- [Coverage and completeness](#coverage-and-completeness)
- [Coordinates](#coordinates)
- [Decisions](#decisions)

## Inventory

398 videos: Handgun 141, Machine_Gun 139, No_Gun 118, as the data paper states. C1 (IP camera,
640×480, 25 FPS) 175 videos; C2 (phone, 1920×1080, 30 or 29.92 FPS) 223. Full OpenCV decodes
equal the container frame counts for all 398. Scheduled windows: 6,375. All brightness codes
are HB. The cameras were at disjoint locations (C1 at P1/P2/P4, C2 at P3/P5), so no recording
was filmed by both cameras. Repeated takes of one action by one subject at one place form a
`recording_group` (used only for a sensitivity bootstrap).

## Label files

- The 280 firearm videos each have one COCO file; No_Gun has none, by design.
- `Handgun/PCH5_C1_P1_V1_HB_1` has only `label_antiguo.json` ("old label"). It validates
  clean and its boxes sit on the handgun in 3 montages. It is used and flagged.
- All 280 files pass validation:
  - image ids are exactly 1..N, where N is the decoded frame count;
  - image sizes equal the video size;
  - the category is the folder's;
  - every box lies in bounds and is non-empty;
  - no image has more than one box.

## Frame mapping

- Every file lists one image per decoded frame (ids 1..N), each named `video.mp4`.
- Origin 1 (image_id k is zero-based decoded frame k−1, MATLAB Video Labeler's convention) is
  the only bijection onto frames 0..N−1. Under origin 0, frame 0 has no entry and image N has
  no frame.
- In 39 montages (13 videos: 2–3 per category × camera, plus the label_antiguo video), the box
  of image_id k on frames k−2, k−1 and k sits on the firearm in all three. The boxes are
  interpolated and loose, so a one-frame offset cannot be settled by eye.
- Decision: origin 1 is primary. Localization metrics are repeated under origin 0 as a
  sensitivity analysis.

## Coverage and completeness

- Boxed frames: Handgun mean 58 %, Machine_Gun 61 % of frames.
  - Every first and last box lies strictly inside the video.
  - 28 videos have more than one boxed run.
- Listed but unboxed frames, 48 sampled from 16 videos:
  - Most show an empty scene.
  - Several show the subject far away, or partly visible at an image edge, with the firearm
    possibly visible and no box (e.g. PCW5_C2_P5_V3_HB_1 id 274, PAH4_C2_P5_V2_HB_1 id 241,
    PAH4_C1_P2_V1_HB_2 id 33).
  - These frames are **not** established negatives: no recall and no false-positive
    evaluation.
- Boxed frames:
  - The paper documents one individual per video.
  - Every file has at most one box per image.
  - The boxes sit on the firearm.
  - The box is treated as the complete firearm ground truth of that frame (recall and
    precision). This is an assumption and is stated as such.
- No_Gun, 24 frames sampled from 12 videos: no firearm visible. Phones, bottles and a dark
  laptop-like object are present (hard negatives). With the paper's class definition, No_Gun
  frames are established negatives: false positives are evaluated, recall is undefined.

## Coordinates

- Native xywh → [x1, y1, x2, y2] on the model's 0–1000 grid of the 448×448 input:
  divide by the native width/height and multiply by 1000.
- Tested by drawing red boxes on synthetic 640×480 and 1920×1080 frames and resizing them
  with the runner's exact PIL call. The bounds agree within 1.5 input pixels. IoU is
  invariant under the per-axis resize.
- Ground-truth boxes are small: median area 0.64 % of the image, 90th percentile 3.0 %.
- Size bins are the GT-area tertiles, 0.336 % and 1.097 % (GT only, fixed before any model
  output).

## Decisions

| Frame status | Recall | False positives / precision |
|---|---|---|
| positive (boxed) | yes | yes (single-firearm assumption) |
| listed_no_box | no | no |
| negative_category (No_Gun) | n/a | yes |
| unknown | no | no |
