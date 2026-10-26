# ImageBind feature validation

The gate for `run_firearm_study.py features`: our extractor (LAVAD's procedure on the official
ImageBind package, `vera_runner/imagebind_features.py`) against the features VERA released for
UCF-Crime RoadAccidents127, the one video experiment01 used. The evidence is on Oz in
`/local/datasets/sp2117/firearm_visilab/features_check/`.

## Contents

- [Result](#result)
- [Diagnosis](#diagnosis)
- [Decision](#decision)

## Result

`features-check`, job `firearm_features_check_20261025T175401`, GPU-6a490dc2 (FP32):

| Check | Value |
|---|---|
| shape (ours, released) | 162 × 1024 both |
| row cosine to released: min / median / mean | 0.9545 / 0.9949 / 0.9907 |
| row cosine quartiles 25 % / 75 % / max | 0.9865 / 0.9991 / 0.9996 |
| median row L2 norm (ours, released) | 0.97613, 0.97605 |
| frame ROC-AUC, experiment01 labels + refinement, released features | 0.99494 (= experiment01) |
| frame ROC-AUC, same labels, our features | 0.99663 |
| refined segments whose 0.1-rounded score differs | 11 of 162 (max temporal difference 0.063) |

The planned gate had two parts. The median cosine of at least 0.99 passed. Identical refined
frame scores did not.

## Diagnosis

- **Alignment.** Shifting our rows by −2..+2 against the released rows gives mean cosine
  0.9874, 0.9893, **0.9907** (shift 0), 0.9826, 0.9768. There is no anchor offset.
- **Decoder** (`diagnostics/decode_diag.py`). PyAV and OpenCV decode identical pixels (mean
  absolute difference 0.0 on 27 sampled frames). Embeddings from either are equal (cosine 1.0),
  so our extraction is deterministic, and the gap does not come from our decoder.
- **Subsampling** (`diagnostics/samples_diag.py`, 21 rows including the lowest).
  - LAVAD's N = 10 (median cosine 0.985 on these hard rows) and N = 32 give identical
    embeddings. Any subsample shorter than 2 s becomes five identical 2-frame clips (its first
    and last frame).
  - Using every clip frame (five clips over 10 s) is worse: median 0.977.
  - LAVAD's N = 10 is the closest procedure tested.
- **Where the gap is.** The low-cosine rows (e.g. 128–135, 13, 59, 72) are high-motion
  stretches, where a different source frame matters most. The most likely cause is the
  authors' frame extraction environment (an FFmpeg/OpenCV build that decodes some frames
  differently) or library numerics. Neither can be checked without their frames.

## Decision

The features are used:

- The procedure matches (alignment, norms, upper-quartile cosine 0.999).
- The refined AUC differs from the released features' by +0.0017 on the validation video.
- A and AG are refined with the identical feature file of each video, so the gap cannot bias
  the A-versus-AG comparison. It affects only absolute refined scores, which the report labels
  as computed with re-extracted features.
- This deviation is stated in the report's methods and limitations.
