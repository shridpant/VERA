# Frozen CPU-only temporal-position audit, v1

Authorized 2026-10-27. Commit and push this protocol before downloading/processing full features or measuring performance. This is a retrospective score-stage audit on consulted UCF test data, not a new detector or untouched evaluation. Removing position weighting is already a VERA Appendix B.4/Table17 ablation. Novelty is conditional on a practically meaningful controlled robustness finding; no algorithm novelty is claimed.

## Immutable inputs and no-GPU boundary

- Upstream source checkout `008ed4019781c39e7cb16244b7500244f3986f92`, unchanged `compute_auc.py` SHA256 `21e9705da3e8b04deb5855665a0c8fbc41e6baa705374f2c362b814d24b5eda8`.
- `Data/UCF_Eval.json` SHA256 `f3857cddd623f767e7b86bd50a4aeb122445fc6e241ddd7130b7e3ec0c4d8b9f`.
- Author initial-score ZIP SHA256 `b7457cee812510ec2331c24cb5da451327c9500152e2437839d31af29166a3b0`: 290 IDs and69,634 anchors, exactly `range(0,length,16)`.
- Authors' feature Drive ID `1gwkgIKiPg6fpRaO5yXkxuRcN3X1-97LF`, `ucf_crime_vision_features.zip`, expected size309,807,835bytes. Record cryptographic digest after download, ZIP CRCs, all member names and feature hashes/shapes/dtypes/finite values/positive norms. Require all290 IDs and matching feature rows in score insertion order. Match RoadAccidents127 feature to existing verified SHA256 `f0c2470ccdae5e6d8091dd3e41fe281f02243f37b80bfbf71fd0e0ed668eb657`. Whole-archive hash is integrity/provenance, not an independently published authenticity proof. On missing/misaligned features, stop; never substitute GPU extraction.
- No CUDA, PyTorch/model imports, inference, verbalized learning, question changes or weight training. Run locally on CPU, set CUDA_VISIBLE_DEVICES empty and BLAS thread limit4. No Oz GPU/CPU experiment jobs. Oz used only to read/copy already pinned assets. Do not interfere with firearm jobs.
- Download fallback allowed: Drive connector has256MiB limit; use public release directly. Do not store bearer/signed download URLs in artifacts. Use safe ZIP extraction with no traversal/symlinks.

## Reconstruction gate

Reproduce released upstream scene refinement, smoothing, Python round-to-one-decimal and last-segment expansion exactly. Preserve anchor insertion order. Primary retrieval ratio is0.15 and similarity softmax is exp(similarity*10), as implemented in released code; paper describes UCF ratio0.10, which is a separately reported sensitivity, never selected based on results. Kernel size15/sigma10, original NumPy convolution, final center/sigma=floor(expanded frame count/2). Do not silently normalize or repair code quirks.

Run the unmodified upstream script with only filesystem path indirection in a temporary staging directory and capture final vectors. Independent replay must match the full weighted vector within absolute1e-10/relative1e-10 and label vector exactly. Record differences from the paper's86.55% without tuning to match it. Freeze implementation hashes before this gate/performance run; any fix gets a separate documented amendment and preserves prior outputs.

## Labels and sensitivity

Use upstream half-open [start,end) labels, clipping ends to metadata length exactly as its code does. Preserve originals. Known exceeded endpoints: Arson011, Arson016, Explosion033, Fighting003, Shooting015 (1–2frames). Primary all290; repeat excluding these five. No semantic reannotation. Constant-score ROC-AUC must be0.5 when both classes exist; normal-only per-video AUC is undefined, not zero. Report tie-aware pooled ROC-AUC and average precision (AP, not trapezoidal PR area).

## Fixed counterfactuals and endpoints

Save each video's pre-position frame scores r and label vector y. For fractions f=(0,.2,.4,.6,.8), use k=floor(f*F) and circularly roll r,y together. All score/label relationships, scene refinement and already-smoothed context remain fixed. Compare:

1. Original final weighting: Gaussian w_F times rolled r, against rolled y.
2. No-position control: rolled r against rolled y.
3. Prior-only diagnostic: w_F alone against rolled y.

This isolates the last multiplier; it is not a physical-video edit, new window sampling or end-to-end temporal-model robustness. No-position invariance is expected mathematically, not a novel result.

Primary pooled frame ROC-AUC: report all five shifts, spread(max−min) for original, worst-position benefit min(AUC_no_position)−min(AUC_original), and unshifted cost AUC_original(0)−AUC_no_position(0). Secondary pooled AP, per-video/category summaries and label-position distribution. Category summaries must include negative frames; undefined single-class cases explicitly marked. No threshold tuning or method selection.

Paired video bootstrap:1000 resamples of290 whole videos with replacement, NumPy default_rng seed20261027, percentile95% interval. Use identical multiplicities for methods/shifts. Recompute min across fixed shifts within each replicate, not a post-hoc selected shift. Weighted frame statistics must reproduce explicit duplicated-video calculations in a small correctness test. For five-video exclusion use the same seed and1000 resamples of285 videos. If computation is too slow, preserve partial results and document an amendment before altering bootstrap count or precision; do not silently approximate.

## Agreed pass/fail gate

All required:
1. Original pooled AUC spread >=3percentage points across fixed shifts.
2. No-position worst-shift benefit >=3points and paired95% CI lower bound>0.
3. Unshifted AUC cost <=1.5points; the five-endpoint exclusion must not reverse the pass/fail conclusion.

Criteria are unchanged from the recommendation. Report0.10 retrieval sensitivity separately; contradictory results limit generality, not license to pick a winner. If gate fails, stop this direction and report the negative audit. If it passes, this is still provisional: independent XD/fresh confirmation and broader novelty review are needed. No automatic XD download/experiment or GPU allocation.

## Reproducibility and reporting

Separate branch `research/vera-position-audit`, frozen protocol/implementation commits, immutable asset manifests, raw reference/replay vectors stored outside Git with hashes, portable results/validation/figures in Git, compiled .tex and a draft PR. Preserve PR11 experiments. Record CPU wall time, software/BLAS environment, resources and all deviations. Never merge PRs. Initial CPU budget30minutes for scene reconstruction and30minutes for statistics; any timeout preserves logs and requires a documented resource-only amendment, never GPU fallback.
