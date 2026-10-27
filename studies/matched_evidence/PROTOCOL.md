# Controlled corroboration experiment for video anomaly detection

Version 0.1, October 27, 2026. Status: executable mathematical core and prospective implementation specification. No new model inference or detection result exists. Freeze the completed inference implementation and run manifest before measuring new performance.

## Hypothesis and scope

An anomaly score can receive additional support when a region selected by its effect on the model remains influential on different sampled frames and under two corruption operators. The selected region should both preserve more anomaly support and remove more anomaly support than equal-area alternatives. This tests model dependence, not semantic truth or real-world causality.

The main candidate is called matched evidence corroboration, abbreviated MEC in filenames. REV-VAD is retained only in the Git branch name for continuity. The original temporal-removal formula is retired.

The detector is training-free only in the primary fixed-prompt condition. A secondary condition may reuse VERA's learned questions, but must be described as a VERA extension using previously learned prompts. Neither condition changes model weights.

## Fixed inputs and numerical readout

Use InternVL2-8B revision 6fb9ad6924f69424e57fab2ab061d707688f0296, BF16, eval mode, with the existing verified preprocessing. All changes live in studies/matched_evidence. Do not modify the old engine or its contracts in place.

The primary prompt is fixed before inference: "Judge only the visible video content. Is an abnormal event occurring, such as physical harm, dangerous loss of control, damage, or unauthorized interference? Do not infer sound or hidden intent. Reply with 1 for an abnormal event or 0 otherwise."

Use the conditional support score s = sigmoid(logit_1 - logit_0) at the first answer position. Validate the exact tokenizer boundary and candidate tokens. This is a normalized two-candidate support score, not a calibrated event probability. The label-only condition is distinct from VERA's explanation-first generation. Keep prompt wording identical between all transformed views. Save native logits and the existing FP32 final-head-row readout; reproduce their agreement/rounding checks before selecting a ranking readout. Proposed main readout: FP32 final-head-row difference, matching the prior engineering solution for BF16 ties.

For UCF, match VERA's anchor grid range(0, metadata_length, 16) and its clipped [a-150,a+150) window. Decode by frame index. Preserve the hard-coded 30-fps baseline convention for comparability and report actual-fps discrepancies; do not silently change the baseline.

Sampling A uses VERA's original eight indices. Sampling B selects the midpoint within each of the eight equal temporal bins. Require eight unique, in-range B indices disjoint from A; if unavailable, mark unverified and retain the original score. A and B are correlated observations from the same event, not independent statistical replications. B can miss brief events; report this failure mode.

## Score first and select work without labels

Score EVERY A window before any evidence ranking. Persist scores with model, prompt, source, preprocessing, and frame hashes. Neither annotations nor a VERA reference score may select MEC windows.

Within each video, select at most ceil(0.10 times number_of_anchors) windows with s >= 0.5, sorted by descending score and ascending anchor on ties. No event annotations enter this decision. All unselected scores remain available and unchanged. Also inspect a deterministic hash-selected 1 percent of unselected windows as a missed-opportunity diagnostic; these diagnostic interventions never alter the primary detector. Record selection coverage and overlap across windows. This is a fixed compute allocation, not a claim that low-scoring anomalies do not exist.

## Select a spatial tube

Partition the 448 by 448 model image into four 224 by 224 quadrants. A tube is the same quadrant across all eight frames. This preserves frame count, timestamps, and order, and can remove a persistent localized cue across the whole window. It does not track moving objects.

On sampling A, blur each quadrant separately in four scorer calls. Choose E as the quadrant whose removal produces the largest SIGNED decrease from the original A score. Tie break in row-major order. If all decreases are nonpositive, retain the score and record unresolved evidence. Never call the lower-slot tie break a confident explanation.

The other three quadrants are the controls C. Their area, shape, frame count, operator, and position in time are matched. Their visual content and spatial position are not exchangeable. These are controlled contrasts, not randomized causal estimates or p-values. Using all three controls avoids choosing a convenient background region after seeing verification results.

## Confirm on sampling B

Use two separately recorded operators after the normal geometric preprocessing, before normalization:

* blur: replace masked pixels with the corresponding pixels of the whole image blurred with PIL GaussianBlur radius 17.92;
* mean: replace masked pixels with that frame's original global RGB mean, rounded once to uint8.

Both operators are deterministic. Mean fill is different from the old constant-grey experiment. Do not reuse its cache entries. These are deliberately distinct perturbations, but both remain artificial; agreement does not prove an in-distribution intervention.

For every quadrant Q in {E,C1,C2,C3} and operator f, compute:

* deletion D_f(Q): replace Q across all B frames;
* retention R_f(Q): replace the complement of Q across all B frames.

Retention leaves coarse context in the blur condition; the mean condition is a stronger destruction of context. Each selected and control retention changes exactly 75 percent of the pixels, and each deletion changes exactly 25 percent. All views retain eight frames and the same input-token geometry.

Let q_f_D(Q) and q_f_R(Q) be their support scores. Define

    deletion_contrast_f = median_C q_f_D(C) - q_f_D(E)
    retention_contrast_f = q_f_R(E) - median_C q_f_R(C)
    b = max(0, min(deletion_contrast_blur, deletion_contrast_mean,
                   retention_contrast_blur, retention_contrast_mean))
    s_MEC = s + (1-s)*b

Use full-precision signed contrasts until the final max. Do not select E using B results or switch the operator after inspection. B unmodified is scored once as a shift diagnostic, even though its value cancels from the deletion contrast. If abs(s_A-s_B) > 0.20, any input check fails, or any required value is missing/nonfinite, return s_A and record the reason. The 0.20 value is a proposed fixed stability guard, not a significance threshold; freeze it now and report an ablation without it. No tuning on UCF test outcomes.

This rule has no learned fusion coefficient. The weakest of the four contrasts limits corroboration. It guarantees s <= s_MEC <= 1 for valid inputs and leaves inconclusive cases unchanged. It can still boost wrong predictions and worsen ranking. At a fixed numeric threshold it cannot reduce false positives; a possible reduction is only at a matched recall/TPR after reranking. It cannot recover low-score windows outside the verification allocation.

## Weaknesses and engineering responses

| Weakness | Response and remaining limit |
| --- | --- |
| Persistent anomaly survives removal of two frames | Apply a spatial tube across every sampled frame; distributed events still use the original score |
| Redundant evidence makes necessity small | No negative inference from a failed deletion: b becomes zero and score remains unchanged |
| Corruption itself changes a prediction | Equal-area controls, two operators, input hashes, and unmodified B check; effects remain intervention-specific |
| Selection finds a lucky sensitive patch | Select on A and confirm on disjoint B frames; correlated frames still allow selection bias |
| A static contextual cue masquerades as an event | Require both retention and deletion advantages; semantic false support can survive, so benchmark and source-grounded audits are essential |
| Coarse cells split actors or miss movement | Report unresolved evidence and fallback; do not add a detector/tracker or optimize masks in v0.1 |
| Event covers most of the image | Corroboration may vanish; fallback preserves the base score |
| More calls alone improve performance | Equal-call averaging and random-region corroboration controls |
| A score explanation is presented as explanation of final VERA output | Keep scorer targets separate; trace all downstream refinement weights for any refined timeline |

## Pipeline and immutable records

Stages: inventory -> source/tokenizer/preprocessing verification -> CPU fixtures -> reference VERA replay -> engineering smoke -> score all windows -> freeze selection -> interventions -> reranking -> dense frame expansion -> metrics -> XAI diagnostics -> validation -> report.

The inference CLI to implement must have separate plan, score, select, verify, evaluate, and report commands, plus resume and dry-run. Planning must print both unique calls and cache hits. Content-address the requests using model revision, precision/readout, prompt bytes, source version, frame indices and pixel hashes, mask geometry, operator, and mode. Masking changes neither slot count nor tokens. Use a separate output root and lock per run. Write raw logits, scores, contrasts, selected/control masks, signed effects, selection reason, fallback reason, model input hashes, latency, and GPU memory for every record. Preserve failed attempts. Never mix old VERA score caches with the new prompt.

New numerical forwards per verified window: four A ranking probes + one B baseline + sixteen B verification views = 21. The dense baseline costs N forwards; primary verification costs at most 21 times sum_v ceil(0.10 N_v), less early exits. Diagnostic samples and equal-compute controls are extra and must be counted. N is expected to be 69,634 on the known UCF metadata, but recount it. At the historical 0.356 seconds per pass, dense baseline plus 10 percent verification is approximately 21.35 GPU-hours before IO, model loading, diagnostics, and controls. This is not a fresh timing measurement; benchmark the actual implementation before launching a full run. A16 BF16 capacity is insufficient for the pinned 8B setup. Use a currently free Ada by UUID; never assume previously idle hardware remains available.

## Development and evaluation

UCF test data have already been consulted by this project. Do not call them pristine held-out confirmation. Use old designated development videos for engineering only and exclude them from any confirmation sensitivity. Do not choose formulas or masks using new test metrics. If a substantial redesign follows observed UCF performance, label it exploratory and require a frozen transfer to XD for confirmation.

Engineering smoke: two old development windows plus 16 windows chosen by ID hash from the old sparse-input manifest, without looking at their scores for selection. Include correct input transformations, resume equivalence, exact readout checks, call counts, missing-data fallback, and a case with no positive ranking effect. This is a mechanics check, not a statistical benefit gate. Do not invent a 3-point stopping target from the unrelated position audit. Correct concrete software defects with preserved failed runs and versioned amendments. Scientific failure remains failure.

Primary detector comparison on all 290 videos:

1. Released VERA, exact pinned upstream replay, both paper retrieval setting 10 percent and released-code setting 15 percent, labeled separately.
2. Fixed-prompt direct VLM support scores, with the exact MEC sampling/grid and frame expansion.
3. MEC, same base support scores and same frame expansion.
4. Equal-compute ensemble of fresh full-frame temporal samples, using the same number of scorer calls as MEC per video. Predefine offsets and averaging before inference.
5. Same corroboration computation with the evidence quadrant chosen by seeded ID hash instead of A scores.
6. Ablations: deletion only; retention only; one operator; selection/evaluation using identical frames; no stability guard. Use the same cached views where identities match, and report differing compute when they do not.

The primary causal attribution of an algorithm change is MEC versus its direct baseline and compute controls, not merely MEC versus VERA. A VERA-conditioned MEC extension is secondary, with its own baseline and declared learned prompts. CoReVAD is a contextual baseline: reproduce its native pipeline separately; use a grid-matched variant only with an explicit label. Published numbers alone are reference rows, never a claim of paired reproduction.

Primary UCF endpoint: paired difference in pooled frame ROC-AUC, MEC minus direct baseline. Report VERA differences, AP, macro within-video AUC for videos containing both classes, anomaly-only pooled AUC, normal-video alarm burden at thresholds fixed before evaluation or through declared curve analysis, and intervention coverage. Use average precision, not trapezoidal PR area. Cluster bootstrap 2,000 resamples by source video, seed 20261027, paired across all methods. Do not bootstrap overlapping windows as independent. Single-class bootstrap replicates are undefined; count them and preserve the rule. Multiple ablations are exploratory; show all outcomes, including negative ones.

For dense expansion, assign each anchor score to [anchor, next_anchor), ending at metadata_length. Main direct/MEC runs use no position weighting or smoothing, so the intervention is isolated. A secondary identical VERA refinement adapter can be applied to both, with retrieval ratio, rounding, smoothing, and position prior separately logged. UCF paper comparisons with different postprocessing are not controlled method comparisons. Never round-trip through one flat score vector without explicit video IDs and frame indices.

UCF official endpoint convention and temporal annotations must match the pinned baseline, including its half-open intervals and five endpoint clips. Compare all 290 IDs and frame counts, and report the existing exclude-five sensitivity. Normal-only per-video AUC is undefined. Missing/corrupt videos must block a claimed full-benchmark result, not disappear silently.

## Explanation evaluation

Show the selected quadrant through time, controls, all four signed contrasts, the resulting boost, and any unresolved flag. This explains the proposed scorer and correction. Do not describe quadrants as object-localization ground truth or use fluent descriptions as evidence of accuracy.

Quantitative XAI evaluation must not merely reuse b as its success metric. Freeze an unseen third operator, pixelation with a 16 by 16 downsample/nearest-neighbor upsample at 448 resolution, and evaluate selected versus control masks without allowing these outputs to change the detector. Report original VERA generated-label sensitivity as a separate-target diagnostic, not an explanation of identical semantics. Report repeatability, A/B stability, operator-transfer effects, coverage, and latency. Spatial semantic correctness requires independent annotations or a blinded human study; UCF's temporal labels alone cannot validate a spatial attribution. A later human audit can use 60 windows selected by a frozen stratified hash, with reviewers blinded to the method and annotation-negative windows treated cautiously.

Explain refined timelines by recording which source scores and weights contribute at each target. A local mask explains the local score, not automatically a retrieved-and-smoothed final frame score.

## XD confirmation

Freeze the UCF version before XD access to new performance. Confirm original split IDs and source-related movie/YouTube groups; bootstrap source groups where feasible. Reuse prompt, mask bank, operators, stability threshold, and 10 percent allocation without retuning. Report official frame AP primarily and ROC-AUC secondarily. The released VERA XD AP and AUC scripts differ in position weighting; reproduce each separately. Existing local score ZIP coverage does not imply XD raw-video availability. Verify acquisition paths, licenses, decoding, and frame-label alignment before making an execution claim.

## Decision rules

Continue from engineering smoke to the complete frozen UCF run if mechanics and resource preflight pass. Do not select a successful subset after seeing labels. A positive detection claim requires a positive paired effect with uncertainty, no hidden exclusions, and comparison against equal-compute controls. A result that improves pooled AUC but harms within-video ranking must be described as a tradeoff. Claim that evidence selection matters only if it outperforms the random-region control. Claim usefulness of both tests/operators only if the corresponding ablations support it. No observed benefit is required for honest completion: a complete negative report is a valid endpoint.
