# Firearm Localization and Its Effect on VERA’s Anomaly Judgments
## Implementation handoff: configurable runner only

Status: implementable study specification. Updating this handoff does not mean experiments have run. User requested this study; no VERA Lab integration, new human study, question learning or weight training is included.

## 1. Scope and hypotheses
Run the full VISILAB firearm-action dataset (398 published videos; reconcile actual inventory) on ONE RTX 6000 Ada. Include Handgun, Machine_Gun and No_Gun, both CCTV and phone footage. Preserve native resolution and report camera/resolution strata separately.

Questions:
1. What anomaly judgments does unchanged VERA produce in each dataset category?
2. How accurately can the frozen backbone localize firearms against existing boxes?
3. Does requesting localization in the same response change anomaly judgments and localization outcomes?

A change in anomaly prediction is not evidence of improvement without anomaly ground truth. Existing firearm boxes support object-localization evaluation, not anomaly accuracy or action-state accuracy.

## 2. Repository and experiment conditions
Use ONLY https://github.com/shridpant/VERA/tree/feature/configurable-runner . Previously inspected pin: eb342d9461a480e20f143d48cd5cfa0501ad076a. Check current branch status and repository instructions; record exact base and implementation commits. Create an isolated working branch/worktree. Do not merge/push without instruction.

| ID | Input and prompt | Output | Independence |
|---|---|---|---|
| A | Original eight-frame VERA input, original learner instruction and released UCF questions | Original anomaly label and explanation | Unchanged baseline |
| G | SAME eight-frame window, fixed firearm-grounding-only prompt | Boxes indexed by sampled-frame slot; raw text | Fresh call; no A response, known category, boxes or labels provided |
| AG | SAME eight-frame window, original VERA instruction/questions plus fixed grounding instruction | Boxes indexed by slot, anomaly label and explanation in one response | Fresh call; no A/G responses or ground truth provided |

G and AG must receive the same eight images to avoid confounding single-image versus multi-image context. This supersedes earlier single-image grounding suggestions. Score them on identical window-slot instances; repeated source frames across overlapping windows are repeated-context observations, not independent samples. Report per-video aggregates and source-video clustered uncertainty. No additional independent single-frame condition is required.

Implement G/AG as explicit extensions to the configurable runner. The older branch has no grounding implementation. Do not assume Lab helpers exist; any reused code must be reviewed, attributed and tested. Preserve A's backend call, prompt, parser records and artifacts.

## 3. Frozen settings and prompt contract
Frozen OpenGVLab/InternVL2-8B; verify revision 6fb9ad6924f69424e57fab2ab061d707688f0296 and model hashes. CUDA BF16, original FlashAttention setting, greedy decoding, one beam, max_new_tokens=1024 in all three conditions. No optimizer or weight updates.

Original A sampling: stride 16 frames, nominal FPS=30, centered window_seconds=10, eight RGB samples, 448x448 preprocessing and ImageNet normalization. No crops, blur, downsampling or causal-window substitution. Preserve actual FPS/timestamps: 300 frames span 12 s on 25-FPS footage. This is offline inference, not early warning.

A prompt is byte-for-byte original. G and AG prompts are fixed before full inference and saved verbatim. Grounding instruction must specify:
- Locate all visible firearms (handguns and long guns), regardless of whether activity is anomalous.
- Return each sampled frame slot with a list of xyxy boxes in the declared 0–1000 coordinate system; empty list means no firearm detected; distinct unknown state means unable to assess.
- Do not receive dataset category or annotation information.
- Emit one firearm class for primary localization evaluation; category-specific breakdown uses dataset metadata afterward.

For AG, insert the grounding instruction before the original concluding anomaly instruction, requesting boxes before the final anomaly decision. Keep the original final Output label format. Save the exact edited prompt and label this a modified-prompt condition. Check whether eight-image slot references are interpretable in InternVL2's actual input convention; never silently alter A's image packing.

Run bounded formatting/coordinate checks before locking prompts. Do not tune on full-dataset localization scores. Record any pilot examples and prompt revisions; identify the study as exploratory if pilot results influenced prompt design.

Keep the 1024-token cap equal; record truncation/EOS and generated-token counts. If AG truncates, report it as a limitation, not a normal label or empty box. Increasing the cap is a separately declared follow-up, not a silent change.

## 4. Dataset and annotation audit
Sources:
- https://data.mendeley.com/datasets/bbzpxhd22j/2
- https://zenodo.org/records/15387426
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10827673/

Inventory IDs, hashes, source paths, dimensions, timebase/FPS, decoded count, duration, category, actor/camera/lighting where documented. Nested video.mp4 filenames need stable unique staged names. Do not overwrite original files. Retain license/provenance.

Attached example inspected: label.json contains 150 image entries at 640x480 and 97 Handgun annotations, with COCO-style xywh boxes. This alone does NOT establish dataset-wide completeness or whether unannotated frames are negative.

Required audit: determine image_id-to-decoded-frame mapping (including 0/1 origin), verify box overlay alignment, check bounds/category IDs, and establish annotation coverage from documentation and representative visual checks. This is annotation validation, not a new participant study. Do not assume the first annotated image is frame zero. Convert source boxes to the actual 448x448 input coordinates and normalize consistently; test conversions with synthetic known geometry and real overlays.

For frames with unknown annotation completeness, report localization overlap/recall on known annotated objects only; do not report exhaustive precision or false-positive rates. Evaluate No_Gun false detections only where negative status is established. Report exclusions and missing annotation coverage explicitly.

## 5. Execution and durability
1. Preflight storage, environment and available RTX GPU UUID; one GPU process at a time.
2. Inventory all videos, generate immutable experiment specification and estimate expected windows=sum(ceil(frame_count/16)).
3. Real bounded A/G/AG smoke tests; validate prompt identity, frame hashes, slot parsing, coordinate mapping and output capture. Profile runtime/storage before full run.
4. Run all conditions over all windows, sequentially on one GPU. Freeze and log condition order. Prevent chat-history leakage across windows and conditions.
5. Generate aligned ImageBind features once for the original videos. Refine A and AG scores with IDENTICAL features and original refinement parameters. G has no anomaly refinement.
6. Reconcile every expected video/window/condition and build analyses/report.

Implement durable per-video/condition completion and retry ledger; older runner has no general resume. Skip work only when model/config/input hashes and artifact completeness match. Keep each failed/retried attempt. No silent substitutions. Save raw text, input token IDs, sampled indices/images/hashes, generated token counts, parsed labels/boxes, timing, memory and status.

Save strict and legacy A/AG anomaly labels. Use strict valid labels for refinement and report parser disagreements; invalid labels are missing, never normal. Original legacy parsing may be corrupted by numbers in AG boxes: isolate the final declared anomaly Output field for strict parsing and test it.

Original refinement may be undefined/problematic for short sequences. Do not pad clips or silently modify smoothing. Preserve initial results for all valid clips and explicitly report refinement exclusions. Complete coverage means every scheduled unit has a documented disposition, not that every generation succeeded.

## 6. Localization metrics
Compare G versus AG on identical evaluable window-slot instances, including those A labels normal. Predicted boxes map to sampled-frame slots. For each image, deterministic one-to-one matching maximizing eligible IoU at thresholds 0.50 and 0.75; document matching implementation and tie breaking. Duplicate unmatched boxes count as false positives where annotation completeness permits. Report TP/FP/FN, precision, recall and F1, plus matched-box IoU separately from missed objects.

Do not report standard confidence-ranked mAP without meaningful ranked detection scores; this protocol provides fixed-output operating-point metrics. Do not invent confidence scores. Report malformed/unknown outputs separately and count them as failures to localize annotated objects in end-to-end recall. Separate whole-image boxes (>=95% area) as diagnostic outputs while retaining their geometric effect in the declared primary metric. Report localization performance by weapon size where sample sizes allow.

## 7. Required report tables (all explicit)
T1 Dataset composition: category; videos; duration; frames; scheduled windows; camera/resolution/FPS breakdown.
T2 Coverage: condition x category; attempted/completed videos/windows; valid labels/boxes; malformed outputs; unknowns; truncations; timeouts; refinement exclusions.
T3 Original VERA behavior: category; valid A windows; anomaly count/%; no-anomaly count/%; invalid count; mean and median per-video anomaly fraction.
T4 A versus AG: category; paired-valid windows; anomaly % for each on identical paired support; percentage-point difference; video-cluster bootstrap CI; separate full-coverage summaries.
T5 Prediction transitions: category; normal-normal; normal-anomaly; anomaly-normal; anomaly-anomaly counts/%; unparsed pairs separately.
T6 Refined scores: condition x category; eligible videos; per-video mean-score distribution (mean, median, IQR); median maximum score; paired A–AG differences on common eligible videos.
T7 Grounding outcomes: G/AG x category; submitted slots; localized boxes; whole-image boxes; empty output; unknown; malformed; raw annotation coverage.
T8 Localization: G/AG x category and IoU threshold; evaluable slots/GT objects; TP/FP/FN; precision/recall/F1 where justified; matched IoU; paired difference uncertainty; N/A with reason when labels insufficient.
T9 Per-video results appendix/CSV: ID/category/camera/duration; expected/completed windows; A/AG label counts/fractions; refined summaries; G/AG localization metrics; artifact links.
T10 Resources: condition/stage; units/calls; total time; median/p95 latency; generated tokens; truncation count; peak GPU memory. Feature extraction separate.
T11 Qualitative cases: ID/time/category; actual A/AG text/labels; G/AG boxes and GT overlays; selection rule; observed differences and limitations.

Percent anomaly uses valid windows; invalid rate uses all scheduled windows. Report pooled-window and equal-video summaries distinctly. Existing categories are NOT anomaly truth; T3 is not a confusion matrix. No anomaly accuracy/AUC/AP or action-state accuracy without valid corresponding labels.

## 8. Required plots
P1 A/AG stacked prediction bars by category, including invalid outcomes.
P2 Paired per-video anomaly fractions and AG-minus-A difference distributions.
P3 Prediction-transition heatmaps by category.
P4 Per-video refined-score distributions and paired changes.
P5 Category x camera/resolution panels with sample counts; descriptive associations, not causal resolution claims.
P6 G/AG box-output status bars.
P7 G/AG localization precision/recall/F1 at IoU .50/.75, matched-IoU distribution and size breakdown where eligible.
P8 Representative A/AG timelines with identical sampled frames, explanations, generated boxes and GT overlays.
P9 Runtime/token distributions by condition.

Predefine qualitative selection: largest positive/negative anomaly-fraction changes, typical unchanged cases, and high/low localization outcomes; report rule and IDs. Do not select only successes.

Use paired video-cluster bootstrap confidence intervals with a fixed recorded seed. Where recordings share actors/takes, report dependence limitations and cluster at recording group if reliably identified. Overlapping windows are not independent replicates.

## 9. Deliverables and reproducibility
Deliver exact experiment_A_G_AG.annotated.yaml with comments explaining every knob, validated conversion to runner-compatible JSON, and immutable resolved configs. Include tested commands for prepare, validate, smoke, run/resume, extract features, refine, analyze, reproduce plots and compile report. New commands/scripts must actually be implemented and tested; placeholders are not completion.

Knob guide: prompt and output order, frame count/stride/window/nominal FPS, resolution, precision/attention, generation cap, box coordinate scheme/matching thresholds, refinement, timeouts, GPU selection, seeds. Mark scientific changes versus operational controls.

Overleaf-ready project: main.tex, bibliography, all generated tables/figures, methods, results, limitations, appendix and compiled PDF; source ZIP and reproducible build commands. Populate results only from saved artifacts. No claim of live Overleaf upload unless performed.

Report title: Firearm Localization and Its Effect on VERA’s Anomaly Judgments.
Methods must explain A/G/AG, exact prompts/packing, native timing, full inclusion, existing-label audit, strict parsing, paired metrics, annotation exclusions, offline refinement and computational controls. Discuss AG as a bundled prompt change: changes cannot prove internal localization caused improved reasoning. Future work: human judgments, carrying/raising/pointing labels, VERA question learning and causal early-warning evaluation.

## 10. Review and acceptance
Independent reviewer checks scope, coordinate/frame validity, information leakage, denominator consistency, identical A/G/AG inputs and grounding parser robustness before full execution. AI review is not a human signature.

Completion: every inventory item and scheduled condition accounted for; original A preserved; valid outputs and failures traceable; all T1–T11 and P1–P9 produced or marked not estimable with specific evidence; no invented ground truth/metrics; report compiles; exact configurations and commands reproduce analysis. No UI integration, extra GPUs, prompt optimization, new human annotation collection or additional experiments under this handoff.
