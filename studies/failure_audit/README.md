# VERA failure audit: evidence and verification

The report supports task-specific limitations, not the proposition that VERA only works on UCF-Crime. A completed, target-retrained **binary anomaly** failure was not located. See `search_scope.json` for the bounded search and candidate exclusions. This missing link must not be replaced with category labels or anecdotal outputs.

## What was verified 1:1

`verification.json`: **741 checks passed, zero failed**. `metric_ledger.csv` exposes actual and expected values at full precision. Equality of counts is exact; floating-point reproduction uses an absolute/relative tolerance of 1e-12. Percentages in the report are rounded for display. Raw response parsing and per-record checks cover many more records than the top-level check count.

- SHA-256 and byte-preserving transfer of every input in both remote manifests.
- 240/62/96 split membership and recording-group separation.
- Every test target against the split's category/group/code; strict reparsing of seed and learned raw answers.
- All nine selected checkpoints from the recorded validation rule; each validation category accuracy recalculated from its predictions.
- All nine test category/group/code accuracies and their bootstrap intervals against C1; all nine paired category differences/intervals against C2. The neutral seed text agrees on all 96 test videos in all runs.
- A/G/AG: 6,375 complete non-synthetic distinct windows per condition; reparsed every original label and G/AG slot payload; checked identical A/AG sampled indices and RGB hashes. Recomputed the paired judgment shift and its video-cluster bootstrap interval.
- All G/AG empty and whole-image-box counts, and recall against original COCO annotation JSON at IoU 0.50 and 0.75. Whole-image boxes are retained, not discarded. Window-slot repeats are not independent samples or unique objects.
- Grounding gate: all 398 records, 276 evaluable lifts, V1/V2 estimates and recording-group bootstrap intervals; 112/112 box-mask and 5/112 control-mask flips among correct firearm answers.
- Selective-explanation run/repair counts and all 18 valid-but-empty manual explanations; final repair record lacks a decision.
- Temporal-position point effects recomputed from saved AUC arrays, and the gate logic reproduced. Its bootstrap interval is **source-checked only**, not independently regenerated from frame-level data in this audit.

## Reproduce without GPU or network

Requirements: Python 3, NumPy, SciPy. The original retained inference is not rerun. Versions used for the final audit are recorded in `software.json`.

```sh
mkdir evidence annotations
tar -xzf evidence.tar.gz -C evidence
tar -xzf annotations.tar.gz -C annotations
python3 reproduce.py --evidence evidence --annotations annotations --out verification-rerun.json
pdflatex -halt-on-error -interaction=nonstopmode main.tex
pdflatex -halt-on-error -interaction=nonstopmode main.tex
```

The two archives contain logs, metadata, and annotation JSON; no source videos, sampled images, model weights, or credentials. `manifest.json` in each archive records exact original Oz paths, SHA-256 values, byte sizes, and collection time. Input archives:

| Archive | SHA-256 |
|---|---|
| evidence.tar.gz | `2866a5aff7a874ba067bbd76eeca3ae14e14397fe776e549337d3607da4f3199` |
| annotations.tar.gz | `746173ee50527f348ae2e16f3f47b011240df60679523bcf11bc05cf3d178572` |

## Source map and proof chain

Oz host: `oz.gccis.rit.edu`. Main data root: `/local/datasets/sp2117/firearm_visilab` (abbreviated D below). Source revision IDs are immutable; an open PR's head can subsequently change. The report's evidence is the saved snapshot, not a promise about future PR content.

| Report source | Exact source and how the claim is established |
|---|---|
| T1 | D`/training1/runs/<run>/test.jsonl`, `validations.jsonl`, `spec.json`; D`/training1/data/{train,val,test}.json`; D`/training1/compare/tables/C1.csv`, `C2.csv`; D`/training1/analysis/tables/S4.csv`. Run directory names are enumerated in the archived `compare_manifest.json`. Raw predictions and grouping reproduce all stated category intervals. |
| Faithful procedure | GitHub `shridpant/VERA`, `experiment/firearm-vera-training1` at `c239dc06017756eac4a048d8bdfb4a6b11c35221`: `vera_runner/verbalized.py`, `training.py`, `studies/firearm_training/training1.json`, `tests/firearm/test_verbalized.py`, and the design document. The fake-model faithfulness test is source evidence, not fresh evidence of model quality. Task-label, parser, initialization, order, and checkpoint-selection adaptations remain disclosed. |
| AG | D`/runs/firearm_a_g_ag_v1/records/{A,G,AG}.jsonl`, immutable `spec.json`; D`/analysis/tables/T3.csv`, `T4.csv`, `T7.csv`, `T8.csv`. Git source `534483b35b03717ab1a82bfda73bacd975994edb`, branch `study/firearm-a-g-ag`. Original annotations are D`/original/Gun_Action_Recognition_Dataset/<source parent>/<label_file>`; inventory gives the exact per-video path. Primary origin is 1, mapping image id to frame id minus 1. |
| G1 | D`/grounding1/phase1_seed_measurements.jsonl`, `gate.json`, `analysis/numbers.json`; `phase1_v0_partial_measurements.jsonl` preserved and excluded from the final gate. Source `/local/datasets/sp2117/firearm_visilab/code/VERA-grounding-analysis`, branch `experiment/firearm-grounding`, commit `af94dea7dd37dd7033f37149ee95bf64a596b01c`; `studies/firearm_grounding/gate.py`, `DEVIATIONS.md`, `report/results.tex`. The fixed gate uses shared frames for V1. Phase 2 has no completed arms. |
| Single-window box pilot | `/local/datasets/sp2117/grounding_pilot_01/jobs/grounding_pilot01_20261024T181219_99287/stdout.log`, `exit_status`, and `pilot.py`, archived in evidence. This emits coordinates but has no quantitative box ground truth. |
| SE | `supporting/selective_complete.jsonl`, `selective_partial.jsonl`, `selective_output_audit.json`, `selective_results.tex`; original repo paths and hashes in `supporting/manifest.json`. Complete/partial raw calls are distinct; no imputed last label. |
| POS | `supporting/position_summary.json`, `position_gate.tex`; primary `cohorts.ratio_15_all`. Report point effects are recomputed from `point.original` and `point.no_position`. Interval is the saved `worst_benefit_ci95`, not a newly generated CI. |
| E2/E3 | `supporting/exp02_REPORT.md`, `exp02_ERRATA.md`, `exp03B_R1.md`. These are source-inspected qualitative/integrity context; no new attribution or independent R1 pixel-proof run is claimed. |
| PT | `pr_snapshot.json`, PR 9, exact head and update timestamp retained. It reports corrected gates and a running sweep. No final held-out result was inspected or inferred for this report. |
| P | [VERA paper v2](https://arxiv.org/html/2412.01095v2), Table 9 and training description, accessed October 27, 2026. Published multi-dataset evidence contradicts the unqualified UCF-only assertion; this audit does not independently reproduce the published benchmarks. |

The supporting snapshots come from source commit `5dde005928d42f05625ceab948b8fa2afab772f3`; their individual source paths/hashes are in `supporting/manifest.json`. They are preserved as historical sources, not silently reclassified as fresh experiments.

## Statistical and interpretation limits

Faithful learning changes questions only. Its original loop is adapted to three category fields and actions; it is not an exact repetition of the original binary target. A/B/C name run seeds, not independent datasets. Accuracy intervals are conditional on one fixed split, use recording groups within category, and do not quantify uncertainty across training seeds. The best run is not evidence of robust superiority; the positive third faithful run must also not be suppressed.

The G/AG result applies to one packing/prompt/backbone setting. Annotation `listed_no_box` frames do not provide exhaustive negative ground truth. Recall uses annotated objects, while No_Gun supports negative-frame precision. The attention gate uses all splits, including test boxes, for feasibility; it does not provide untouched test-set inference for a learned feedback method. Measurement deviations and the stopped partial run are disclosed in the original grounding source.

There are no binary anomaly labels for the VISILAB judgment-frequency analysis. No actual target-retrained binary failure was found. Unannotated UCA gaps are not verified anomaly-free or insufficient-evidence cases. A failed prespecified gate can coexist with a measurable positive effect. The report does not equate parser, numerical, validator, scientific, and novelty failures.

## Files

`main.tex` / `main.pdf`: summarized report with discussion at the end. `README.md`: this proof document. `reproduce.py`, `code/`: CPU checks and pinned helpers. `verification.json`, `metric_ledger.csv`: exact outcomes. `supporting/`: contextual snapshots with hashes. `search_scope.json`: bounded search and exclusions. `pr_snapshot.json`: pre-update PR evidence. Two archives preserve the primary logs and original box annotations.

## Validation of the unpublished grounding source

Existing grounding gate tests: 4 passed locally. Existing grounding analysis test: 1 passed locally. Existing grounding-arm fake-model tests: 6 passed on Oz's retained environment. The first local arm-test attempt lacked OpenCV; it was rerun successfully on Oz without dependency changes or GPU inference. An initial local gate invocation used the wrong working directory; the corrected invocation passed. None of these setup errors is a scientific result.

Final PDF: five pages, compiled both by the desktop editor and Oz pdflatex; no undefined references, overfull boxes, underfull boxes, or LaTeX warnings in the final Oz log. Every final page was visually inspected.

## Annotation attribution

Original VISILAB annotations: Jesus Ruiz-Santaquiteria, Juan Daniel Munoz, Francisco Jose Maigler, Oscar Deniz, and Gloria Bueno, *Action recognition and object detection dataset for firearm-related actions*, version 2. [Mendeley Data, DOI 10.17632/bbzpxhd22j.2](https://doi.org/10.17632/bbzpxhd22j.2); [Zenodo record 15387426](https://zenodo.org/records/15387426). Data paper: *Firearm-related action recognition and object detection dataset for video surveillance systems*, Data in Brief 52 (2024), 110030, [DOI 10.1016/j.dib.2024.110030](https://doi.org/10.1016/j.dib.2024.110030). Annotation JSON is preserved unmodified; no new annotations or source images are distributed in this audit.
