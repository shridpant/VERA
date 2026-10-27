# Grounding statistical amendment v1 — October 27, 2026

## Authorization and timing

The human in chat `01a17dce-3275-78b0-b005-0f3a408d5cbe` requested: “add a proper statistical test for grounding too, before looking at final results. keep the scoring rules unchanged and document it as a protocol amendment.”

Training had already started under immutable source `8d9232c` and protocol `3f1f1e8a9aa78a0c5124f765df6712d2c33c0a1db770947edd3318cbec4abe80`. The initial metadata-only check showed the first seed's paired scientific warm-ups, no completed method/seed outcomes, no selection manifest, and no evaluation/report directory. No final predictions, targets, comparative scores or statistical outcomes were inspected to select this test. Development format/reward feasibility results were already known and disclosed. This is a **prospective, pre-final-result analysis amendment after training began**, not original pre-training preregistration. Historical use of the reporting dataset remains disclosed.

The base protocol and active training release stay immutable. A new clean analysis release writes a separately hashed amendment under `ROOT/analysis_amendments/grounding-v1.json`, binding the base protocol file/hash, exact analysis source and commit, timestamp, human authorization, and metadata-only evidence that selection/evaluation/reporting had not begun. Freeze fails closed if any selection, evaluation directory/job, or report already exists. Training need not restart because the amendment cannot affect gradients, rewards, sampling, checkpoint selection, or stopping.

## Question, endpoint and unchanged scoring

Contrast: **RG minus SG**, using paired training seeds. Question: does grounded GRPO improve box localization over continued supervised grounding from the same starting adapters?

For each of the three fixed seeds, pool the **unchanged** localization scorer's true positives, predicted boxes and ground-truth boxes across the reporting split. Derive `F1 = 2 TP / (N_pred + N_GT)` at the existing IoU threshold **0.50**. The endpoint is the arithmetic mean of these three seed-specific pooled F1 values; the effect is RG minus SG. This balances recall and false predictions in one prespecified endpoint. It is not an average of frame/video F1 values.

The underlying `report.localization` function remains byte-for-byte unchanged: same strict parser, native-coordinate conversion, inverse letterbox, one-to-one IoU matching, unknown-annotation exclusions, whole-image-box suppression, and treatment of invalid outputs. Reward coefficients, training loss, class scores, IoU threshold, annotation status rules, format validity and all scientific stops remain unchanged. The supplement decomposes the existing scorer by video and asserts that recombined TP/prediction/GT and other counts exactly match the base report for every method/seed/split.

F1 is an added **analysis endpoint**, not a revised scoring rule. Existing precision, recall, No_Gun false-box rate, whole-image-box rate, invalid/truncated output rate and annotation coverage remain mandatory supporting results. F1 superiority does not imply every component improved; report adverse component changes rather than hiding them. No new false-box noninferiority margin is invented here.

## Paired group test and confidence interval

The independent sampling unit is the **recording group**. All videos, frames, boxes and three seed pairs within a group remain together. Seeds describe training variability, not three independent dataset replicates.

- Test statistic: seed-averaged pooled F1 difference, RG minus SG.
- Null: joint groupwise exchangeability of RG and SG count vectors, conditional on these fixed training seeds. Inference assumes independent representative recording groups and method-label exchangeability under this null; this is not a randomized treatment experiment or an assumption-free test of a weak mean-only null.
- Two-sided paired method-label permutation: swap RG/SG jointly for an entire recording group across all videos and seeds. Enumerate all `2^G` swaps for `G<=16`; otherwise draw **100,000** swaps, using the **plus-one** Monte Carlo p-value. Absolute test statistics define both tails. RNG seed **20261028** for permutation draws.
- Effect uncertainty: **5,000** paired recording-group bootstrap samples, recomputing pooled ratios on each sample. Report a **95% percentile interval**; RNG seed **20261027**. Do not resample individual frames, boxes, or seeds independently.
- Positive grounding evidence requires positive effect, two-sided **p<=0.05**, and a strictly positive lower 95% interval bound. Report exact/Monte Carlo mode, group counts, replicate counts, seed-specific F1 and actual effect size.
- Declare inconclusive if fewer than two positive-GT recording groups or fewer than 95% of bootstrap draws retain positive-GT support. Coarse permutation resolution, wide intervals, or sparse groups can prevent a claim; a nonsignificant result is not equivalence.

The permutation convention follows the paired-sample/exact-or-randomized framework and conservative randomized p-value convention described in the [official SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html). Here the swaps occur at group level, not frame level.

## Claim hierarchy and multiplicity

The **test split** is the sole confirmatory reporting split for this claim family. Validation is exploratory; its statistics do not select a model, endpoint, threshold, or test variant.

Use a fixed sequence at nominal alpha 0.05:

1. Apply the **existing unchanged test RG–SG recognition rule**: positive lower 95% paired balanced-category-accuracy interval and the existing No_Gun +0.05 noninferiority/support screen.
2. Only if step 1 passes may the grounding endpoint support a confirmatory superiority claim, using the p-value/effect/interval rule above.

If recognition does not pass, still report grounding estimates, intervals and nominal p-values, but label them exploratory and do not claim confirmatory success. The existing base report's numerical flags remain unchanged; this supplement governs the two-claim interpretation. Other contrasts, validation outcomes, precision/recall components and seed comparisons are descriptive/exploratory, not additional uncorrected confirmatory discoveries. This avoids adding an independently selectable second primary claim. Error control remains conditional on the validity/approximation assumptions of the component tests, including the existing bootstrap recognition rule.

## Automatic execution and audit

The supplemental watcher uses no GPU. It verifies its immutable amendment/source and the base protocol, then waits for **all 15 selections to freeze and the full base evaluation/report to complete**. Only then can it read reporting data/predictions. It checks prediction and annotation hashes against completed evaluations, reproduces base counts, and writes a separate `ROOT/report/grounding-statistics-v1/` report, results JSON and completion status. It never overwrites the original report or scores. Any mismatch or failed base pipeline stops it with a recorded error.

The study is not fully reported until this supplemental status is complete as well as the base report. Preserve the base protocol/release, failed pilots, all seeds, raw outcomes, and the amendment timestamp/source binding. Any later change to this plan or its analysis source requires a separately versioned amendment; never silently update v1 after seeing outcomes.
