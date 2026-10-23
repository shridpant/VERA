# Coding-agent execution handoff: VERA temporal attribution feasibility (v2)

(Verbatim copy of the brief received on 2026-10-22, kept with the experiment for reviewers.)

**Execute this bounded experiment now in the existing Oz project. Return raw results and a decision report.**

This v2 brief replaces the earlier 32-frame / 24-probe alternative-sequence search. That design is superseded. Do not execute it alongside this one. This experiment tests whether useful attribution can be added to the already working VERA inference pipeline. It does not start a human study or a full benchmark.

## 1. Starting state and preservation

The existing coding-agent chat has completed setup and experiment01 on RoadAccidents127_x264. Resume that state. Do not recreate it.

- SSH: `ssh sp2117@oz.gccis.rit.edu`.
- Remote root: `~/vera_research/`. Resolve the absolute path after login.
- Existing source base: `15b8bcb8574a977c229c577f50bfe6f06d07106e`.
- Reported patched source head: `008ed4019781c39e7cb16244b7500244f3986f92`. Inspect current Git state; do not reset it.
- Model: `OpenGVLab/InternVL2-8B`, revision `6fb9ad6924f69424e57fab2ab061d707688f0296`, BF16, frozen weights.
- Existing environment: expected `setup/environment/env/bin/python`.
- Existing helpers: `setup/scripts/launch.sh`, `run_experiment.py`, `vera_pipeline.py`, `validate_run.py`, `fetch_zip_member.py`.
- Completed runs: `experiments/experiment01/runs/run_001` and `run_002`.
- Registry: `EXPERIMENTS.md`. Use the next unused ID, expected experiment02, purpose `temporal_attribution_feasibility`. If another experiment02 exists, allocate the next ID; do not overwrite or renumber it.
- Existing Mac mirror: `/Users/shrid/Documents/research/vera_research/`.

The prior run used a single RTX 6000 Ada successfully. Reported hardware was four RTX 6000 Ada cards (~46 GB) and eight A16s (~15 GB), on Oz itself without a scheduler. Recheck current site instructions and GPU availability. Use an available suitable GPU without disrupting other jobs.

Home usage was about 138.5/150 GiB. Reuse weights and caches. Do not delete old experiments, execute the old cleanup script, purge caches, install a second model copy, or rebuild the environment. The user's latest cleanup instruction was to leave the old experiments alone. Preserve setup and prior outputs. Record any necessary environment changes; isolate new code and outputs.

Troubleshoot the existing model/configuration. Report a blocker before changing model or precision. Optional attribution paths that require a different attention implementation must be isolated and explicitly tested for forward-output agreement. Do not silently change the baseline.

First perform a read-only preflight: host, root, applicable instructions, free disk/quota, GPU availability, environment and source identity. Do not treat documentation of past success as a current runtime verification.

## 2. Scientific objective and boundaries

Question: can we measure temporal input sensitivity of VERA's local VLM judgments, and does testing neighboring frames jointly reveal effects missed by testing one frame at a time?

Motivation: Appendix D found that agreement about the presence of aggression can coexist with disagreement about supporting frames. This motivates evaluating evidence, but does not prove any selected evidence is correct or that this method is novel.

Implement and evaluate a small temporal-attribution component. Compare individual-frame perturbations with consecutive temporal-group perturbations. Separately assess whether a usable spatial attribution route exists for this InternVL checkpoint.

Keep these targets distinct:

1. **Original VERA local prediction:** generated explanation plus parsed binary label from its original prompt.
2. **Instrumented class scorer:** an optional, explicitly modified label-only inference condition used for continuous scores and possibly gradients.
3. **Final VERA timeline:** combines other windows through ImageBind and numerical refinement. This experiment does not attribute that timeline.

Do not report an explanation of target 2 as if it were target 1 or 3. Do not call token probabilities calibrated uncertainty. Do not infer frame importance from the brightness of independently normalized heatmaps.

No training, question optimization, alternative-frame search, full-dataset inference, multi-agent system, human recruitment, participant interface, or heatmap-feedback-to-VLM experiment in this run. No new ImageBind extraction: use the local VLM only. Original ImageBind features are not interchangeable with perturbed inputs.

## 3. Cases and data

Use UCF-Crime and the existing extraction/preprocessing. Keep **exactly eight image positions** in all primary comparisons. Each case uses its original sampled frame IDs. Do not introduce a 32-frame candidate pool.

Read the actual original raw records to recover windows and frame files. Expected diagnostic cases:

| Case | Video / anchor | Role |
|---|---|---|
| R-start | RoadAccidents127_x264 / 0 | Ordinary early-video diagnostic |
| R-pedestrian | RoadAccidents127_x264 / 496 | Baseline false-alert candidate; benchmark-negative interval |
| R-accident | RoadAccidents127_x264 / 2240 | Main accident window |
| R-end | RoadAccidents127_x264 / 2576 | Late-video diagnostic |
| N-middle | Normal_Videos_018_x264 / midpoint | Independent normal-video control |

Road video expected metadata: 2,580 frames, 30 FPS. R-accident window `[2090,2390)`; benchmark accident `[2160,2300)` under the existing implementation convention. R-pedestrian window `[346,646)`. Validate all values against manifests. A benchmark-negative interval may contain other semantically unusual behavior; do not declare it objectively normal without inspection.

If the normal clip is absent, use the existing official-archive range-fetch helper to obtain that member only. Inspect its real CLI. Do not download the entire dataset. Use the original ten-second sampling around a valid midpoint anchor. Save the exact anchor, interval, eight frame IDs, metadata, provenance, hashes, and annotation convention.

If the normal clip cannot be acquired, complete available work and flag the missing independent control. No silent substitution.

Use all five cases for scoring/prompt diagnostics. Run full temporal perturbations on R-accident, R-pedestrian, and N-middle only. These are feasibility cases from two videos, not a general-performance sample. Seeds, perturbations and windows are not independent videos. No dataset-wide accuracy or significance claim.

## 4. Stage A — baseline and output scoring

### A1. Recover and reproduce original inference

Reuse exact original system/user templates, five UCF questions, image preprocessing, image order, BF16 model, generation settings, and strict parser. Preserve UTF-8 handling and InternVL's effective EOS generation configuration. Never silently replace a failed input with a random video.

Rerun at least R-accident and R-pedestrian and compare against saved labels/text. Record discrepancies before continuing. Generate baseline outputs for N-middle. Reuse verified historical records where appropriate, but label cache versus fresh execution.

Primary outcome remains the original-prompt parsed binary label. Save raw text as well. Parse failures are missing/error states, never normal labels.

### A2. Implement a bounded numeric-score diagnostic

The existing binary outputs may be too coarse for stable rankings. Investigate an instrumented scorer for at most 60 minutes of engineering before falling back to binary perturbation diagnostics.

Use the same model, visual inputs and guiding questions, but an explicitly saved prompt asking for only the class label. Compute the conditional log-likelihood of each candidate answer, `0` and `1`, at a fixed assistant-response boundary. Define `S = log P(answer=1 | input,prompt) - log P(answer=0 | input,prompt)`.

Verify tokenizer and chat-template behavior. Do not assume each answer is one token. For multi-token answers, use full continuation log-likelihood with a consistent boundary/termination policy. Exclude prompt/image tokens from the answer score. Test scoring alignment against the model's forward logits and actual label-only generation. Save candidate token IDs and exact scoring code/version.

Do not use an already generated original explanation as a fixed prefix for this scorer: that can carry the old decision into the perturbed run. This is a **modified scoring condition**, not exact instrumentation of the original explanation-first VERA decision.

For all diagnostic cases, compare original-prompt labels, label-only generation, score sign, and score magnitude. Record disagreements separately. Repeat an identical input to estimate numeric repeatability. Check finite values and nonzero sensitivity on strongly altered inputs. A small agreement table is only a feasibility diagnostic; it does not validate the scorer on UCF-Crime.

If numeric scoring is blocked or evidently unrelated to original outputs, do not manufacture a proxy from explanation length, text sentiment, or attention. Continue original-label tests, and report numeric attribution as blocked or provisional. Do not silently rewrite the original prompt.

### A3. Small prompt ablation, not optimization

On the five unperturbed diagnostic cases, also test the original prompt with **only the sound question removed**. Preserve other wording and formatting as closely as possible; save the exact diff. Record binary changes and unsupported audio claims in raw responses.

This checks the known image-only/audio-question mismatch. It does not justify selecting a globally better prompt from five cases. Keep the original prompt as primary for the rest of this feasibility run. No optimizer-VLM training. Do not use human-study data to select questions.

## 5. Stage B — temporal perturbations

Implement pixel-space transformations before normalization. Preserve the original eight image slots, timestamps, ordering, and all prompt text. Record exact transformation parameters and save representative transformed inputs or their reproducible recipes/hashes.

Use two declared replacement families:

- **Blur:** strong Gaussian blur of the selected original frames. Start with radius `0.04 * min(width,height)` at the pre-normalization image resolution; record exact resolution/radius. This is a provisional information-suppression operation, not proven erasure.
- **Constant:** replace selected frames with a fixed RGB `(127,127,127)` image at the same resolution. This is a conspicuous out-of-distribution control, not a realistic missing-frame model.

Do not delete or duplicate frames in the primary comparison. Explain each replacement's limitations. If transforms have visibly negligible effect, document before any parameter revision; preserve the failed setting and create a new config.

For each primary case and replacement family, evaluate:

- Eight individual-frame interventions.
- Seven adjacent two-frame interventions.
- Six adjacent three-frame interventions.
- One all-eight intervention as an extreme diagnostic.

That is 22 unique masks per replacement family, before caching. Run original-prompt generation for all. Also run numeric scoring if Stage A provides a working implementation, retaining its separate target label.

For each group G, store original/perturbed outputs and `Delta(G) = S(original) - S(perturbed G)` if a valid numeric score exists. For original binary outputs, store the corresponding signed binary difference and whether the decision flipped. Preserve negative values; masking can increase anomaly preference.

Positive Delta indicates support for anomaly preference, not necessarily support for the model's predicted class. For a normal prediction, distinguish evidence supporting normality from evidence supporting anomaly. Do not call all absolute changes "positive anomaly evidence."

Report group interactions descriptively: `Delta(group) - sum(Delta(single frame) for frames in group)`. This is non-additivity on the chosen score scale, not proof of causal synergy or novelty. Saturation and thresholding can create apparent interactions.

Keep group effects as a primary interval-level output. If a frame ranking is needed, implement a clearly labelled heuristic: for each frame, average `Delta(G)/|G|` over the tested groups containing it. Record group sizes and unequal edge coverage. Compare singleton-only versus pair/triple-group rankings. Do not present this heuristic as a theoretically validated attribution rule. Test a constant synthetic group-effect fixture to expose boundary artifacts.

## 6. Stage C — small evaluation independent of ranking construction

For informative numeric rankings, construct rankings using blur effects and evaluate them using constant replacements. Also reverse the direction. This is a robustness check, not proof that the second replacement is realistic.

Compare singleton ranking, temporal-group ranking, and seeded random ranking. For random controls, use seeds 0,1,2. Break attribution ties deterministically by original frame index and flag ties; do not claim a meaningful ranking when all scores tie.

At evidence budgets 2 and 4:

- **Removal:** replace the top-ranked frames.
- **Retention:** retain the top-ranked frames and replace the rest.

Record actual original-prompt labels and optional numeric scores. Cache overlap with Stage B. Do not compute glossy insertion/deletion AUC claims from only two budgets. Report explicit score changes and labels at each budget.

For binary-only analysis, ties and degeneracy may prevent ranking. Report that outcome rather than forcing ranked selection. Compare meaningful observed flips and group-versus-single differences descriptively.

Explain whether temporal groups produce signal beyond simply perturbing more pixels. Include matched-cardinality random or nonconsecutive groups in comparisons. Add three seeded nonconsecutive pairs and three seeded nonconsecutive triples per primary case and replacement family, where feasible within the cap. Save masks so this is reproducible.

Evaluate anomaly preference and preservation of the original predicted class separately. Preserving an incorrect baseline prediction is not improved correctness.

## 7. Stage D — spatial attribution feasibility, tightly bounded

The user wants to understand whether a visual heatmap can accompany VERA in a later human study. Answer that feasibility question; do not promise Grad-CAM++ in advance.

Only attempt this if numeric scoring is implemented and basic temporal perturbations are complete. Budget at most 45 additional engineering minutes and one case, R-accident. Inspect actual InternVL vision features and language-model connectivity. Determine whether gradients of the declared score reach spatially indexed visual features or input pixels.

Frozen weights do not forbid input/activation gradients. However, existing inference-mode/no-grad wrappers may need an isolated differentiable forward path. Do not modify the baseline globally. Avoid storing full model parameter gradients. Report exact target layer/tensor, spatial/token mapping, class score, gradient finiteness, and VRAM cost.

If a reproducible implementation of Grad-CAM++ is compatible, document its derivative/weight formula and adaptations. Otherwise report it unsupported in this time budget. Do not rename raw attention, gradient-times-input, or ordinary Grad-CAM as Grad-CAM++.

A simple correctly named gradient or occlusion map can be a diagnostic, but it explains the instrumented score only. Do not interpret visually pleasing output as validation. If attention backend changes are necessary, compare forward scores with the baseline path and report deviations. If that is infeasible, stop this branch.

No spatial heatmap is required for the temporal branch to finish. Report separate feasibility outcomes for original binary sensitivity, numeric attribution, and spatial heatmaps. If gradient work threatens the working environment or budget, return the concrete limitation and the least costly next test.

## 8. Human-study implications to report, not execute

The eventual pilot may compare classification alone, plus text, plus attribution, and plus both. This feasibility task only produces reusable stimuli data and establishes whether attribution is interpretable enough to justify that pilot.

Do not generate attribution for frames never supplied to the model. Eight input-frame scores do not provide a measured dense importance score for an entire clip. Do not interpolate such scores and present them as measurements. Report how sparse model input could be visibly marked in a later numbered-frame interface.

Do not equate model-human overlap with correctness. Human and model evidence access must be described. A later study can examine influence and appropriate reliance; this feasibility run cannot establish those outcomes.

If inspecting explanation claims, use supported/contradicted/not assessable only after actually examining frames. List which case images were inspected. Otherwise mark visual grounding unverified. Model self-reports and an LLM judge are not ground truth.

## 9. Execution limits and validity checks

Use at most 400 unique transformed input sequences and three GPU wall-clock hours, whichever comes first. Count generation, numeric scoring and gradient calls separately. Stop within budget and report incomplete stages. Engineering time limits for A2 and D are additional scope controls, not statistical guarantees.

Do not use the entire cap merely because it exists. Stop a branch when it cannot answer its question. Do not start a full run after a positive result without a new instruction.

Build a small experiment-local runner, config, and analyzer. Reuse setup loaders. Include meaningful checks for:

- Correct source frames/timestamps and eight preserved input slots.
- Only intended pixels/frames changed; preprocessing identical elsewhere.
- Group enumeration, matching intervention sizes, and cache-key completeness.
- Correct parsing, likelihood indexing, finite scores and repeatability.
- Signed effects, normal/anomaly target distinction, ties and missing outcomes.
- Resume safety and unchanged original runs.

Cache keys must include model revision, prompt/scoring contract, source identity, frame IDs, transformation family/parameters, and preprocessing. Do not reuse original-prompt generation as a label-only scoring record. Save effective generation settings, including model-added values.

Primary tests are three cases from two videos. Report descriptive case results. Do not treat the number of masks as the scientific sample size. Do not claim generic accuracy gains, new theory, calibrated uncertainty, guaranteed faithfulness, or top-tier novelty.

## 10. Deliverables and completion contract

Under the new experiment folder, create reusable code/configs and immutable run outputs:

- `REPORT.md`: short plain-language summary, then actual design, findings, costs, limitations, and decision.
- `resolved_config.yaml`, `run_manifest.json`, source/environment/host records and logs.
- `cases.json`: original frame IDs, timestamps, intervals, provenance, annotations and hashes.
- `scoring_contract.md`: exact targets, prompts, tokenization, formulas, and distinctions from original VERA.
- `baseline_responses.jsonl`, `prompt_ablation.jsonl`, `perturbation_responses.jsonl`: exact inputs, prompts, raw generated text, labels, scores, errors and cache status.
- `temporal_effects.csv`: frame/group effects, signed targets and interaction diagnostics.
- `evaluation.csv`: retention/removal and matched-cardinality controls.
- `spatial_feasibility.md`: attempted path, actual results, implementation limits, and whether any map is valid only for the auxiliary scorer. If maps were successfully generated, save native arrays and a small labelled diagnostic preview, not a participant dashboard.
- `validation.json`, measured runtime/VRAM/call counts, and `NEXT_STEP.md`.

Report at least one concrete input/output example for each primary case. Compare text at the claim level, not only token overlap. Do not assert unsupported grounding.

End with a separate GO / CONDITIONAL GO / NO-GO decision for:

1. Original VERA binary temporal sensitivity.
2. Instrumented numeric temporal attribution.
3. Spatial heatmaps suitable for further validation.

Recommend one next experiment. Explicitly answer: Is there useful sensitivity? Does grouping add information beyond mask size? Do rankings depend on replacement choice? Does the numeric scorer diverge from original VERA? Are sound-question effects material in these examples? Can heatmaps be implemented within the thesis budget? What remains before using explanations with humans?

Update `EXPERIMENTS.md` and relevant setup-status links without overwriting historical results. Copy compact reports, scripts, configurations and raw records to the Mac mirror if the existing authenticated transfer route is available. Exclude weights, credentials and bulk frame caches. Otherwise provide exact remote paths and a tested retrieval command where possible.

Execute the bounded work rather than stopping after writing a plan. Resolve routine implementation issues within scope. If access, data or required capabilities are blocked, preserve completed work and provide the exact blocker. Do not substitute a different model, experiment target, or precision silently.

## Verification status of this handoff

Prepared from the returned experiment01 configuration, registry, raw-output structure and helper files. Local helper paths and group-count arithmetic were inspected. Remote reachability, new scoring code, gradients and perturbation experiments are not verified by the planning assistant. Only mark them tested after execution with recorded evidence.

Background sources: VERA https://arxiv.org/abs/2412.01095 ; RISE https://arxiv.org/abs/1806.07421 ; Grad-CAM++ https://arxiv.org/abs/1710.11063 . These motivate feasibility checks, not claims that the new implementation is already validated or novel.
