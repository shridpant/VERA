# experiment02: VERA temporal attribution feasibility. Report (run_002, with run_001)

## Plain-language summary

I tested whether we can measure **which of the 8 frames** VERA's vision-language model relies on when it labels a
10-second window. I replaced single frames, or short runs of neighbouring frames, with a heavily blurred version or a
plain grey image, then reran the model.

The test used three windows from two UCF-Crime videos:

- **R-accident:** a truck–train collision;
- **R-pedestrian:** a window VERA flags but the benchmark marks normal, so a likely false alert;
- **N-middle:** a benchmark-normal video (Normal_Videos_018; to the agent's eye an airport hall).

**What happened:**

- **R-accident.** VERA's yes/no answer never changed when *one* frame was replaced.
  - It flipped to "normal" for some groups of two or three neighbouring frames: 4 of 13 such groups under blur and 3
    of 13 under grey. Every one of them contained the 74.7 s frame.
  - It never flipped for the six same-size non-neighbouring control groups.
  - Those controls also had smaller combined single-frame effects, so this run cannot separate "neighbouring" from
    "more evidence removed".
- **A continuous score adds resolution, under a different condition.** I also computed an auxiliary score: the
  model's log-odds of answering "1" rather than "0" under a label-only prompt.
  - That prompt drops VERA's explanation step, so it is a different decision condition. It already disagrees with
    VERA on the pedestrian window.
  - Effects use S_hi, an fp32 recomputation of the model's final logit difference.
  - On the accident window, both attribution rankings put the same two frames on top whichever replacement was used.
  - Removing those two erased about 95 % of the anomaly preference, and keeping only those two preserved 86–88 %.
  - Random pairs erased 15 % on average and preserved at most 32 %. One random draw that happened to include the
    74.7 s frame erased 57–62 %.
- **R-pedestrian.** The pedestrian "anomaly" is fragile. Almost *any* replacement, and also simply deleting the
  sound question from the prompt, turns it back to "normal". There is no stable frame attribution to report there.
- **N-middle.** Nothing flips, and every per-frame effect is smaller than the model's numerical resolution.
- **Heatmaps.** Spatial heatmaps can be computed cheaply. These ones are not trustworthy yet: 23–35 % of their mass
  sits on the black letterbox bars, which are 25 % of the image area.

This is a feasibility check on 3 windows from 2 videos. It shows the measurements work and can be informative. It
does not show that they generalise, are faithful, or help people.

**Decisions:**

| Area | Decision |
|---|---|
| Original binary temporal sensitivity | CONDITIONAL GO |
| Numeric temporal attribution | CONDITIONAL GO (provisional) |
| Spatial heatmaps | CONDITIONAL GO for technical validation only; NO-GO for participants |

## 1. What was run

| | |
|---|---|
| Host / GPU | oz.gccis.rit.edu, one NVIDIA RTX 6000 Ada (46,068 MiB ≈ 45 GiB), no scheduler; idle at preflight |
| Model | OpenGVLab/InternVL2-8B @ `6fb9ad69`, **BF16**, frozen, FlashAttention 2 (unchanged from experiment01) |
| Source | VERA `15b8bcb8` + experiment01 patch → `008ed401` (clean, unchanged) |
| Code | `experiments/experiment02/code/` (run_attr.py, engine.py, attr_lib.py, analyze_attr.py, validate_attr.py, spatial_probe.py, tests). It is isolated from `setup/`, and the setup files are unchanged (validator V12). |
| Config | `experiments/experiment02/config.yaml` (v2), frozen copy in `run_002/resolved_config.yaml` |
| Runs | `run_001`: Stage A only under config v1, numeric scorer **blocked** (kept unchanged). `run_002`: Stages A, B, C and D under config v2, with post-hoc gate changes disclosed in §3. |
| Cost | 0.30 GPU-hours (of 3); 265 transformed-sequence computations charged against the 400 cap (259 distinct inputs: run_001's 6 were recomputed in run_002); 285 generations across all prompt variants (original, label-only, no-sound); 296 scorer forwards (295 counted plus 1 uncounted Stage D reference); 10 alignment checks; 1 gradient call. See `results/costs.json`. |
| Validation | run_002 `validate_attr.py`: all checks pass (`results/validation.json`). CPU tests `test_attr_lib.py` pass. |

### Cases (`cases.json`): eight original frames each

Frame IDs for the four RoadAccidents127 cases match the experiment01 records. N-middle uses the new midpoint rule.

| Case | Video / anchor | Window (frames / s) | Benchmark annotation | Role |
|---|---|---|---|---|
| R-start | RoadAccidents127 / 0 | [0, 150) / 0–5 s | accident [2160, 2300) elsewhere | diagnostics only |
| R-pedestrian | RoadAccidents127 / 496 | [346, 646) / 11.5–21.5 s | no overlap (benchmark-negative) | primary |
| R-accident | RoadAccidents127 / 2240 | [2090, 2390) / 69.7–79.7 s | contains all 140 annotated frames (72.0–76.7 s) | primary |
| R-end | RoadAccidents127 / 2576 | [2426, 2580) / 80.9–86.0 s | no overlap | diagnostics only |
| N-middle | Normal_Videos_018 / 592 | [442, 742) / 14.7–24.7 s | normal video (all −1) | primary (independent control) |

- **RoadAccidents127:** 2580 frames at 30.0 fps, matching the manifest.
- **Normal_Videos_018:** range-fetched alone from the official `Testing_Normal_Videos.zip` (9.4 MB over HTTP,
  CRC-checked). It is 1181 frames at 30.0 fps; the midpoint anchor 592 is the VERA anchor closest to frame 590.5.
- **Annotation convention:** `compute_auc.py` reads [start, end) as zero-based. The official annotation file does not
  state its convention, which affects at most one frame at each end.

## 2. Stage A: baseline, numeric scorer, sound ablation

**A1. Reproduction.**

- Rerunning VERA's original prompt reproduced experiment01 exactly for R-start, R-pedestrian, R-accident and R-end:
  identical raw text, labels and token ids, in both run_001 and run_002.
- N-middle (new): label 0.

**A2. Numeric scorer.** It uses a modified label-only prompt (`results/prompts/label_only_instruct.diff`: 3 edits
touching 4 lines. Two are reworded, and the "Answers to Prompt Questions" and "Output:" format lines are removed. All
5 questions, frames and the system message are unchanged). It reads the next-token log-probabilities of '0'
(token 300) and '1' (token 312) at the start of the assistant reply: `S = log P(1) − log P(0)`.

- The scorer's logits equal `generate()`'s own first-step scores **exactly** (max difference 0.0) on all 5 cases.
- Identical inputs give identical S.
- P('0') + P('1') = 0.82–0.91.
- **Resolution limit:** the two logits sit near 36, where bf16 numbers are 0.25 apart, so **S moves in 0.25-nat steps**
  whenever both logits are ≥ 32. That holds for every unperturbed input. The all-grey inputs drop logit('1') to 31.625,
  which is on a 0.125 grid.

| Case | Original-prompt label | Label-only generation | S (bf16) | S_hi |
|---|---|---|---|---|
| R-start | 0 | 0 | −1.00 | −0.95 |
| R-pedestrian | **1** | **0** | **0.00 (exact tie: both logits 36.0)** | −0.13 |
| R-accident | 1 | 1 | +2.75 | +2.85 |
| R-end | 0 | 0 | −1.50 | −1.42 |
| N-middle | 0 | 0 | −1.75 | −1.81 |

**A3. Sound ablation** (only question 5 removed; `results/prompts/nosound_questions.diff`):

- **R-pedestrian flips 1 → 0.** "a person crossing the road in a way that is not consistent with their usual
  behavior … there is an anomaly" becomes "a person walking across the road in a crosswalk, which is a typical activity
  … no anomaly".
- The other 4 labels are unchanged.
- All 4 baseline claims of "no unusual sounds or noises" disappear. They were made without any audio input.

## 3. Disclosed post-hoc changes (run_001 → run_002)

My v1 gate, fixed before any model output, classed the numeric scorer as **blocked** (`run_001/results/scorer_diagnostics.json`).

- **Why:** on R-pedestrian, S = 0 exactly. The v1 tie rule sends S = 0 to anomaly, but greedy decoding picked '0'
  (token 300, the lower id). Every other criterion passed. run_001 is kept unchanged.
- **Config v2 (run_002) changes**, made after seeing that result:
  1. Exact ties are "undecided" in the gate, rather than counted as agreement or disagreement.
  2. The code caps the scorer status at **`provisional_post_hoc_gate`**.
  3. Effects use **S_hi**: the same final bf16 hidden state and head rows, with the last two dot products kept in fp32
     instead of rounded to the 0.25 grid. Every S_hi is checked to round back to the model's own bf16 logits (all
     pass). `Delta_bf16` on the coarse S is reported alongside.
     - This also moved the ranking direction and the scorer class onto S_hi.
     - For R-pedestrian it reversed the direction. The pre-declared rule (sign of S, ties to anomaly) gives +1 at
       S = 0; S_hi = −0.131 gives −1 (normal).
     - Under the pre-declared S ≥ 0 rule, the §4 label/scorer agreement would read 247 rather than 255 records.
- **Changes before any model call**, after an 11-agent code-review workflow.
  - `run_001/run_manifest.json` (`config_amendments`) records the config part: explicit gate criteria, with missing or
    non-finite values counted as failures, plus the tie and resolution rules, the perturbed candidate-mass flag, the
    ranking requirements and per-request budget enforcement.
  - Code changes from the same review are not separately recorded. They include retrying errored requests, a run
    lock and code-hash freezing; `run_001/code_snapshot/` holds the resulting code.

## 4. Stage B: temporal perturbations

The setup per primary case and replacement family:

- 22 masks: 8 single frames, 7 adjacent pairs, 6 adjacent triples and all 8.
- Plus 6 seeded **nonconsecutive** controls (pairs 1-6, 2-4, 4-6; triples 0-3-7, 0-4-7, 3-5-7).
- Each mask got one original-prompt generation and one numeric score.
- **Blur:** PIL GaussianBlur, σ = 0.04 × 448 = 17.92 px on the 448×448 image, after VERA's resize and before
  normalisation. In the agent's inspection of the three slot-3 examples in `results/example_inputs/`, the pedestrians
  are no longer discernible but coarse colour blobs (the red truck) remain.
- **Constant:** RGB (127, 127, 127), a conspicuous out-of-distribution control.
- Only the intended slots changed, with bit-identical tensors elsewhere (validator V03, V17).

Full tables: `results/temporal_effects.csv` and `results/frame_attribution.csv`.

`Delta = S_hi(original) − S_hi(perturbed)`. Positive means the replaced frames supported the anomaly preference.

### R-accident: original label 1, S_hi 2.85

**Original-prompt (VERA) label flips:**

| Mask type | Blur | Constant |
|---|---|---|
| single frames | **0/8** | **0/8** |
| adjacent pairs | 2/7 (3-4, 4-5) | 1/7 (4-5) |
| adjacent triples | 2/6 (3-4-5, 4-5-6) | 2/6 (2-3-4, 4-5-6) |
| nonconsecutive pairs and triples | 0/6 | 0/6 |
| all eight | flip | flip |

**Numeric effects on S_hi (single frames, blur / constant):**

| Slot | Time | Delta |
|---|---|---|
| 4 | 74.7 s | +1.59 / +1.74 |
| 3 | 73.4 s | +0.74 / +0.80 |
| 5 | 75.9 s | +0.54 / +0.43 |
| 1 | 70.9 s | +0.28 / +0.21 |
| others | | |Delta| ≤ 0.11, except slot 0 (−0.13 / −0.14) and constant slot 7 (−0.21) |

**Adjacent pairs and triples:**

- Pair 3-4: +2.72 / +2.73 (interaction +0.39 / +0.19). Pair 4-5: +2.66 / +2.65. Triple 3-4-5: +3.70 / +3.95.
- Mean interaction for adjacent triples is +0.35 / +0.31, against +0.01 / +0.03 for nonconsecutive triples.
- Adjacent pairs: +0.20 / +0.16, against nonconsecutive pairs +0.08 / +0.10.
- Nonconsecutive pairs containing slot 4 (2-4, 4-6) still move S_hi by about +1.6 to +1.9, but never flip the label.

**Frame inspection (agent, not human).** Slot 3 is the visible moment the train strikes the trailer. Slot 4 shows
the train passing with the detached truck cab, and slot 5 the train with debris on the road.

### R-pedestrian: original label 1, S = 0 (S_hi −0.13): a fragile alert

- **The label flips almost everywhere:** 5/8 single frames (blur) and 8/8 (constant). It also flips for most
  adjacent and nonconsecutive groups. Only 5 of 28 blur masks and 3 of 28 constant masks keep label 1.
- **The numeric effects are tiny** (|Delta| ≤ 0.71, all-eight aside). Most are below one bf16 step (0.25): only 16 of
  54 non-all-eight masks exceed it.
- **The text swings.** Replacing slot 0 (blur) gives "people crossing the road in a crosswalk … typical" → 0. Replacing
  slots 5–6 (blur) gives "crossing diagonally instead of using the crosswalk" → 1.
- **Agent inspection:** two pedestrians walk across the crossing area. "Person crossing the road" is *supported*.
  "Not consistent with usual behavior" is *not assessable*; it looked ordinary to the agent.

### N-middle: original label 0, S_hi −1.81

- 0 flips on all 56 masks.
- Every non-all-eight |Delta| is ≤ 0.23, below one bf16 step. The scorer's normal preference weakens slightly when frames are removed.
- All-eight: −1.26 (blur) and −0.93 (constant). Blank inputs make the scorer less sure of normality.

### Original label vs numeric scorer under perturbation

Across the 255 distinct perturbed inputs (Stages B and C), the original-prompt label and the scorer class (S_hi ≥ 0)
agree on 225:

| Case | Agree |
|---|---|
| R-accident | 72/82 |
| R-pedestrian | 69/89 |
| N-middle | 84/84 |

Counting request records instead, which include 33 cached repeats, the figure is 255/288. These are two different
decision conditions, so disagreement does not mean either one is wrong.

Example: R-accident all-eight blur gives S_hi +0.85 (scorer still says anomaly) while the original-prompt label is 0.

## 5. Stage C: cross-family ranking evaluation

How it was set up:

- Rankings were built from one replacement family and evaluated with the other, in both directions.
- Three ranking methods were compared: singleton; temporal-group (size-balanced mean of `Delta/|G|` over adjacent pairs
  and triples, chosen before data because the brief's pooled mean fails the constant-effect fixture); and random (seeds 0, 1, 2).
- Budgets were 2 and 4 frames, each in removal and retention mode.
- The ranking direction is support for the scorer's baseline class: anomaly for R-accident, normal for R-pedestrian
  (S_hi < 0) and N-middle.

Full table: `results/evaluation.csv`.

### R-accident: attribution-ranked frames clearly beat random

Values are S_hi after replacement (baseline 2.85). The original-prompt label is in brackets.

| Budget, mode | Singleton | Group | Random seeds 0/1/2 (mean) |
|---|---|---|---|
| 2, remove top (blur eval) | 0.13 [0] | 0.13 [0] | 2.91, 1.24, 3.14 (2.43) |
| 2, keep only top (blur eval) | 2.51 [1] | 2.51 [1] | −1.26, 0.91, −0.81 (−0.39) |
| 4, remove top (blur eval) | −0.86 [0] | −0.71 [1] | 2.42, 1.52, 3.00 (2.31) |
| 4, keep only top (blur eval) | 3.00 [1] | 3.02 [1] | 0.05, 0.77, −0.86 (−0.01) |
| 2, remove top (constant eval) | 0.12 [1] | 0.12 [1] | 3.01, 1.08, 3.17 (2.42) |
| 2, keep only top (constant eval) | 2.44 [1] | 2.44 [1] | −1.25, 0.72, −0.92 (−0.48) |
| 4, remove top (constant eval) | −1.10 [0] | −0.93 [0] | 2.16, 1.67, 2.87 (2.23) |
| 4, keep only top (constant eval) | 2.87 [1] | 2.88 [1] | 0.11, 0.97, −1.10 (−0.01) |

- The top-2 frames are {3, 4} under both attribution-based methods (singleton and group) in both source families.
  The random controls' top-2 were {6,7}, {2,4} and {0,7}.
- Rankings barely depend on the replacement family:

| Comparison | Kendall τ_b | top-2 overlap | top-4 overlap |
|---|---|---|---|
| singleton, blur vs constant | 0.86 | 2/2 | 4/4 |
| group, blur vs constant | 0.93 | 2/2 | 4/4 |
| singleton vs group, within a family | 0.79–0.86 | 2/2 | — |

- The original-prompt label is less consistent across replacement families than the scorer class. Over the 41
  R-accident masks evaluated in both families, the label differs between blur and constant on 5, the S_hi class on 1.
  - Removing slots 3-4 gives label 0 under blur but 1 under constant.
  - Removing 2-3-4-5 gives label 1 under blur (S_hi −0.71) but 0 under constant.
  - The two are different decision conditions; this is not evidence that either is the correct one.

### R-pedestrian and N-middle: no stable ranking

- **R-pedestrian:** effects are small but not all sub-resolution. In Stage C, 21 of 40 masks exceed 0.25 (max 0.96
  nats); in Stage B, 16 of 54 do (max 0.71).
- **N-middle:** effects are sub-resolution except the all-eight masks and two 4-frame blur-eval masks (0.26–0.30).

- **Rankings are unstable across methods:**

| Case | singleton vs group τ_b | top-2 overlap |
|---|---|---|
| R-pedestrian | 0.07 / 0.21 | 0 |
| N-middle | 0.21 / −0.14 | 0 |

- **Across families:** singleton τ_b is 0.64 (R-pedestrian) and 0.57 (N-middle).
- **Ranked vs random masks.** The magnitudes are similar: ≤ ~1 nat for R-pedestrian, ≤ 0.3 for N-middle.
- **R-pedestrian direction.** Ranked masks move S_hi in the direction the ranking predicts more often than random
  masks do:
  - removing the top normal-supporting frames raised S_hi in 6/8 ranked masks (mean +0.17) but 1/12 random ones (mean −0.23);
  - keeping only them lowered S_hi in 7/8 ranked masks (mean −0.53), against 3/12 random (mean +0.26).
  - This is a small, descriptive signal on the auxiliary scorer for a window the scorer itself barely calls normal.
- **R-pedestrian labels.** The original label flips under ranked and random masks alike.

### Binary-only rankings are mostly not rankable

- R-accident and N-middle have no single-frame flips in either family, so every frame ties.
- R-pedestrian under constant: all frames flip, so all tie. Under blur: two values only.

## 6. Stage D: spatial attribution (see `spatial_feasibility.md`)

- Gradients of the auxiliary score reach the pixels, the InternViT patch tokens (32×32 per frame) and the projected
  visual tokens (16×16). They are finite.
- One backward pass costs 0.99 s and 36.4 GiB, which fits only on the Ada cards. The forward path agrees with the
  scorer to 1e-7.
- **Maps:** Grad-CAM on ViT tokens, the Grad-CAM++ weighting formula applied to ViT tokens (adaptation documented),
  gradient × input, and absolute pixel gradient.
- **Occlusion:** greying the 5 % of cells Grad-CAM ranks highest drops S from 2.75 to −0.75; random cells give 1.75, 1.75 and 3.00.
- **But 23–35 % of map mass lies on the content-free letterbox bars**, which are 25 % of the area.
- The frame-level mass also disagrees with Stage B. Every map ranks slot 0 first or second, yet replacing slot 0
  removes no anomaly support: S_hi rises slightly (Delta −0.13 / −0.14) and the label stays 1.
- These are diagnostics of the auxiliary scorer only.

## 7. Answers to the brief's questions

- **Is there useful sensitivity?** For the accident window, yes, at both levels.
  - Some neighbouring-frame replacements that include the 74.7 s frame flip VERA's binary decision (7 of 26 adjacent
    pair/triple masks across both families). Single frames flip 0 of 16.
  - The numeric score gives a stable, replacement-robust ranking whose top frames outperform random ones in removal
    and retention.
  - For the likely false alert, the "sensitivity" is fragility: the label collapses under almost any change. That is
    informative but not attribution.
  - For the normal window, every single-frame and group effect is below the model's numeric resolution. Only
    replacing all eight frames moves the score by more (0.93–1.26 nats), and no label flips.
- **Does grouping add information beyond mask size?** Not established; there are suggestive signs on the accident
  window only.
  - Some adjacent pairs and triples flip the label where single frames and size-matched nonconsecutive groups never do.
  - But every flipping adjacent group has a summed single-frame effect ≥ 2.06, while every nonconsecutive control is
    ≤ 1.85. The flips are therefore consistent with crossing a decision threshold through more total evidence removed,
    not specifically adjacency.
  - Some adjacent groups with sums above 2 did not flip (blur 2-3-4; constant 3-4 and 3-4-5).
  - The adjacency-specific signal is positive non-additivity: adjacent triples average +0.31 to +0.35 against +0.01 to
    +0.03 for nonconsecutive triples. That rests on 3 controls per size.
  - Nonconsecutive pairs that include slot 4 move S_hi about as much as adjacent pairs.
  - No grouping advantage appears in the other two cases.
- **Do rankings depend on replacement choice?** Hardly on R-accident (τ_b 0.86–0.93, identical top-2). More on the
  low-signal cases (τ_b 0.57–0.79), whose effects are mostly sub-resolution.
- **Does the numeric scorer diverge from original VERA?** Partly.
  - On bf16 S, the two agree on the 4 decided baseline cases, and R-pedestrian is an exact tie (S = 0).
  - On S_hi, the scale used for all effects, R-pedestrian is −0.13, i.e. normal. That disagrees with VERA's anomaly
    (agreement 4/5), and label-only generation also says normal.
  - Under perturbation they agree on 225 of 255 distinct inputs.
  - The scorer drops the explanation step, so it is a different decision condition.
- **Are sound-question effects material here?** Yes, for the one fragile case: removing the sound question alone
  removes the R-pedestrian false alert. It also removes all four unsupported "no unusual sounds" claims. Other labels
  are unchanged; there are only 5 cases.
- **Can heatmaps be implemented within the thesis budget?** Technically yes: under 15 minutes of engineering (agent's
  estimate), and 1 s and 36 GiB per window on an Ada card. They are not validated and put substantial mass on the
  letterbox bars.
- **What remains before showing explanations to people?** See §9 and `NEXT_STEP.md`.

## 8. Concrete examples (raw text unedited; `/` marks a line break)

### R-accident (original, label 1)

> Answers to Prompt Questions: The video shows a red truck and a train crossing the same road at the same time, which is an unusual and potentially dangerous situation. There is visible damage to the train, indicating an anomaly. / Output: 1

**After replacing slots 3–4 with blur (label 0, S_hi 0.13):**

> Answers to Prompt Questions: The video shows a red truck and a train crossing the same road at the same time. This is an unusual situation as typically, vehicles and trains do not cross paths on the same road simultaneously. There is no visible damage or unusual movement, and no people or vehicles are in positions or engaging in activities that are not consistent with their usual behavior. There are no unusual sounds or noises in the video. Based on the analysis, there is no anomaly in the video. / Output: 0

**Claim level:**

- "Truck and train crossing the same road": kept.
- "Visible damage": **reversed**.
- New claim: no unusual sounds.
- Conclusion: flips to normal, matching the text's own last sentence ("there is no anomaly"). The text is internally
  inconsistent: it calls the crossing "an unusual situation" and then concludes there is no anomaly.
- Replacing slot 4 alone leaves the text essentially unchanged and the label 1, but S_hi falls from 2.85 to 1.26.

### R-pedestrian (original, label 1)

> Answers to Prompt Questions: The video shows a person crossing the road in a way that is not consistent with their usual behavior. There are no vehicles or objects in unusual positions, and no visible damage or unusual movement. There are no unusual sounds or noises. Based on the analysis, there is an anomaly in the video. / Output: 1

**After blurring slot 0 (label 0):**

> Answers to Prompt Questions: The video shows people crossing the road in a crosswalk, and there is no visible damage or unusual movement. The people are in their typical positions and engaging in activities consistent with their usual behavior. / Output: 0

**Claim level:** the crossing claim is kept; "unusual behavior" becomes "typical"; the sound claim is dropped.

### N-middle (original, label 0)

The text lists "no …" answers to all five questions, including "There are no unusual sounds or noises in the video
that suggest an anomaly." It is unchanged after blurring slots 3–4.

**After replacing all eight with grey:** the model answers in brackets with the same "no anomaly" content (label 0).
The R-accident all-grey input gives the identical text, as expected for identical pixels.

## 9. Grounding of baseline claims (agent inspection of all 40 input frames, not a human audit)

| Case | Claim | Assessment |
|---|---|---|
| R-accident | red truck and train at the same crossing | supported (slots 3–5) |
| R-accident | visible damage to the train | not assessable at 320×240. The trailer is visibly destroyed; train damage is not visible. |
| R-pedestrian | a person crossing the road | supported |
| R-pedestrian | crossing "not consistent with usual behavior" | not assessable; looked ordinary to the agent |
| R-start | red truck and white van on a road with traffic lights | supported |
| R-end | train on the tracks | supported |
| R-end | "no visible damage" | contradicted: debris from the collision is on the road in all 8 frames |
| all with sound claims | "no unusual sounds or noises" | not assessable (no audio supplied) |

No LLM judge was used. The model's self-reports are not treated as ground truth.

## 10. Limitations

- **Sample.** 3 primary windows from 2 videos, with 1 true anomaly. Masks, seeds and families are not independent
  samples, and no statistics are claimed.
- **Replacements.** Blur keeps colour blobs and layout, and grey is out of distribution. Neither is a model of a missing frame.
- **Numeric score.**
  - It is an auxiliary label-only condition, not VERA's explanation-first decision.
  - Its gate was amended after seeing data.
  - Its native resolution is 0.25 nats. S_hi refines the readout but cannot change what the bf16 model would output.
- **Frame heuristics.** Group heuristics and interactions are descriptive. Thresholds and saturation can create apparent interactions.
- **Grounding.** Visual grounding was assessed by the agent only.
- **Scope.** Attribution covers only the 8 frames the model saw. Nothing was interpolated to other frames, and VERA's
  final refined timeline was not attributed.

## 11. Decisions

1. **Original VERA binary temporal sensitivity: CONDITIONAL GO.**
   - For: on the accident window, group replacements that include the event frames flip VERA's decision where single
     frames never do. Whether adjacency matters, beyond total evidence removed, is not established here.
   - Conditions:
     - Use group-level masks with nonconsecutive controls matched on summed single-frame effect, not only on size.
     - Report label fragility (R-pedestrian) as fragility, not attribution.
     - Confirm on more windows before relying on it.
2. **Instrumented numeric temporal attribution: CONDITIONAL GO (provisional).**
   - For:
     - At baseline, the bf16 score matches the model's own first-step logits exactly. S_hi, used for effects, rounds
       back to them.
     - It is deterministic.
     - On the accident window it is robust to the replacement family and clearly better than random in removal and
       retention.
   - Conditions:
     - Pre-register S_hi and the tie rule (this run's gate was post hoc).
     - Report sub-resolution effects as such.
     - Keep it labelled as a separate target from VERA's text decision.
     - Test on more windows.
3. **Spatial heatmaps: CONDITIONAL GO for technical validation only; NO-GO for participants now.**
   - For: they are implementable and cheap.
   - Against: there is a letterbox-mass artifact, the frame-level mass disagrees with the perturbation evidence, and no
     sanity checks have been done.

**Recommended next experiment:** a pre-registered, interval-level perturbation study on 20–30 UCF-Crime windows. Details in `NEXT_STEP.md`.

## 12. Files (relative to `~/vera_research/experiments/experiment02/`; mirrored on the Mac)

- `config.yaml`, `BRIEF_v2.md`, `code/`
- `runs/run_002/`:
  - `resolved_config.yaml`, `run_manifest.json`, `environment.txt`, `preflight/`, `logs/`
  - `cases.json`, `scoring_contract.md`, `spatial_feasibility.md`, `NEXT_STEP.md`
  - `results/`:
    - `baseline_responses.jsonl`, `prompt_ablation.jsonl`, `perturbation_responses.jsonl`, `cache_records.jsonl`
    - `scorer_diagnostics.json`, `stage_a_summary.json`
    - `temporal_effects.csv`, `frame_attribution.csv`, `evaluation.csv`, `rankings_*.json`
    - `summary.json`, `costs.json`, `validation.json`, `prompts/`, `example_inputs/`
  - `spatial/`: native `.npy` maps, `spatial_summary.json`, `border_mass_check.json`, preview PNG
- `runs/run_001/`: Stage A under v1 (scorer blocked), `code_snapshot/` of the code that produced it.

Rerun on oz, from `~/vera_research`, in a new run directory:

```bash
R=experiments/experiment02/runs/run_003
setup/environment/env/bin/python experiments/experiment02/code/run_attr.py --config experiments/experiment02/config.yaml --run-dir $R --stages prep
bash setup/scripts/launch.sh --name exp02 --log-dir $R/logs --time-limit 7200 -- setup/environment/env/bin/python experiments/experiment02/code/run_attr.py --config experiments/experiment02/config.yaml --run-dir $R --stages A,B,C
bash setup/scripts/launch.sh --name exp02_D --log-dir $R/logs --time-limit 2400 -- setup/environment/env/bin/python experiments/experiment02/code/spatial_probe.py --run-dir $R
setup/environment/env/bin/python experiments/experiment02/code/analyze_attr.py --run-dir $R
setup/environment/env/bin/python experiments/experiment02/code/validate_attr.py --run-dir $R
```
