# experiment03B: appendix validation of VERA frame rankings — protocol v1.1

**Status: frozen protocol, implementation and bounded integration check only. The 290-video run has not been
launched.** Version 1.1, 2026-10-23. v1.0 was revised before any experiment03B model output, after an adversarial review; every change is listed in `AMENDMENTS.md`. The source brief is `VERA_Experiment03_Appendix_Validation_Handoff.md`
(23 October 2026). Its engineering requirements come from `VERA_Fixed_Adaptive_Workflow_Handoff.md`, whose
scientific settings are superseded. The thesis appendix text is `VERA_Appendix_Validation_Writeup.md`.

**Lineage.**
- **experiment03A.** The 30-window interval-perturbation protocol keeps its own folder and history.
- **What 3B is.** An appendix validation that builds on 3A. It reuses 3A's VERA window arithmetic, zip-listing
  parser and input-identity checks. It replaces 3A's design only for the appendix question.
- **experiment01 and experiment02.** Their records are read-only inputs. experiment02's erratum E1 applies here
  (§2.3).

## 1. Question and planned claim

VERA's local learner sees eight sampled frames and returns text plus a 0/1 anomaly label. It does not say which
frames mattered.

**Question.** Does a ranking built from changes in a *separate numerical scorer* identify frames whose replacement
changes **VERA's original generated alert** more often than a uniformly chosen pair? The study also measures how the
ranking depends on the replacement, the numerical readout and earlier removals.

**The claim, if supported, is deliberately narrow.** *For the sampled test windows and declared replacements, a
blur-derived two-frame ranking predicts VERA's original-label sensitivity under grey replacement better than chance
pair selection.*
- **Not claimed.** This is not causal ground truth, not dense video saliency, not human usefulness, and not an
  explanation of VERA's refined timeline. Ordinary frame occlusion carries no novelty claim.
- **A null result** is a valid appendix finding and is reported the same way.

## 2. Targets, model and readouts

### 2.1 Locked model and prompts

These are reused unchanged from experiment01 and experiment02:
- **Model.** InternVL2-8B revision `6fb9ad6924f69424e57fab2ab061d707688f0296`, BF16, FlashAttention 2, greedy
  decoding, `max_new_tokens` 1024.
- **Source.** VERA at commit `008ed4019781c39e7cb16244b7500244f3986f92`.
- **Prompt.** The released UCF guiding questions and the original learner prompt.
- **Parsing.** The strict parser `setup/scripts/vera_pipeline.py::strict_parse_label`.
- **Excluded.** No training, question rewriting, feature extraction, timeline smoothing, audio claim or spatial
  heatmap.

### 2.2 Three requests per input

| Target id | Prompt | Output used |
|---|---|---|
| `original_generation` | VERA's original prompt | strict-parsed label `L ∈ {0, 1, missing}` plus raw text. **Primary evaluation outcome.** |
| `nosound_generation` | the original prompt with only guiding question 5 (sound) removed | label; baseline input only (prompt-dependence diagnostic) |
| `label_only_score` | experiment02's label-only prompt, contract `label_only_first_token_v1` | native BF16 margin `S = log P('1') − log P('0')`, and `S_hi`, the FP32 difference of the two final head-row dot products on the frozen BF16 hidden state |

**The auxiliary score is not the original generated answer.** Its baseline class agreement with original VERA is
measured, not assumed. If a candidate-token identity or the score/alignment contract changes, the protocol is
versioned and the integration check re-run.

### 2.3 Readout choice (frozen before any new result)

- **Rankings use `S_hi`.** Native S records are kept separately and analysed as a sensitivity check.
- **Native margin in analysis.** It is computed as `logit_1 − logit_0`, the exact difference of the two BF16
  logits. This equals the stored `S = log P('1') − log P('0')` up to about 1e-7 of log-softmax rounding,
  which would otherwise hide exact native ties.
- **Rounding check.** Every score record checks that each FP32 head logit rounds to the model's own BF16 logit.
- **Erratum E1 (experiment02).** A sub-0.25 change in `S_hi` can cross a BF16 rounding boundary and move native S
  by a full step. So there is no "invisible below 0.25" rule anywhere in this protocol. Native differences are
  reported, and the validator reproduces E1's counts from experiment02's raw records.

## 3. Population and window manifest (frozen from metadata before inference)

**Source.** The pinned `Data/UCF_Eval.json` test manifest at `008ed401`; read it with
`git show 008ed401:Data/UCF_Eval.json` or from the historical checkout.
- **Preflight recount.** 290 videos (140 anomalous, 150 normal), 13 anomaly categories and 162 annotated intervals.
  Any difference fails the preflight until it is resolved.
- **Intervals.** They come from `temporal_label` pairs `(start, end)` with `start != -1`. `end` is clipped to the
  manifest `length` (this affects 5 intervals). An interval is the half-open range `[start, end)`, the same
  convention as experiment03A.

**VERA arithmetic** (from `vera_pipeline.anchors` and `window_for_anchor`):
- **Anchors.** `range(0, length, 16)`.
- **Window.** `[max(a − 150, 0), min(a + 150, length))`, with 30 fps hard-coded as in VERA.
- **Frames.** `np.arange(s, e, (e − s)/8).astype(int)`.
- **Clipping.** A clipped window (at the start or end) is recorded as clipped, not altered.

### 3.1 Window rules

1. **Event-centred: one per annotated interval** (≤ 162).
   - **Anchor.** The anchor nearest the interval midpoint `(start + end)/2`; on a tie, the earlier anchor.
   - **Recorded for each:** interval coverage (how many of the 8 sampled frames fall inside the interval, and
     which slots), and whether the window is clipped.
   - **Caution.** The midpoint is a sampling rule, not a statement about where the evidence is.
2. **Benchmark-negative: one per anomalous video** (≤ 140).
   - **Eligibility.** The window must be full length, `a − 150 ≥ 0` and `a + 150 ≤ length`, and must not overlap any
     interval.
   - **Choice.** The window that maximises its gap to the nearest interval; on a tie, the earlier anchor.
   - **Otherwise** the status is `unavailable`.
   - **Caution.** "Benchmark-negative" means unannotated, not necessarily semantically normal.
3. **Normal-video: one per normal video** (150). The anchor nearest `length/2`; on a tie, the earlier anchor.

This gives at most 452 window records.

**Deduplication.** An *input* is a `(video_id, frame_indices)` pair, identified by
`window_uid = "<video_id>@<anchor>"`. Window records that share an input are run once, and each stratum counts that
input once. All links to intervals and strata are kept.

**Statuses.**
- **Manifest window records:** `selected`, `unavailable` (no eligible negative window) or `duplicate_input`.
- **Inputs after retrieval** (`inputs.json`):
  - `ok`;
  - `fetch_failed`, for any acquisition or extraction failure, including missing or mismatched frames, with the
    cause in `retrieval.jsonl`;
  - `invalid_index`, when a needed frame does not decode.
- **Metadata flags.** Mismatches are recorded as flags, not statuses: `fps_not_30`, `decoded_count_mismatch` and
  `container_count_mismatch`. A flagged input still runs.

**Development videos.** `RoadAccidents127_x264` and `Normal_Videos_018_x264` are tagged `development`.
- They are excluded from the primary confirmation estimate.
- They are kept in the coverage tables and in a declared sensitivity estimate.
- Nothing else is excluded: not by size, class, difficulty or correctness.

**Real metadata checks** happen during retrieval:
- container fps, container frame count, decoded frame count, and that every needed frame index decodes;
- compared against the manifest `length` and the 30 fps that VERA assumes.

A mismatch is a reported status, never a silent substitution. The window still runs if all eight frame indices
decode. The analysis flags it and gives a sensitivity estimate without such windows.

**Frames.** Only the needed frames are extracted, using VERA's own decode loop (`cv2.VideoCapture` sequential
`read()`, `cv2.imwrite` defaults, `{:06d}.jpg`). On a development video this sparse extraction is verified to be
byte-identical to a full extraction.

### 3.2 Units

- **The statistical unit is the source video.** Masks, families and overlapping windows are repeated observations
  within a video.
- **Seeds.** They change bootstrap draws only. Cached deterministic model outputs are not replications.

### 3.3 Primary analysis population

The primary population is every unique selected input that meets all three conditions:
- it is not from a development video;
- fresh original VERA generation on it gives a parsed `L0 = 1`;
- it belongs to any stratum, pooled.

Each stratum is also shown separately.

**Also reported:**
- baseline alert rates for all selected inputs, so false negatives and unavailable attribution are visible;
- per category and stratum.

**Caution.** Event-centred alerts are not called verified true positives without a review of their frames.

## 4. Inputs and replacements

For input `x` (the eight original frames of a window) and mask set `R ⊆ {0..7}`, the replaced input is
`x_R^family`:
- **How it is built.** The listed slots of the **original** eight pixel arrays are replaced each time. There is no
  repeated blur, no dropped slot, and order and timestamps are unchanged. Slots are zero-based (0..7); source frame
  IDs and times are saved next to them.
- **Where the replacement happens.** In VERA's pixel path, after `dynamic_preprocess` (448×448) and before
  `ToTensor`/`Normalize` (experiment02 `attr_lib.slot_tensors`).
- **Identity.** The same mask set always gives the same slot-tensor hashes. Every record checks that exactly the
  masked slots differ from the baseline.

| Family | Spec (study settings, not universal standards) |
|---|---|
| `blur` | PIL `GaussianBlur(radius = 0.04 × 448 = 17.92)` on the 448×448 image. Blur can keep low-frequency evidence. |
| `grey` | constant RGB (127,127,127). It can look like a camera fault. It is the same as experiment02's `constant` family, so cache identities are shared. |

The two families' rankings are never averaged into one number.

## 5. Effects and rankings

**Class direction.** `q = +1` if `L0 = 1` and `q = −1` if `L0 = 0`. It stays locked to the baseline original label
for the whole trajectory. If `L0` is missing, target-directed attribution is marked unavailable.

**Signed original-class effect** of an additional group `G` given the masked set `R`:

`D(G | R) = q × [S_hi(x_R) − S_hi(x_(R ∪ G))]`

- The same family is used throughout, and `x_∅ = x`.
- Positive means the replacement reduced support for VERA's original class.
- Signs are kept: no absolute values, clipping or per-input normalisation.

### 5.1 Fixed ranking (primary)

> We calculate frame importance once using the unperturbed eight-frame input and freeze the resulting ranking. For example, replacing frames 3 and 4 means replacing both in the original sequence and rerunning the model. We measure that combined effect. We do not then recalculate the importance of the remaining frames. Each evaluation mask is reconstructed from the original input; perturbations do not accumulate as repeated transformations.

(The example uses human frame numbers. Internally slots are 0..7.)

- **Singleton effect.** `I_i = D({i} | ∅)` for each slot `i`, computed on blur.
- **Order.** `π` is the slots sorted by `I_i` descending; exact ties go to the lower slot index.
- **The tie-break is a convention, not evidence.** It still fixes `π`, so it also fixes the prefixes `R_k` and
  `T_w = {π_1, π_2}`. An all-tied window gets `T_w = {0, 1}`. Tied windows stay in the primary estimate under this
  rule, are flagged, and are dropped in the tie-free sensitivity estimate (§7.1).
- **Recorded per window:** all signed values, every exact tie, whether a tie straddles the cut after positions
  1, 2, 3 and 4, and the top-two margin `I_π2 − I_π3`.
- **Flags:** `all_tied` (all eight equal) and `all_nonpositive` (every `I_i ≤ 0`). Flagged windows stay in the
  population; a tied order is not presented as a confident explanation.
- **Prefixes.** The fixed prefix at budget `k` is `R_k = {π_1, …, π_k}`. It is never re-ranked.
- **Grey singleton ranking.** The same construction on grey (`I_i^grey`, `π^grey`), used only for
  replacement-sensitivity analyses.
- **Native ranking.** The same construction with native `S` in place of `S_hi`, as a readout-sensitivity analysis.

### 5.2 Adaptive ranking (secondary)

1. Start with `R = ∅`.
2. At each step, compute `D({j} | R)` on blur for **every** remaining slot `j`. The state score is `S_hi(x_R)`,
   while `q` stays from `L0`.
3. Select the remaining slot with the largest effect (exact tie → lower slot), and add it to `R`.
4. Repeat until four slots are selected.

**Saved at every state:**
- the state mask and its sequence key, and the state score;
- every candidate's mask, sequence key, cache status, score and effect;
- the ties and the selected slot;
- the model-call accounting: new adaptive calls, forwards shared with the fixed phase, cached calls and missing
  calls.
- **Shared forwards.** The first state's candidates are the blur singles and the second state's are 7 of the 28
  blur pairs, both shared with Phase 1. States 3 and 4 add 6 triples and 5 quadruples.
- **First step.** Adaptive step 1 always equals fixed `π_1`.
- **What it is not.** The adaptive deletion order is not a retention or insertion ranking.

### 5.3 No triple or quadruple sweep

There are 56 possible triples and 70 quadruples. The study evaluates *one fixed and one adaptive prefix* at each
budget and makes no claim of exhaustive selection at `k = 3, 4`.

Pairs are fully enumerated in both families. **Pair non-additivity** is
`J_ij = D({i,j} | ∅) − D({i} | ∅) − D({j} | ∅)`.
- It is a numerical non-additivity, not proof of temporal synergy.
- Any adjacent-versus-nonadjacent comparison must account for singleton effects and position, and is exploratory.

**Cross-family isolation.** Rankings and adaptive choices use blur outputs only. Grey outputs never change a blur
ranking; a validator test enforces this. The reverse direction (grey-constructed, blur-evaluated) is a secondary
numerical check only.

## 6. Requests per input

| Phase | Requests | Numerical forwards | Original generations |
|---|---|---:|---:|
| 0 baseline | label-only score of `x`; original generation of `x`; no-sound generation of `x` | 1 | 2 |
| 1 fixed | blur singles (8), grey singles (8), blur pairs (28), grey pairs (28); original generation of all 28 grey pairs; fixed grey prefixes `k = 1, 3, 4` (score + generation; `k = 2` is in the 28) | 8 + 8 + 28 + 28 + 2 | 28 + 3 |
| 2 adaptive | blur conditional candidates at states 3 and 4 (6 + 5); adaptive grey prefixes `k = 3, 4` when they differ from the fixed ones (`k = 1, 2` are shared) | 11 + ≤2 | ≤2 |
| **Upper bound** | | **88** | **35** |

- **Shared requests.** Identical requests share one cache entry: the same model, prompt, contract, generation
  settings, source, frames, mask and family give the same key, using experiment02's `request_key`.
- **Counting.** Scheduled requests and unique model calls are reported separately, as are cache hits from earlier
  runs.
- **Transformed inputs.** A transformed input is a unique (frames, mask, family) sequence; a score and a
  generation on it count as one input.
- **Phase 0 on label-unavailable inputs.** If `L0` is missing, only Phase 0 runs and the input is marked
  attribution-unavailable.

**Retries.**
- **Errors** (exceptions) are retried inside the request, at most twice, so at most 3 attempts. Attempts are
  counted per request across all logs and invocations. A device error stops the run instead.
- **Parse failures.** If a generation's parse fails, the identical input is regenerated up to two more times
  (`repeat` 1, 2). The label is the first parsed attempt; if any repeats disagree, the input is flagged
  non-deterministic.
- **Missing.** A label still missing after that is missing.

## 7. Endpoints and analysis (frozen before new results)

### 7.1 Primary endpoint: budget two, original VERA behaviour

For each primary-population input `w`:
- `T_w = {π_1, π_2}` is the top two of the fixed **blur `S_hi`** ranking.
- For each of the 28 pairs `P`, `F_w(P) = 1` if original VERA's parsed label on `x_P^grey` is `0` and
  `F_w(P) = 0` if it is `1`. A missing label stays missing.
- `d_w = F_w(T_w) − (1/28) Σ_{|P|=2} F_w(P)`.

The comparator includes `T_w` itself by design.

**Estimator.**
1. Average `d_w` over a video's non-missing inputs.
2. Average the video means over all videos with at least one non-missing `d_w`.
3. Report the result in percentage points.

**Interval: video-cluster bootstrap.**
- Videos are sorted by `video_id`. `rng = random.Random(seed)` (Python standard library).
- For each of `B = 10,000` resamples, draw `V` indices with `rng.randrange(V)` and take the mean of the chosen video
  means.
- The 95 % interval is the 2.5th and 97.5th percentiles of the sorted statistics, by linear interpolation at
  position `(B − 1)·q`.
- The seed is recorded; the default is execution seed `0`.
- Monte Carlo stability: the endpoints are recomputed with seeds `seed + 1 … seed + 4`, and their range is
  reported.

The interval is reported even if it includes zero. Masks and seeds are never reported as sample size.

**Missing data.**
- **When `d_w` is missing.** If any of the 28 grey-pair labels or `T_w`'s label is missing after retries, `d_w` is
  missing.
- **Denominators.** Full denominators are reported.
- **Worst-case bounds.**
  - Lower bound: missing selected flips are set to 0 and missing comparator flips to 1.
  - Upper bound: the reverse.
  - Both propagate through the video averaging. `T_w`'s own label, being both selected and in the comparator, takes
    opposite values in the two roles at each bound.
- **No exclusions after viewing effects.**

**Declared sensitivity estimates.** Each uses the same estimator; none replaces the primary estimate.

| Estimate | Rule |
|---|---|
| tie-free | drop inputs whose blur ranking has a tie straddling the cut after position 2 |
| scorer–VERA agreement | keep inputs whose baseline native-S class (`S > 0` → 1, `S < 0` → 0, `S = 0` undecided) equals `L0` |
| with development videos | add them back |
| metadata-clean | drop inputs with any metadata mismatch flag |

`all_tied` and `all_nonpositive` inputs are counted and kept.

### 7.2 Secondary analyses (declared in advance; descriptive unless stated)

1. **Grey numerical margin.** For every label-available input (`L0 ∈ {0, 1}`), compute
   `e_w = q·[S_hi(x) − S_hi(x_{T_w}^grey)] − (1/28) Σ_P q·[S_hi(x) − S_hi(x_P^grey)]`. It uses the same video
   aggregation and bootstrap. It is an auxiliary-score result.
2. **Fixed versus adaptive grey prefixes, `k = 1..4`.**
   - Original-label change (`L ≠ L0`) and the numerical change `q·[S_hi(x) − S_hi(x_{R_k}^grey)]`.
   - Within-video differences, adaptive − fixed.
   - The extra adaptive model calls are reported. Any adaptive advantage reflects a larger query budget and is not
     an equal-cost win.
3. **Agreement measures.**
   - Blur versus grey singleton ranking: top-two overlap `|T_w ∩ T_w^grey|` and Kendall `τ_b` of the
     `I`-vectors.
   - Native versus `S_hi` blur ranking: top-two overlap and `τ_b`.
   - Original/scorer class agreement at baseline and on the 28 grey pairs.
   - Tie counts and candidate mass.
4. **Pair non-additivity** `J_ij` for all 28 pairs in both families, and slot-selection frequencies of `T_w`.
5. **Normal-video artefact check.** `0 → 1` flips on normal-video inputs under the 28 grey pairs.
6. **Breakdowns.** By stratum and category, with denominators. Small classes get no class-level significance claim.
7. **No-sound diagnostic.** The baseline label change rate with the sound question removed. It measures prompt
   dependence, not a repair.
8. **Annotation coverage.** The distribution of annotation-slot coverage for event-centred inputs.
9. **Reciprocal numerical check.** The grey-singleton top two, compared on blur: `D_blur(T_w^grey | ∅)` minus the
   mean `D_blur(P | ∅)` over the 28 blur pairs. It is never promoted to the primary endpoint.

**Exploratory only:** comparisons of adjacent and nonadjacent pairs.

**What this study cannot establish.** No conclusion about explanation truth follows from human frame overlap, a
score shift or a deletion curve. The later human pilot asks a different question. Its clips are chosen by a
recorded rule, not because this appendix produced dramatic effects.

## 8. Reporting

`APPENDIX_RESULTS.md` is generated from the analyzer outputs, never typed by hand.

**Table A1**, in the layout of `VERA_Appendix_Validation_Writeup.md`:

| Row | Content |
|---|---|
| first | the primary confirmation estimate |
| next | event-centred, annotation-free anomaly-video and normal-video strata |
| last | each of the 13 anomaly categories (event-centred and negative inputs of that category) |

**Columns:**
- `N videos`, `N windows` (unique inputs), `N alerts` (`L0 = 1`), `N complete` (all 28 grey-pair labels present);
- top-two flip %, random-pair flip % and paired advantage (all video-weighted), with the interval in the primary
  row only;
- mean top-two blur/grey overlap.

**Rules.**
- **Definitions.** `N complete` counts alert windows with all 28 grey-pair labels. The flip percentages and the
  advantage use the alert windows whose `d_w` is defined.
- **NA.** A row with no eligible alerts gets `NA` with the reason, never a zero.
- **Coverage block.** It comes before the table: videos, window records by stratum and status (including the
  unavailable negatives), and planned versus executed inputs.
- **Prompts.** The exact question strings and the no-sound diff are saved in `results/prompts/` and named in the
  table note.
- Category and stratum rows are descriptive and exclude development videos; a separate sensitivity row adds them
  back.
- The generated table note states the model revision, prompt and scorer versions, replacements, excluded
  development videos, complete-case counts and missing bounds, and the bootstrap seed and resample count.

**Companion CSVs:** per-input metrics, all-pair records, fixed/adaptive `k = 1..4` trajectories, and the native-S
sensitivity. The table distinguishes planned, executed and failed cases.

## 9. Workflow, validation and freeze

**Workflow.**
- `code/run_workflow.py` runs the stages in dependency order: `preflight → errata → manifest → retrieval → baseline →
  fixed → adaptive → evaluation → analysis → validation → report`.
- **Supported:** dry-run (no model, no download, no writes outside a dry-run report), resume, a run lock, a stage
  ledger, failure propagation, and command argument arrays.
- **Freeze.** The config and code hashes are frozen per run; a change needs a new run ID. Seed overrides are
  accepted.
- **Limits.** Call, transformed-input and GPU-time limits are checked before each model call. A limit that is
  reached **stops the run**: the stage is failed and dependants are skipped. It is never a silent truncation.
- **Scratch.** The scratch limit is checked before each video fetch, as current scratch plus the member size.
  Fetched videos are purged after sparse extraction. Sparse frames and run records are not counted against it.
- **Locking and completeness.** One exclusive lock per run directory. `--videos` and `--shard` select subsets that
  run one at a time. Analysis requires that every planned input was retrieved and every planned request has a
  terminal record; otherwise it fails, unless `--allow-partial` is given, in which case results are flagged partial.
- **Code freeze.** It covers every result-computing file, including the analyzer, validator and table generator. A
  code change needs a new run ID, which can reuse earlier model outputs through `prior_caches`.
- **Fidelity before extraction.** In fetch mode, no frames are extracted until the sparse-extraction fidelity check
  passes on this host.
- **Saved artefacts.** Prompt strings and the no-sound diff go in `results/prompts/`. Replacement examples go in
  `results/example_inputs/`; they are images of dataset frames and stay local.

**Independent validation.** `code/validate3b.py` is a separately written, standard-library implementation. From raw
records it rebuilds:
- all pair effects, the fixed and adaptive orders, every adaptive state, every prefix mask and `d_w`;
- the primary estimate and its bootstrap;
- the input-identity checks.

It compares these with the analyzer's outputs and writes `validation.json` with concrete discrepancies.

**Synthetic tests** (`code/tests/`) cover the brief's eleven required behaviours, including:
- the analytic interaction fixture `F = 10a + 5b + c + 5ac` (fixed order `[0,2,1]`, adaptive `[0,1,2]`);
- additive and missing-label fixtures;
- grey outputs being unable to change blur rankings;
- reproducing E1 from experiment02;
- a hand-worked video-cluster aggregation.

Passing them establishes implementation correctness, not attribution truth.

## 10. Bounded integration check (authorized now)

**Cases.** experiment02's development windows R-accident (anchor 2240) and R-pedestrian (anchor 496), both
`L0 = 1` in experiment02, with seed 0 on one free RTX 6000 Ada.

**Limits:** at most 100 new transformed inputs, 0.5 GPU-hour and 500 MiB scratch.

**Coverage.** The check must cover:
- original label generation;
- all-pair enumeration;
- adaptive recomputation to `k = 4`;
- raw validation;
- generation of the table from a synthetic run.

**Caching and genuine forwards.**
- experiment02's records are reused only where the cache key is identical.
- **Retrieval rehearsal** (CPU and network only): the fetch path runs on the development video Normal_Videos_018
  with `config.retrieval_rehearsal.json`. It covers the range-fetch, the manifest identity check, sparse extraction,
  the identity file and the purge.
- At least one cached score and one cached generation are recomputed fresh and compared.
- One development video's sparse frame extraction is checked against its full frame cache.

The 290-video run is **not** launched. Its command, locked counts, measured rates and storage needs are reported in
`IMPLEMENTATION_REPORT.md` for the next execution step.

## 11. Budget (to be re-measured before launch)

At 452 inputs, with the per-input upper bound of 88 numerical forwards and 35 generations and experiment02's mean
timings (0.356 s per forward, 2.48 s per generation):
`452 × (88 × 0.356 + 35 × 2.48) / 3600 ≈ 14.8` aggregate GPU-hours. That is **about 15 core GPU-hours**, before
retrieval, failures and overhead.

**Superseded estimate.** The old 80–100 GPU-hour figure was for an exhaustive triple sweep and does not apply.

**Before launch, recompute:** the unique-call count (dry-run on the locked manifest), the measured rate
(integration check) and download/storage needs (zip listings).

## 12. Amendments

Any change to the readout, endpoint, population, masks, replacements, bootstrap or missing-data rules after v1.1 is recorded in `AMENDMENTS.md` with a date and reason, and makes a new protocol version. Results computed
under one version are never relabelled under another.

## 13. Limits

- **Replacements shift the input distribution.** Blur may keep coarse cues; grey may read as a camera fault.
  Cross-replacement evaluation reduces one narrow circularity but does not remove distribution-shift limits
  (Hooker et al. 2019; Wang & Wang 2024).
- **The auxiliary scorer is not VERA's explanation-first decision.**
- **Labels and windows.** UCF interval labels locate benchmark events, not human evidence. Event-centred windows
  are oracle-localised units, not autonomous alert discovery or video-level AUC.
- **Classes and correlation.** Some classes have very few videos (Abuse 2, Assault 3). Windows from one video are
  correlated.
- **Scope.** The refined timeline, audio and spatial evidence are out of scope.

**References:**
- VERA (CVPR 2025);
- occlusion sensitivity (Zeiler & Fergus 2014);
- meaningful perturbations (Fong & Vedaldi 2017);
- RISE (Petsiuk et al. 2018); this design is **not** RISE;
- video occlusion (Uchiyama et al. 2023);
- deletion-metric distortions (Hooker et al. 2019; Wang & Wang 2024).
